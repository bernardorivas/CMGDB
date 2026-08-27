"""Compact, strict checkpoints for a cached :class:`CMGDB.MapGraph`.

The native graph already stores its adjacency in compressed sparse row (CSR)
form.  This module persists that storage without expanding edges into Python
integers, lists, sets, or JSON.  A checkpoint is a small directory containing
two memory-mappable ``.npy`` arrays and canonical metadata::

    checkpoint/
      metadata.json
      offsets.npy       # little-endian int64, length V + 1
      targets.npy       # little-endian int32 or int64, length E

Rows retain the native MapGraph order.  CMGDB cover operations produce sorted,
duplicate-free rows, and the native ``csr_view`` validates that invariant
before exposing its read-only zero-copy arrays.  Empty rows are represented by
equal adjacent offsets; they are never conflated with a cemetery vertex.

This is an adjacency checkpoint only.  It makes no enclosure, continuity,
index-pair, or Conley-index claim.  Applications must keep geometry and
box-map provenance in a separate, fingerprint-bound checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import operator
import os
import shutil
import tempfile
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np


MAP_GRAPH_CSR_SCHEMA: Final = "cmgdb-mapgraph-csr-checkpoint-v1"
_FORMAT_REVISION: Final = 1
_METADATA_NAME: Final = "metadata.json"
_OFFSETS_NAME: Final = "offsets.npy"
_TARGETS_NAME: Final = "targets.npy"
_EXPECTED_DIRECTORY_ENTRIES: Final = frozenset(
    {_METADATA_NAME, _OFFSETS_NAME, _TARGETS_NAME}
)
_HASH_CHUNK_ITEMS: Final = 1 << 20
_COPY_CHUNK_ITEMS: Final = 1 << 20
_MAX_METADATA_BYTES: Final = 4 << 20
_SEMANTICS: Final = {
    "source": "exact_cached_native_MapGraph_adjacency",
    "empty_rows_preserved": True,
    "cemetery_vertex_added": False,
    "edge_values_changed": False,
    "outer_enclosure_claim_added": False,
}

__all__ = [
    "MAP_GRAPH_CSR_SCHEMA",
    "MapGraphCSR",
    "MapGraphCSRCheckpointCaps",
    "MapGraphCSRRow",
    "load_map_graph_csr_checkpoint",
    "read_map_graph_csr_metadata",
    "write_map_graph_csr_checkpoint",
]


def _exact_nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a nonnegative exact integer")
    try:
        result = operator.index(value)
    except TypeError:
        raise TypeError(f"{label} must be a nonnegative exact integer") from None
    if result < 0:
        raise ValueError(f"{label} must be nonnegative")
    return int(result)


@dataclass(frozen=True)
class MapGraphCSRCheckpointCaps:
    """Explicit resource ceilings checked before exporting or mapping arrays."""

    max_vertices: int
    max_edges: int
    max_payload_bytes: int

    def __post_init__(self) -> None:
        for name in ("max_vertices", "max_edges", "max_payload_bytes"):
            value = _exact_nonnegative_integer(getattr(self, name), label=name)
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, int]:
        return {
            "max_vertices": self.max_vertices,
            "max_edges": self.max_edges,
            "max_payload_bytes": self.max_payload_bytes,
        }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _normalize_configuration(configuration: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(configuration, Mapping):
        raise TypeError("configuration must be a JSON-compatible mapping")
    try:
        encoded = _canonical_json_bytes(dict(configuration))
        normalized = json.loads(encoded)
    except (TypeError, ValueError):
        raise TypeError(
            "configuration must contain only finite JSON-compatible values"
        ) from None
    if not isinstance(normalized, dict):  # pragma: no cover - dict above is decisive
        raise TypeError("configuration must normalize to a JSON object")
    return normalized


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_array_bytes(array: np.ndarray) -> str:
    """Hash array payload bytes in bounded views, excluding the NPY header."""

    flat = np.asarray(array).reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, int(flat.size), _HASH_CHUNK_ITEMS):
        chunk = flat[start : start + _HASH_CHUNK_ITEMS]
        if not chunk.flags.c_contiguous:
            chunk = np.ascontiguousarray(chunk)
        digest.update(memoryview(chunk).cast("B"))
    return digest.hexdigest()


def _payload_size(vertices: int, edges: int, target_itemsize: int) -> int:
    return 8 * (vertices + 1) + target_itemsize * edges


def _check_caps(
    *,
    vertices: int,
    edges: int,
    target_itemsize: int,
    caps: MapGraphCSRCheckpointCaps,
) -> int:
    payload_bytes = _payload_size(vertices, edges, target_itemsize)
    if vertices > caps.max_vertices:
        raise MemoryError(
            f"MapGraph has {vertices} vertices, above cap {caps.max_vertices}"
        )
    if edges > caps.max_edges:
        raise MemoryError(f"MapGraph has {edges} edges, above cap {caps.max_edges}")
    if payload_bytes > caps.max_payload_bytes:
        raise MemoryError(
            f"MapGraph CSR payload needs {payload_bytes} bytes, above cap "
            f"{caps.max_payload_bytes}"
        )
    return payload_bytes


def _target_dtype(vertices: int, requested: str) -> np.dtype[Any]:
    if requested not in {"auto", "int32", "int64"}:
        raise ValueError("target_dtype must be 'auto', 'int32', or 'int64'")
    int32_fits = vertices <= int(np.iinfo(np.int32).max) + 1
    if requested == "int32" and not int32_fits:
        raise OverflowError("MapGraph vertex identifiers do not fit int32")
    if requested == "int32" or (requested == "auto" and int32_fits):
        return np.dtype("<i4")
    return np.dtype("<i8")


def _write_npy_chunked(
    path: Path,
    source: np.ndarray,
    *,
    dtype: np.dtype[Any],
) -> np.memmap:
    output = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=(int(source.size),),
        fortran_order=False,
    )
    for start in range(0, int(source.size), _COPY_CHUNK_ITEMS):
        stop = min(start + _COPY_CHUNK_ITEMS, int(source.size))
        output[start:stop] = source[start:stop]
    output.flush()
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    return output


def _fingerprint_fields(metadata: Mapping[str, object]) -> dict[str, object]:
    files = metadata["files"]
    if not isinstance(files, Mapping):
        raise ValueError("CSR checkpoint file metadata is malformed")
    offsets = files["offsets"]
    targets = files["targets"]
    if not isinstance(offsets, Mapping) or not isinstance(targets, Mapping):
        raise ValueError("CSR checkpoint array metadata is malformed")
    return {
        "schema": metadata["schema"],
        "format_revision": metadata["format_revision"],
        "configuration_sha256": metadata["configuration_sha256"],
        "vertices": metadata["vertices"],
        "edges": metadata["edges"],
        "offsets_dtype": offsets["dtype"],
        "targets_dtype": targets["dtype"],
        "offsets_sha256": offsets["sha256"],
        "targets_sha256": targets["sha256"],
    }


def write_map_graph_csr_checkpoint(
    map_graph: object,
    path: str | Path,
    *,
    configuration: Mapping[str, object],
    caps: MapGraphCSRCheckpointCaps,
    target_dtype: str = "auto",
) -> Path:
    """Atomically write the exact cached MapGraph adjacency as mmap-ready CSR.

    Resource caps and cache availability are checked before requesting the
    native zero-copy view or creating any output array.  Existing targets are
    never overwritten.
    """

    if not isinstance(caps, MapGraphCSRCheckpointCaps):
        raise TypeError("caps must be a MapGraphCSRCheckpointCaps instance")
    normalized_configuration = _normalize_configuration(configuration)
    required = ("has_cache", "num_vertices", "num_cached_edges", "csr_view")
    if any(not hasattr(map_graph, name) for name in required):
        raise TypeError("map_graph does not expose the cached native CSR interface")
    if not bool(map_graph.has_cache()):
        raise RuntimeError(
            "CSR checkpoint requires CMGDB_MAPGRAPH_CACHE to be enabled"
        )
    vertices = _exact_nonnegative_integer(
        map_graph.num_vertices(), label="MapGraph vertex count"
    )
    edges = _exact_nonnegative_integer(
        map_graph.num_cached_edges(), label="MapGraph edge count"
    )
    persisted_target_dtype = _target_dtype(vertices, target_dtype)
    payload_bytes = _check_caps(
        vertices=vertices,
        edges=edges,
        target_itemsize=persisted_target_dtype.itemsize,
        caps=caps,
    )

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite CSR checkpoint {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    # csr_view returns two read-only int64 arrays backed by the immutable
    # native vectors.  Their NumPy base retains the MapGraph holder.
    native_offsets, native_targets = map_graph.csr_view()
    native_offsets = np.asarray(native_offsets)
    native_targets = np.asarray(native_targets)
    if (
        native_offsets.dtype != np.dtype(np.int64)
        or native_targets.dtype != np.dtype(np.int64)
        or native_offsets.shape != (vertices + 1,)
        or native_targets.shape != (edges,)
        or native_offsets.flags.writeable
        or native_targets.flags.writeable
    ):
        raise RuntimeError("native MapGraph returned a malformed CSR view")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    )
    try:
        offsets_output = _write_npy_chunked(
            temporary / _OFFSETS_NAME,
            native_offsets,
            dtype=np.dtype("<i8"),
        )
        targets_output = _write_npy_chunked(
            temporary / _TARGETS_NAME,
            native_targets,
            dtype=persisted_target_dtype,
        )
        offsets_hash = _sha256_array_bytes(offsets_output)
        targets_hash = _sha256_array_bytes(targets_output)
        del offsets_output, targets_output

        files = {
            "offsets": {
                "name": _OFFSETS_NAME,
                "dtype": np.dtype("<i8").str,
                "shape": [vertices + 1],
                "payload_bytes": 8 * (vertices + 1),
                "file_bytes": (temporary / _OFFSETS_NAME).stat().st_size,
                "sha256": offsets_hash,
            },
            "targets": {
                "name": _TARGETS_NAME,
                "dtype": persisted_target_dtype.str,
                "shape": [edges],
                "payload_bytes": persisted_target_dtype.itemsize * edges,
                "file_bytes": (temporary / _TARGETS_NAME).stat().st_size,
                "sha256": targets_hash,
            },
        }
        metadata: dict[str, object] = {
            "schema": MAP_GRAPH_CSR_SCHEMA,
            "format_revision": _FORMAT_REVISION,
            "checkpoint_complete": True,
            "representation": "compressed_sparse_row",
            "row_invariant": "targets_strictly_increasing_and_duplicate_free",
            "vertices": vertices,
            "edges": edges,
            "payload_bytes": payload_bytes,
            "configuration": normalized_configuration,
            "configuration_sha256": _sha256_json(normalized_configuration),
            "write_caps": caps.to_dict(),
            "semantics": dict(_SEMANTICS),
            "files": files,
        }
        fields = _fingerprint_fields(metadata)
        metadata["fingerprint"] = {
            "algorithm": "sha256",
            "fields": fields,
            "sha256": _sha256_json(fields),
        }
        metadata_path = temporary / _METADATA_NAME
        metadata_path.write_bytes(
            json.dumps(
                metadata,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        with metadata_path.open("rb") as stream:
            os.fsync(stream.fileno())
        directory = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        os.replace(temporary, target)
        parent = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def _read_metadata(path: Path) -> dict[str, object]:
    metadata_path = path / _METADATA_NAME
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise ValueError("CSR checkpoint lacks a regular metadata.json")
    if metadata_path.stat().st_size > _MAX_METADATA_BYTES:
        raise ValueError("CSR checkpoint metadata exceeds its fixed size limit")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("CSR checkpoint metadata is unreadable") from None
    if not isinstance(metadata, dict):
        raise ValueError("CSR checkpoint metadata must be a JSON object")
    return metadata


def read_map_graph_csr_metadata(path: str | Path) -> dict[str, object]:
    """Read metadata without mapping arrays or accepting it as a resume."""

    source = Path(path)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("CSR checkpoint path must be a real directory")
    entries = frozenset(item.name for item in source.iterdir())
    if entries != _EXPECTED_DIRECTORY_ENTRIES:
        raise ValueError("CSR checkpoint directory has missing or unknown files")
    return _read_metadata(source)


class MapGraphCSRRow(Collection[int]):
    """A lazy, allocation-free Python-integer view of one canonical CSR row."""

    __slots__ = ("_targets", "_begin", "_end")

    def __init__(self, targets: np.ndarray, begin: int, end: int) -> None:
        self._targets = targets
        self._begin = int(begin)
        self._end = int(end)

    def __len__(self) -> int:
        return self._end - self._begin

    def __iter__(self) -> Iterator[int]:
        for value in self._targets[self._begin : self._end]:
            yield int(value)

    def __contains__(self, value: object) -> bool:
        if isinstance(value, bool):
            return False
        try:
            target = operator.index(value)  # type: ignore[arg-type]
        except TypeError:
            return False
        row = self._targets[self._begin : self._end]
        position = int(np.searchsorted(row, target))
        return position < len(row) and int(row[position]) == target

    def as_array(self) -> np.ndarray:
        """Return the read-only mmap slice without converting its elements."""

        return self._targets[self._begin : self._end]


@dataclass(frozen=True)
class MapGraphCSR(Mapping[int, MapGraphCSRRow]):
    """Strict mmap-backed MapGraph-compatible adjacency relation."""

    offsets: np.ndarray
    targets: np.ndarray
    metadata: Mapping[str, object]
    path: Path

    def __len__(self) -> int:
        return int(self.offsets.size) - 1

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self)))

    def __getitem__(self, source: int) -> MapGraphCSRRow:
        if isinstance(source, bool):
            raise TypeError("MapGraph source must be an exact integer")
        try:
            index = operator.index(source)
        except TypeError:
            raise TypeError("MapGraph source must be an exact integer") from None
        if index < 0 or index >= len(self):
            raise IndexError("MapGraph source is outside the checkpoint")
        begin = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return MapGraphCSRRow(self.targets, begin, end)

    def num_vertices(self) -> int:
        return len(self)

    def num_cached_edges(self) -> int:
        return int(self.targets.size)

    def has_cache(self) -> bool:
        return True

    def adjacencies(self, source: int) -> MapGraphCSRRow:
        return self[source]

    @property
    def fingerprint(self) -> str:
        raw = self.metadata["fingerprint"]
        if not isinstance(raw, Mapping):  # validated by the loader
            raise RuntimeError("loaded CSR fingerprint is malformed")
        return str(raw["sha256"])


def _require_metadata_shape(metadata: Mapping[str, object]) -> None:
    required = {
        "schema",
        "format_revision",
        "checkpoint_complete",
        "representation",
        "row_invariant",
        "vertices",
        "edges",
        "payload_bytes",
        "configuration",
        "configuration_sha256",
        "write_caps",
        "semantics",
        "files",
        "fingerprint",
    }
    if set(metadata) != required:
        raise ValueError("CSR checkpoint metadata fields are not canonical")
    if (
        metadata["schema"] != MAP_GRAPH_CSR_SCHEMA
        or metadata["format_revision"] != _FORMAT_REVISION
        or metadata["checkpoint_complete"] is not True
        or metadata["representation"] != "compressed_sparse_row"
        or metadata["row_invariant"]
        != "targets_strictly_increasing_and_duplicate_free"
    ):
        raise ValueError("CSR checkpoint schema or completion claims are invalid")
    if metadata["semantics"] != _SEMANTICS:
        raise ValueError("CSR checkpoint semantics are invalid")
    write_caps = metadata["write_caps"]
    if not isinstance(write_caps, Mapping) or set(write_caps) != {
        "max_vertices",
        "max_edges",
        "max_payload_bytes",
    }:
        raise ValueError("CSR checkpoint write caps are malformed")
    for name, value in write_caps.items():
        _exact_nonnegative_integer(value, label=f"checkpoint write cap {name}")


def _validate_offsets(offsets: np.ndarray, *, vertices: int, edges: int) -> None:
    if offsets.dtype != np.dtype("<i8") or offsets.shape != (vertices + 1,):
        raise ValueError("CSR checkpoint offsets have the wrong dtype or shape")
    if offsets.flags.writeable:
        raise ValueError("CSR checkpoint offsets mapping is unexpectedly writeable")
    if int(offsets[0]) != 0 or int(offsets[-1]) != edges:
        raise ValueError("CSR checkpoint offsets have invalid endpoints")
    for start in range(0, vertices, _HASH_CHUNK_ITEMS):
        stop = min(start + _HASH_CHUNK_ITEMS, vertices)
        if np.any(offsets[start + 1 : stop + 1] < offsets[start:stop]):
            raise ValueError("CSR checkpoint offsets are not nondecreasing")


def _validate_targets(
    targets: np.ndarray,
    offsets: np.ndarray,
    *,
    vertices: int,
    edges: int,
) -> None:
    if targets.dtype not in {np.dtype("<i4"), np.dtype("<i8")}:
        raise ValueError("CSR checkpoint targets have an unsupported dtype")
    if targets.shape != (edges,) or targets.flags.writeable:
        raise ValueError("CSR checkpoint targets have the wrong shape or mutability")
    for start in range(0, edges, _HASH_CHUNK_ITEMS):
        stop = min(start + _HASH_CHUNK_ITEMS, edges)
        chunk = targets[start:stop]
        if chunk.size and (int(chunk.min()) < 0 or int(chunk.max()) >= vertices):
            raise ValueError("CSR checkpoint target is outside the vertex range")

        # Adjacent values must increase unless the second position starts a
        # new row.  Build only a bounded boolean mask for this chunk.
        compare_start = max(1, start)
        if compare_start >= stop:
            continue
        row_starts = offsets[1:-1]
        is_boundary = np.zeros(stop - compare_start, dtype=bool)
        if row_starts.size:
            first_boundary = int(np.searchsorted(row_starts, compare_start))
            last_boundary = int(np.searchsorted(row_starts, stop))
            boundary_positions = row_starts[first_boundary:last_boundary]
            is_boundary[boundary_positions - compare_start] = True
        previous = targets[compare_start - 1 : stop - 1]
        current = targets[compare_start:stop]
        if np.any((current <= previous) & ~is_boundary):
            raise ValueError(
                "CSR checkpoint row is not sorted and duplicate-free"
            )


def load_map_graph_csr_checkpoint(
    path: str | Path,
    *,
    expected_configuration: Mapping[str, object],
    caps: MapGraphCSRCheckpointCaps,
    expected_fingerprint: str | None = None,
) -> MapGraphCSR:
    """Strictly validate and memory-map an atomic CSR checkpoint for resume."""

    if not isinstance(caps, MapGraphCSRCheckpointCaps):
        raise TypeError("caps must be a MapGraphCSRCheckpointCaps instance")
    source = Path(path)
    metadata = read_map_graph_csr_metadata(source)
    _require_metadata_shape(metadata)

    configuration = metadata["configuration"]
    if not isinstance(configuration, Mapping):
        raise ValueError("CSR checkpoint configuration is malformed")
    normalized_expected = _normalize_configuration(expected_configuration)
    if dict(configuration) != normalized_expected:
        raise ValueError("CSR checkpoint configuration differs from this run")
    configuration_hash = _sha256_json(dict(configuration))
    if (
        not _is_sha256(metadata["configuration_sha256"])
        or metadata["configuration_sha256"] != configuration_hash
    ):
        raise ValueError("CSR checkpoint configuration hash is invalid")

    vertices = _exact_nonnegative_integer(
        metadata["vertices"], label="checkpoint vertex count"
    )
    edges = _exact_nonnegative_integer(metadata["edges"], label="checkpoint edge count")
    files = metadata["files"]
    if not isinstance(files, Mapping) or set(files) != {"offsets", "targets"}:
        raise ValueError("CSR checkpoint file table is malformed")
    offsets_info = files["offsets"]
    targets_info = files["targets"]
    if not isinstance(offsets_info, Mapping) or not isinstance(targets_info, Mapping):
        raise ValueError("CSR checkpoint array table is malformed")
    array_required = {"name", "dtype", "shape", "payload_bytes", "file_bytes", "sha256"}
    if set(offsets_info) != array_required or set(targets_info) != array_required:
        raise ValueError("CSR checkpoint array fields are not canonical")
    if offsets_info["name"] != _OFFSETS_NAME or targets_info["name"] != _TARGETS_NAME:
        raise ValueError("CSR checkpoint array filenames are invalid")
    try:
        offsets_dtype = np.dtype(str(offsets_info["dtype"]))
        targets_dtype = np.dtype(str(targets_info["dtype"]))
    except (TypeError, ValueError):
        raise ValueError("CSR checkpoint array dtype is invalid") from None
    if offsets_dtype != np.dtype("<i8") or targets_dtype not in {
        np.dtype("<i4"),
        np.dtype("<i8"),
    }:
        raise ValueError("CSR checkpoint array dtype is unsupported")
    for label, info in (("offsets", offsets_info), ("targets", targets_info)):
        _exact_nonnegative_integer(
            info["payload_bytes"], label=f"checkpoint {label} payload byte count"
        )
        _exact_nonnegative_integer(
            info["file_bytes"], label=f"checkpoint {label} file byte count"
        )
        if not _is_sha256(info["sha256"]):
            raise ValueError(f"CSR checkpoint {label} checksum is malformed")
    payload_bytes = _check_caps(
        vertices=vertices,
        edges=edges,
        target_itemsize=targets_dtype.itemsize,
        caps=caps,
    )
    if metadata["payload_bytes"] != payload_bytes:
        raise ValueError("CSR checkpoint payload size is inconsistent")
    if offsets_info["shape"] != [vertices + 1] or targets_info["shape"] != [edges]:
        raise ValueError("CSR checkpoint metadata shape is inconsistent")
    if offsets_info["payload_bytes"] != 8 * (vertices + 1):
        raise ValueError("CSR checkpoint offset byte count is inconsistent")
    if targets_info["payload_bytes"] != targets_dtype.itemsize * edges:
        raise ValueError("CSR checkpoint target byte count is inconsistent")

    offsets_path = source / _OFFSETS_NAME
    targets_path = source / _TARGETS_NAME
    for array_path, info in (
        (offsets_path, offsets_info),
        (targets_path, targets_info),
    ):
        if not array_path.is_file() or array_path.is_symlink():
            raise ValueError("CSR checkpoint array is not a regular file")
        if array_path.stat().st_size != info["file_bytes"]:
            raise ValueError("CSR checkpoint array file size is inconsistent")

    try:
        offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
        targets = np.load(targets_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        raise ValueError("CSR checkpoint array is unreadable") from None
    _validate_offsets(offsets, vertices=vertices, edges=edges)
    _validate_targets(targets, offsets, vertices=vertices, edges=edges)
    if _sha256_array_bytes(offsets) != offsets_info["sha256"]:
        raise ValueError("CSR checkpoint offsets checksum does not match")
    if _sha256_array_bytes(targets) != targets_info["sha256"]:
        raise ValueError("CSR checkpoint targets checksum does not match")

    fingerprint = metadata["fingerprint"]
    if not isinstance(fingerprint, Mapping) or set(fingerprint) != {
        "algorithm",
        "fields",
        "sha256",
    }:
        raise ValueError("CSR checkpoint fingerprint is malformed")
    if not _is_sha256(fingerprint["sha256"]):
        raise ValueError("CSR checkpoint fingerprint digest is malformed")
    fields = _fingerprint_fields(metadata)
    if (
        fingerprint["algorithm"] != "sha256"
        or fingerprint["fields"] != fields
        or fingerprint["sha256"] != _sha256_json(fields)
    ):
        raise ValueError("CSR checkpoint fingerprint does not match its content")
    if expected_fingerprint is not None and not _is_sha256(expected_fingerprint):
        raise ValueError("expected_fingerprint must be a lowercase SHA-256 digest")
    if expected_fingerprint is not None and fingerprint["sha256"] != expected_fingerprint:
        raise ValueError("CSR checkpoint fingerprint differs from the requested resume")

    return MapGraphCSR(
        offsets=offsets,
        targets=targets,
        metadata=metadata,
        path=source,
    )
