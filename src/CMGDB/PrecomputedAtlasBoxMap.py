"""Precomputed finite tagged-union box maps for :class:`CMGDB.AtlasModel`.

The ordinary :mod:`CMGDB.PrecomputedBoxMap` helper serves one Euclidean
rectangle from a dense product-lattice table.  An Atlas callback has different
semantics: its source is ``(chart_id, rectangle)`` and its value is a finite
union of tagged target rectangles.  This module provides the corresponding
lookup layer without taking a Euclidean hull or dropping empty images.

Precomputation is deliberately agnostic about how a callback constructs its
image.  In particular, it makes no enclosure or continuity claim.  It stores
and replays the callback's tagged pieces verbatim after structural validation.
"""

from __future__ import annotations

import itertools
import math
import operator
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, TypeAlias


AtlasBounds: TypeAlias = tuple[float, ...]
AtlasSource: TypeAlias = tuple[int, AtlasBounds]
AtlasSourceKey: TypeAlias = tuple[int, tuple[str, ...]]
TaggedRectangleValue: TypeAlias = tuple[int, AtlasBounds]
TaggedUnionValue: TypeAlias = tuple[TaggedRectangleValue, ...]

__all__ = [
    "AtlasLookupStats",
    "AtlasPrecomputeSummary",
    "PrecomputedAtlasBoxMap",
    "exact_atlas_source_key",
    "precompute_atlas_box_map",
]


@dataclass(frozen=True)
class AtlasLookupStats:
    """Lookup counters for a :class:`PrecomputedAtlasBoxMap`."""

    entries: int
    hits: int
    misses: int


@dataclass(frozen=True)
class AtlasPrecomputeSummary:
    """Execution facts about one bounded precomputation pass."""

    source_count: int
    batch_size: int
    evaluator_mode: str
    scalar_calls: int
    batch_calls: int
    elapsed_seconds: float
    semantics: str = (
        "verbatim tagged callback values; no continuous-image enclosure claim"
    )


def _chart_id(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} chart id must be a nonnegative integer")
    try:
        result = operator.index(value)
    except TypeError:
        raise TypeError(
            f"{label} chart id must be a nonnegative integer"
        ) from None
    if result < 0:
        raise ValueError(f"{label} chart id must be nonnegative")
    return int(result)


def _bounds(value: Any, *, label: str) -> AtlasBounds:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} bounds must be a numeric sequence")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        raise TypeError(f"{label} bounds must be a numeric sequence") from None
    if not result or len(result) % 2:
        raise ValueError(
            f"{label} bounds must contain nonempty flattened lower/upper halves"
        )
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} bounds must be finite")
    dimension = len(result) // 2
    if any(result[axis] > result[dimension + axis] for axis in range(dimension)):
        raise ValueError(f"{label} lower bound exceeds its upper bound")
    return result


def _source(value: Any) -> AtlasSource:
    if isinstance(value, (str, bytes)):
        raise TypeError("Atlas source must be (chart_id, bounds)")
    try:
        chart, bounds = value
    except (TypeError, ValueError):
        raise TypeError("Atlas source must be (chart_id, bounds)") from None
    return _chart_id(chart, label="source"), _bounds(bounds, label="source")


def exact_atlas_source_key(
    chart_id: Any,
    bounds: Sequence[float],
) -> AtlasSourceKey:
    """Return a collision-resistant exact key for one Atlas source rectangle.

    ``float.hex`` records the exact binary64 values crossing the Python/C++
    boundary.  Unlike decimal rounding, it cannot silently merge nearby
    dyadic cells.  The chart tag is part of the key.
    """

    chart = _chart_id(chart_id, label="source")
    rectangle = _bounds(bounds, label="source")
    return chart, tuple(value.hex() for value in rectangle)


def _piece(value: Any) -> TaggedRectangleValue:
    if isinstance(value, Mapping):
        if "chart_id" in value:
            chart = value["chart_id"]
        elif "chart" in value:
            chart = value["chart"]
        else:
            raise ValueError("tagged target mapping needs 'chart_id' (or 'chart')")
        if "bounds" not in value:
            raise ValueError("tagged target mapping needs 'bounds'")
        bounds = value["bounds"]
    elif hasattr(value, "chart_id") and hasattr(value, "bounds"):
        chart = value.chart_id
        bounds = value.bounds
    else:
        if isinstance(value, (str, bytes)):
            raise TypeError("tagged target must be (chart_id, bounds)")
        try:
            chart, bounds = value
        except (TypeError, ValueError):
            raise TypeError("tagged target must be (chart_id, bounds)") from None
    return _chart_id(chart, label="target"), _bounds(bounds, label="target")


def _tagged_union(value: Any) -> TaggedUnionValue:
    if value is None or isinstance(value, (str, bytes, Mapping)):
        raise TypeError("Atlas callback must return an iterable of tagged targets")
    try:
        return tuple(_piece(piece) for piece in value)
    except TypeError as error:
        if "tagged target" in str(error):
            raise
        raise TypeError(
            "Atlas callback must return an iterable of tagged targets"
        ) from None


def _public_union(value: TaggedUnionValue) -> list[tuple[int, list[float]]]:
    # Return fresh lists on every lookup.  A caller cannot corrupt the shared
    # table by mutating the ordinary list-valued Atlas callback result.
    return [(chart, list(bounds)) for chart, bounds in value]


class PrecomputedAtlasBoxMap:
    """Strict lookup implementing the ``AtlasModel.set_map`` callback API.

    Parameters
    ----------
    entries:
        Iterable of ``((source_chart, source_bounds), tagged_union)`` pairs.
        Duplicate exact source keys are rejected, even when their values agree.
    source_provenance:
        Optional iterable of ``((source_chart, source_bounds), value)`` pairs.
        Values are retained without interpretation and can be recovered with
        :meth:`provenance`.
    precompute_summary:
        Optional execution metadata produced by :func:`precompute_atlas_box_map`.

    A missing source raises ``KeyError``.  It never becomes ``[]`` because an
    explicit empty union is a meaningful open-exit image in ``AtlasModel``.
    """

    def __init__(
        self,
        entries: Iterable[tuple[Any, Any]],
        *,
        source_provenance: Optional[Iterable[tuple[Any, Any]]] = None,
        precompute_summary: Optional[AtlasPrecomputeSummary] = None,
    ) -> None:
        table: dict[AtlasSourceKey, TaggedUnionValue] = {}
        sources: dict[AtlasSourceKey, AtlasSource] = {}
        for raw_source, raw_value in entries:
            source = _source(raw_source)
            key = exact_atlas_source_key(*source)
            if key in table:
                raise ValueError(
                    "precomputed Atlas table contains an ambiguous duplicate "
                    f"source {source!r}"
                )
            table[key] = _tagged_union(raw_value)
            sources[key] = source

        provenance: dict[AtlasSourceKey, Any] = {}
        if source_provenance is not None:
            for raw_source, value in source_provenance:
                source = _source(raw_source)
                key = exact_atlas_source_key(*source)
                if key not in table:
                    raise ValueError(
                        "source provenance refers to a source absent from the "
                        "precomputed Atlas table"
                    )
                if key in provenance:
                    raise ValueError(
                        "precomputed Atlas provenance contains an ambiguous "
                        f"duplicate source {source!r}"
                    )
                provenance[key] = value

        self._table = table
        self._sources = sources
        self._provenance = provenance
        self.precompute_summary = precompute_summary
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._table)

    def __call__(
        self,
        chart_id: int,
        bounds: Sequence[float],
    ) -> list[tuple[int, list[float]]]:
        key = exact_atlas_source_key(chart_id, bounds)
        try:
            value = self._table[key]
        except KeyError:
            with self._lock:
                self._misses += 1
            raise KeyError(
                "Atlas source was not precomputed; refusing to reinterpret a "
                "cache miss as an empty/open-exit image"
            ) from None
        with self._lock:
            self._hits += 1
        return _public_union(value)

    def batch(
        self,
        sources: Iterable[tuple[int, Sequence[float]]],
    ) -> list[list[tuple[int, list[float]]]]:
        """Look up a batch of sources using the same strict semantics."""

        return [self(chart, bounds) for chart, bounds in sources]

    def provenance(self, chart_id: int, bounds: Sequence[float]) -> Any:
        """Return uninterpreted per-source provenance, or ``None`` if absent."""

        key = exact_atlas_source_key(chart_id, bounds)
        if key not in self._table:
            raise KeyError("Atlas source was not precomputed")
        return self._provenance.get(key)

    def stats(self) -> AtlasLookupStats:
        with self._lock:
            return AtlasLookupStats(len(self._table), self._hits, self._misses)

    def entries(self) -> tuple[tuple[AtlasSource, TaggedUnionValue], ...]:
        """Return immutable table entries in their original insertion order."""

        return tuple((self._sources[key], value) for key, value in self._table.items())


def precompute_atlas_box_map(
    callback: Callable[[int, Sequence[float]], Any],
    sources: Iterable[tuple[int, Sequence[float]]],
    *,
    batch_size: int = 4096,
    batch_callback: Optional[Callable[[Sequence[AtlasSource]], Sequence[Any]]] = None,
    provenance_callback: Optional[Callable[[int, Sequence[float]], Any]] = None,
) -> PrecomputedAtlasBoxMap:
    """Evaluate a finite Atlas source family in bounded, ordered chunks.

    ``batch_callback`` receives at most ``batch_size`` normalized sources and
    must return one tagged union per source, in the same order.  Without it,
    ``callback`` is called once per source.  Duplicate exact source rectangles
    fail before their second value is evaluated.

    This function does not parallelize arbitrary Python callbacks: process
    safety requires a model-specific worker factory.  It supplies the generic
    bounded batch contract that such evaluators can implement.
    """

    if isinstance(batch_size, bool):
        raise TypeError("batch_size must be a positive integer")
    try:
        chunk_size = operator.index(batch_size)
    except TypeError:
        raise TypeError("batch_size must be a positive integer") from None
    if chunk_size <= 0:
        raise ValueError("batch_size must be positive")

    started = time.perf_counter()
    iterator = iter(sources)
    seen: set[AtlasSourceKey] = set()
    entries: list[tuple[AtlasSource, TaggedUnionValue]] = []
    provenance: list[tuple[AtlasSource, Any]] = []
    scalar_calls = 0
    batch_calls = 0

    while True:
        raw_chunk = tuple(itertools.islice(iterator, chunk_size))
        if not raw_chunk:
            break
        chunk: list[AtlasSource] = []
        for raw_source in raw_chunk:
            source = _source(raw_source)
            key = exact_atlas_source_key(*source)
            if key in seen:
                raise ValueError(
                    "Atlas precompute source family contains an ambiguous "
                    f"duplicate source {source!r}"
                )
            seen.add(key)
            chunk.append(source)

        if batch_callback is None:
            raw_values = []
            for chart, bounds in chunk:
                raw_values.append(callback(chart, bounds))
                scalar_calls += 1
        else:
            raw_values = list(batch_callback(tuple(chunk)))
            batch_calls += 1
            if len(raw_values) != len(chunk):
                raise ValueError(
                    "batched Atlas callback returned the wrong number of "
                    f"values: expected {len(chunk)}, got {len(raw_values)}"
                )

        for source, raw_value in zip(chunk, raw_values):
            entries.append((source, _tagged_union(raw_value)))
            if provenance_callback is not None:
                provenance.append(
                    (source, provenance_callback(source[0], source[1]))
                )

    summary = AtlasPrecomputeSummary(
        source_count=len(entries),
        batch_size=int(chunk_size),
        evaluator_mode="batch" if batch_callback is not None else "scalar",
        scalar_calls=scalar_calls,
        batch_calls=batch_calls,
        elapsed_seconds=time.perf_counter() - started,
    )
    return PrecomputedAtlasBoxMap(
        entries,
        source_provenance=provenance if provenance_callback is not None else None,
        precompute_summary=summary,
    )
