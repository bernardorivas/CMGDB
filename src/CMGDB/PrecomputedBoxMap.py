"""Precomputed box-map helpers for CMGDB.

These helpers evaluate a map on the finest corner lattice once, in bounded
chunks, and return a ``box_map(rect)`` callable suitable for ``CMGDB.Model``.
Torch is optional: NumPy-style batched callables work without it, while
``torch.nn.Module`` instances are evaluated on ``mps``, then ``cuda``, then
``cpu`` when ``device="auto"``.
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Callable
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np

BatchPoints = Union[int, Literal["auto"]]
Mode = Literal["adaptive", "uniform"]

_AUTO_BATCH_MIN = 4096
_AUTO_BATCH_MAX = 4 * 1024 * 1024
_AUTO_BATCH_MEMORY_FRACTION = 0.25
_MPS_PER_CHUNK_BUDGET_BYTES = 2 * 1024 * 1024 * 1024

__all__ = [
    "as_batched_evaluator",
    "make_adaptive_precomputed_box_map",
    "make_precomputed_box_map",
    "make_uniform_precomputed_box_map",
    "precompute_corner_grid",
    "resolve_batch_points",
    "select_torch_device",
]


class PrecomputedBoxMap:
    """Callable box map with a batched rectangle helper."""

    def __init__(self, lookup: Callable[[Any], list[float]]):
        self._lookup = lookup

    def __call__(self, rect: Any) -> list[float]:
        return self._lookup(rect)

    def batch(self, rects: Any) -> list[list[float]]:
        return [self._lookup(rect) for rect in rects]


def _import_torch(*, required: bool):
    try:
        import torch
    except ImportError:
        if required:
            raise RuntimeError("Torch support requires torch to be installed") from None
        return None
    return torch


def select_torch_device(device: Any = "auto"):
    """Return a ``torch.device`` using CMGDB's default preference.

    ``device="auto"`` chooses ``mps`` when available, then ``cuda``, then
    ``cpu``. Explicit unavailable accelerators raise a clear error.
    """
    torch = _import_torch(required=True)
    if hasattr(device, "type"):
        return device
    if device is None or device == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    torch_device = torch.device(device)
    if torch_device.type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("requested torch device 'mps' is not available")
    elif torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested torch device 'cuda' is not available")
    return torch_device


def _max_linear_width(module: Any) -> int:
    torch = _import_torch(required=False)
    if torch is None:
        return 1
    widths = []
    for layer in module.modules():
        if isinstance(layer, torch.nn.Linear):
            widths.extend([int(layer.in_features), int(layer.out_features)])
    return max(widths, default=1)


def _validate_batched_output(values: Any, n_points: int) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    if out.ndim == 1 and n_points == 1:
        out = out.reshape(1, -1)
    if out.ndim != 2:
        raise ValueError(
            "batched evaluator must return a 2D array with shape "
            f"(n_points, output_dim); got shape {out.shape}"
        )
    if out.shape[0] != n_points:
        raise ValueError(
            "batched evaluator returned the wrong number of rows: "
            f"expected {n_points}, got {out.shape[0]}"
        )
    return out


def as_batched_evaluator(f: Any, *, device: Any = "auto"):
    """Return a callable that maps ``(n, d)`` float64 NumPy arrays to arrays.

    Non-Torch callables are assumed to already be batched. If Torch is
    installed and ``f`` is a ``torch.nn.Module``, the returned evaluator runs
    the module in ``float32`` on ``device`` and returns ``float64`` NumPy data.
    """
    torch = _import_torch(required=False)
    if torch is not None and isinstance(f, torch.nn.Module):
        torch_device = select_torch_device(device)
        module = f.to(torch_device)
        module.eval()

        def torch_evaluator(points: np.ndarray) -> np.ndarray:
            points = np.asarray(points, dtype=np.float64)
            with torch.no_grad():
                x = torch.as_tensor(points, dtype=torch.float32, device=torch_device)
                values = module(x).detach().cpu().numpy()
            return _validate_batched_output(values, len(points))

        torch_evaluator._cmgdb_torch_device = torch_device
        torch_evaluator._cmgdb_width = _max_linear_width(module)
        return torch_evaluator

    def numpy_evaluator(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return _validate_batched_output(f(points), len(points))

    return numpy_evaluator


def _parse_slurm_mem_bytes() -> Optional[int]:
    for name in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
        raw = os.environ.get(name)
        if not raw:
            continue
        try:
            return int(raw) * 1024 * 1024
        except ValueError:
            continue
    return None


def _available_memory_bytes(device: Any = None) -> Optional[int]:
    if device is not None and getattr(device, "type", None) == "cuda":
        torch = _import_torch(required=False)
        if torch is not None:
            try:
                free, _total = torch.cuda.mem_get_info(device)
                return int(free)
            except Exception:
                pass

    slurm_bytes = _parse_slurm_mem_bytes()
    if slurm_bytes is not None:
        return slurm_bytes

    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (AttributeError, OSError, ValueError):
        return None


def resolve_batch_points(
    batch_points: BatchPoints,
    *,
    n_total: int,
    input_dim: int = 1,
    evaluator_width: Optional[int] = None,
    device: Any = None,
) -> int:
    """Resolve ``batch_points`` to a concrete chunk size."""
    if isinstance(batch_points, int) and not isinstance(batch_points, bool):
        if batch_points <= 0:
            raise ValueError(
                f"batch_points must be positive when an int; got {batch_points}"
            )
        return max(1, min(int(batch_points), int(n_total)))
    if batch_points != "auto":
        raise ValueError(
            f"batch_points must be a positive int or 'auto'; got {batch_points!r}"
        )

    width = max(int(input_dim), int(evaluator_width or 1), 1)
    bytes_per_point = max(64, 8 * width)
    available = _available_memory_bytes(device)
    if available is None:
        budget = 1024 * 1024 * 1024
    else:
        budget = int(available * _AUTO_BATCH_MEMORY_FRACTION)

    if device is not None and getattr(device, "type", None) == "mps":
        budget = min(budget, _MPS_PER_CHUNK_BUDGET_BYTES)

    chunk = budget // bytes_per_point
    chunk = max(_AUTO_BATCH_MIN, min(int(chunk), _AUTO_BATCH_MAX))
    return int(max(1, min(chunk, int(n_total))))


def _validate_bounds(lower_bounds: Any, upper_bounds: Any) -> Tuple[np.ndarray, np.ndarray, int]:
    lower = np.asarray(lower_bounds, dtype=np.float64)
    upper = np.asarray(upper_bounds, dtype=np.float64)
    if lower.ndim != 1 or upper.ndim != 1:
        raise ValueError("lower_bounds and upper_bounds must be one-dimensional")
    if lower.shape != upper.shape:
        raise ValueError("lower_bounds and upper_bounds must have the same shape")
    if lower.size == 0:
        raise ValueError("bounds must have at least one dimension")
    if np.any(lower >= upper):
        raise ValueError("each lower bound must be strictly less than its upper bound")
    return lower, upper, int(lower.size)


def precompute_corner_grid(
    f: Any,
    *,
    lower_bounds: Any,
    upper_bounds: Any,
    corners_per_axis: int,
    batch_points: BatchPoints = "auto",
    device: Any = "auto",
) -> Tuple[np.ndarray, int]:
    """Evaluate ``f`` on a product corner lattice in bounded chunks."""
    lower, upper, dim = _validate_bounds(lower_bounds, upper_bounds)
    corners_per_axis = int(corners_per_axis)
    if corners_per_axis < 1:
        raise ValueError(f"corners_per_axis must be positive; got {corners_per_axis}")

    evaluator = as_batched_evaluator(f, device=device)
    evaluator_device = getattr(evaluator, "_cmgdb_torch_device", None)
    evaluator_width = getattr(evaluator, "_cmgdb_width", None)

    n_total = corners_per_axis**dim
    if corners_per_axis > 1:
        step = (upper - lower) / float(corners_per_axis - 1)
    else:
        step = np.zeros_like(upper - lower)

    chunk_size = resolve_batch_points(
        batch_points,
        n_total=n_total,
        input_dim=dim,
        evaluator_width=evaluator_width,
        device=evaluator_device,
    )

    shape = (corners_per_axis,) * dim
    ys_flat: Optional[np.ndarray] = None
    out_dim = -1

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        flat_idx = np.arange(start, end, dtype=np.int64)
        multi_idx = np.stack(np.unravel_index(flat_idx, shape), axis=-1).astype(np.float64)
        points = lower + multi_idx * step
        values = evaluator(points)
        if ys_flat is None:
            out_dim = int(values.shape[1])
            ys_flat = np.empty((n_total, out_dim), dtype=np.float64)
        if values.shape[1] != out_dim:
            raise ValueError(
                "batched evaluator output dimension changed between chunks: "
                f"expected {out_dim}, got {values.shape[1]}"
            )
        ys_flat[start:end] = values

    if ys_flat is None:
        raise RuntimeError("corner grid unexpectedly had no points")
    return ys_flat.reshape(shape + (out_dim,)), out_dim


def _check_table_size(table_points: int, max_table_points: int, mode: str, detail: str) -> None:
    max_table_points = int(max_table_points)
    if max_table_points < 1:
        raise ValueError(f"max_table_points must be positive; got {max_table_points}")
    if table_points > max_table_points:
        raise ValueError(
            f"{mode} precomputed table size ({table_points} corners) exceeds "
            f"max_table_points ({max_table_points}). {detail}"
        )


def make_uniform_precomputed_box_map(
    f: Any,
    lower_bounds: Any,
    upper_bounds: Any,
    *,
    subdiv_max: int,
    padding: bool = True,
    batch_points: BatchPoints = "auto",
    max_table_points: int = 10_000_000,
    device: Any = "auto",
) -> Callable[[Any], list[float]]:
    """Return a precomputed ``box_map`` for a uniform CMGDB grid."""
    lower, upper, dim = _validate_bounds(lower_bounds, upper_bounds)
    subdiv_max = int(subdiv_max)
    if subdiv_max < 1:
        raise ValueError(f"subdiv_max must be positive; got {subdiv_max}")
    if subdiv_max % dim != 0:
        raise ValueError(
            f"uniform precomputed mode requires subdiv_max ({subdiv_max}) "
            f"divisible by dimension ({dim}); got remainder {subdiv_max % dim}"
        )

    n_per_axis = 2 ** (subdiv_max // dim)
    corners_per_axis = n_per_axis + 1
    table_points = corners_per_axis**dim
    _check_table_size(
        table_points,
        max_table_points,
        "uniform",
        f"For dim={dim}, subdiv_max={subdiv_max} -> {table_points} corners.",
    )

    box_side = (upper - lower) / n_per_axis
    ys_grid, out_dim = precompute_corner_grid(
        f,
        lower_bounds=lower,
        upper_bounds=upper,
        corners_per_axis=corners_per_axis,
        batch_points=batch_points,
        device=device,
    )

    def box_map(rect: Any) -> list[float]:
        rect_arr = np.asarray(rect, dtype=np.float64)
        if rect_arr.shape != (2 * dim,):
            raise ValueError(f"rect must have shape ({2 * dim},); got {rect_arr.shape}")
        center = (rect_arr[:dim] + rect_arr[dim:]) / 2.0
        idx = np.floor((center - lower) / box_side).astype(np.int64)
        idx = np.clip(idx, 0, n_per_axis - 1)
        slicer = tuple(slice(int(idx[i]), int(idx[i]) + 2) for i in range(dim))
        corners = ys_grid[slicer].reshape(2**dim, out_dim)
        out_lower = corners.min(axis=0)
        out_upper = corners.max(axis=0)
        if padding:
            box_size = rect_arr[dim:] - rect_arr[:dim]
            out_lower = out_lower - box_size
            out_upper = out_upper + box_size
        return np.concatenate([out_lower, out_upper]).tolist()

    return PrecomputedBoxMap(box_map)


def make_adaptive_precomputed_box_map(
    f: Any,
    lower_bounds: Any,
    upper_bounds: Any,
    *,
    subdiv_max: int,
    padding: bool = True,
    batch_points: BatchPoints = "auto",
    max_table_points: int = 10_000_000,
    device: Any = "auto",
) -> Callable[[Any], list[float]]:
    """Return a precomputed ``box_map`` for CMGDB's adaptive subdivision tree."""
    lower, upper, dim = _validate_bounds(lower_bounds, upper_bounds)
    subdiv_max = int(subdiv_max)
    if subdiv_max < 1:
        raise ValueError(f"subdiv_max must be positive; got {subdiv_max}")

    max_axis_depth = (subdiv_max + dim - 1) // dim
    n_per_axis = 2**max_axis_depth
    corners_per_axis = n_per_axis + 1
    table_points = corners_per_axis**dim
    _check_table_size(
        table_points,
        max_table_points,
        "adaptive",
        (
            f"For dim={dim}, subdiv_max={subdiv_max}, "
            f"ceil(subdiv_max / dim)={max_axis_depth} -> {table_points} corners."
        ),
    )

    finest_box_side = (upper - lower) / n_per_axis
    ys_grid, out_dim = precompute_corner_grid(
        f,
        lower_bounds=lower,
        upper_bounds=upper,
        corners_per_axis=corners_per_axis,
        batch_points=batch_points,
        device=device,
    )
    combos = np.array(list(itertools.product(range(2), repeat=dim)), dtype=np.int64)
    axis_idx = np.arange(dim, dtype=np.int64)

    def box_map(rect: Any) -> list[float]:
        rect_arr = np.asarray(rect, dtype=np.float64)
        if rect_arr.shape != (2 * dim,):
            raise ValueError(f"rect must have shape ({2 * dim},); got {rect_arr.shape}")
        i_lower = np.round((rect_arr[:dim] - lower) / finest_box_side).astype(np.int64)
        i_upper = np.round((rect_arr[dim:] - lower) / finest_box_side).astype(np.int64)
        np.clip(i_lower, 0, n_per_axis, out=i_lower)
        np.clip(i_upper, 0, n_per_axis, out=i_upper)
        idx_per_axis = np.stack([i_lower, i_upper], axis=0)
        corner_indices = idx_per_axis[combos, axis_idx]
        corners = ys_grid[tuple(corner_indices.T)].reshape(2**dim, out_dim)
        out_lower = corners.min(axis=0)
        out_upper = corners.max(axis=0)
        if padding:
            box_size = rect_arr[dim:] - rect_arr[:dim]
            out_lower = out_lower - box_size
            out_upper = out_upper + box_size
        return np.concatenate([out_lower, out_upper]).tolist()

    return PrecomputedBoxMap(box_map)


def make_precomputed_box_map(
    f: Any,
    lower_bounds: Any,
    upper_bounds: Any,
    *,
    subdiv_max: int,
    mode: Mode = "adaptive",
    padding: bool = True,
    batch_points: BatchPoints = "auto",
    max_table_points: int = 10_000_000,
    device: Any = "auto",
) -> Callable[[Any], list[float]]:
    """Return a CMGDB ``box_map`` using whole-lattice precomputation."""
    if mode == "adaptive":
        return make_adaptive_precomputed_box_map(
            f,
            lower_bounds,
            upper_bounds,
            subdiv_max=subdiv_max,
            padding=padding,
            batch_points=batch_points,
            max_table_points=max_table_points,
            device=device,
        )
    if mode == "uniform":
        return make_uniform_precomputed_box_map(
            f,
            lower_bounds,
            upper_bounds,
            subdiv_max=subdiv_max,
            padding=padding,
            batch_points=batch_points,
            max_table_points=max_table_points,
            device=device,
        )
    raise ValueError(f"mode must be 'adaptive' or 'uniform'; got {mode!r}")
