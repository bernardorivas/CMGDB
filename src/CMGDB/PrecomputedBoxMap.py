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
#: Grid layout of the CMGDB box decomposition being served.
Mode = Literal["adaptive", "uniform"]
#: Where inside each box the map is evaluated, mirroring
#: ``CMGDB.ComputeBoxMap.BoxMap``'s ``mode`` argument. Orthogonal to ``Mode``,
#: which selects the *grid layout* rather than the sampling rule.
EvalMode = Literal["corners", "center", "random"]

_DEFAULT_NUM_PTS = 10
_DEFAULT_SAMPLE_DEPTH = 4

_AUTO_BATCH_MIN = 4096
_AUTO_BATCH_MAX = 4 * 1024 * 1024
_AUTO_BATCH_MEMORY_FRACTION = 0.25
_MPS_PER_CHUNK_BUDGET_BYTES = 2 * 1024 * 1024 * 1024

__all__ = [
    "as_batched_evaluator",
    "evaluation_offsets",
    "make_adaptive_precomputed_box_map",
    "make_precomputed_box_map",
    "make_uniform_precomputed_box_map",
    "precompute_corner_grid",
    "resolve_batch_points",
    "select_torch_device",
]


class PrecomputedBoxMap:
    """Callable box map with a batched rectangle helper.

    ``batch_lookup``, when supplied, evaluates a whole chunk of rectangles with
    array operations rather than one NumPy call chain per rectangle. CMGDB
    routes every adjacency query through this interface, so the per-rectangle
    constant is paid millions of times; on a 2-D map the loop costs 12.3 us/rect
    against 0.48 us/rect vectorized. The scalar ``__call__`` keeps its own
    implementation so single-rectangle latency is unaffected.
    """

    def __init__(
        self,
        lookup: Callable[[Any], list[float]],
        batch_lookup: Optional[Callable[[Any], list[list[float]]]] = None,
    ):
        self._lookup = lookup
        self._batch_lookup = batch_lookup

    def __call__(self, rect: Any) -> list[float]:
        return self._lookup(rect)

    def batch(self, rects: Any) -> list[list[float]]:
        if self._batch_lookup is not None:
            return self._batch_lookup(rects)
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
    corners_per_axis: Union[int, Any],
    batch_points: BatchPoints = "auto",
    device: Any = "auto",
) -> Tuple[np.ndarray, int]:
    """Evaluate ``f`` on a product corner lattice in bounded chunks.

    ``corners_per_axis`` is either one count used for every axis, or a sequence
    giving one count per axis. The per-axis form exists because CMGDB bisects
    coordinate ``depth % dim`` at each depth, so at a subdivision level that is
    not a multiple of ``dim`` the axes are refined unequally.
    """
    lower, upper, dim = _validate_bounds(lower_bounds, upper_bounds)
    if np.isscalar(corners_per_axis):
        shape = (int(corners_per_axis),) * dim
    else:
        shape = tuple(int(c) for c in corners_per_axis)
        if len(shape) != dim:
            raise ValueError(
                f"corners_per_axis has {len(shape)} entries but dim={dim}"
            )
    if any(c < 1 for c in shape):
        raise ValueError(f"corners_per_axis must be positive; got {shape}")

    evaluator = as_batched_evaluator(f, device=device)
    evaluator_device = getattr(evaluator, "_cmgdb_torch_device", None)
    evaluator_width = getattr(evaluator, "_cmgdb_width", None)

    n_total = 1
    for c in shape:
        n_total *= c
    counts = np.asarray(shape, dtype=np.int64)
    step = np.zeros_like(upper - lower, dtype=np.float64)
    multi = counts > 1
    step[multi] = (upper - lower)[multi] / (counts[multi] - 1).astype(np.float64)

    chunk_size = resolve_batch_points(
        batch_points,
        n_total=n_total,
        input_dim=dim,
        evaluator_width=evaluator_width,
        device=evaluator_device,
    )
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


def evaluation_offsets(
    eval_mode: EvalMode,
    dim: int,
    *,
    num_pts: int = _DEFAULT_NUM_PTS,
    sample_depth: int = _DEFAULT_SAMPLE_DEPTH,
    seed: Optional[int] = 0,
) -> Tuple[np.ndarray, int]:
    """Return ``(numerators, depth)`` describing where each box is evaluated.

    Evaluation points are given as rational positions ``k / 2**depth`` along
    each axis of a box, with ``k`` an integer in ``[0, 2**depth]``. The return
    is the integer array of ``k`` values, shape ``(n_points, dim)``.

    Restricting offsets to dyadic rationals is what makes non-corner sampling
    precomputable at all. A box at depth ``t`` on an axis refined ``T`` times
    spans ``2**(T - t)`` finest cells, so an offset ``k / 2**depth`` lands on
    the lattice refined ``depth`` extra levels for *every* box depth at once.
    An arbitrary real offset lands on no shared lattice, and the whole point of
    precomputation is that boxes share their evaluation points.

    ``depth`` is therefore also the number of extra refinement levels the
    lookup table needs, which costs a factor of ``2**(dim * depth)`` in table
    size. ``corners`` needs none.
    """
    dim = int(dim)
    if dim < 1:
        raise ValueError(f"dim must be positive; got {dim}")

    if eval_mode == "corners":
        # Box vertices: offsets 0 and 1, already on the corner lattice.
        return np.array(list(itertools.product((0, 1), repeat=dim)), dtype=np.int64), 0

    if eval_mode == "center":
        # The midpoint needs exactly one extra level: at depth 1 the offset
        # 1/2 is the integer 1, and a box spanning 2**(T - t + 1) nodes of the
        # refined lattice has an integral midpoint for every t <= T.
        return np.ones((1, dim), dtype=np.int64), 1

    if eval_mode == "random":
        num_pts = int(num_pts)
        sample_depth = int(sample_depth)
        if num_pts < 1:
            raise ValueError(f"num_pts must be positive; got {num_pts}")
        if sample_depth < 1:
            raise ValueError(f"sample_depth must be positive; got {sample_depth}")
        # Drawn once, then reused by every box at every depth. Upstream
        # ``BoxMap(mode='random')`` instead calls ``np.random.uniform`` afresh
        # on each invocation, which makes its box map a non-deterministic
        # function of the rectangle and its Morse graphs unreproducible. Fixed
        # offsets are both precomputable and reproducible; the cost is that
        # sibling boxes are probed at the same relative positions.
        rng = np.random.default_rng(seed)
        return (
            rng.integers(0, 2**sample_depth + 1, size=(num_pts, dim), dtype=np.int64),
            sample_depth,
        )

    raise ValueError(
        f"eval_mode must be 'corners', 'center', or 'random'; got {eval_mode!r}"
    )


def _resolve_eval_mode(
    eval_mode: EvalMode,
    dim: int,
    padding: bool,
    *,
    num_pts: int,
    sample_depth: int,
    seed: Optional[int],
) -> Tuple[np.ndarray, int, bool]:
    """Offsets plus the padding actually in force for ``eval_mode``."""
    numerators, depth = evaluation_offsets(
        eval_mode, dim, num_pts=num_pts, sample_depth=sample_depth, seed=seed
    )
    if eval_mode == "center" and not padding:
        # One sample gives a degenerate image box, so an unpadded center map
        # encloses nothing. Upstream ``BoxMap`` forces padding here too.
        padding = True
    return numerators, depth, padding


def make_uniform_precomputed_box_map(
    f: Any,
    lower_bounds: Any,
    upper_bounds: Any,
    *,
    subdiv_max: int,
    padding: bool = True,
    eval_mode: EvalMode = "corners",
    num_pts: int = _DEFAULT_NUM_PTS,
    sample_depth: int = _DEFAULT_SAMPLE_DEPTH,
    seed: Optional[int] = 0,
    batch_points: BatchPoints = "auto",
    device: Any = "auto",
) -> Callable[[Any], list[float]]:
    """Return a precomputed ``box_map`` for a uniform CMGDB grid.

    ``eval_mode`` selects where inside each box the map is sampled, matching
    ``CMGDB.ComputeBoxMap.BoxMap``. ``center`` and ``random`` refine the table
    by ``2**(dim * depth)``; see :func:`evaluation_offsets`.
    """
    lower, upper, dim = _validate_bounds(lower_bounds, upper_bounds)
    subdiv_max = int(subdiv_max)
    if subdiv_max < 1:
        raise ValueError(f"subdiv_max must be positive; got {subdiv_max}")
    if subdiv_max % dim != 0:
        raise ValueError(
            f"uniform precomputed mode requires subdiv_max ({subdiv_max}) "
            f"divisible by dimension ({dim}); got remainder {subdiv_max % dim}"
        )

    numerators, depth, padding = _resolve_eval_mode(
        eval_mode,
        dim,
        padding,
        num_pts=num_pts,
        sample_depth=sample_depth,
        seed=seed,
    )
    scale = 1 << depth

    n_per_axis = 2 ** (subdiv_max // dim)
    # The table is refined ``depth`` extra levels so every evaluation offset is
    # a node; ``depth == 0`` reproduces the plain corner lattice exactly.
    nodes_per_axis = n_per_axis * scale + 1

    box_side = (upper - lower) / n_per_axis
    ys_grid, out_dim = precompute_corner_grid(
        f,
        lower_bounds=lower,
        upper_bounds=upper,
        corners_per_axis=nodes_per_axis,
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
        # Every box is finest here, so it spans exactly ``scale`` table nodes.
        point_indices = idx[None, :] * scale + numerators
        samples = ys_grid[tuple(point_indices[:, k] for k in range(dim))]
        out_lower = samples.min(axis=0)
        out_upper = samples.max(axis=0)
        if padding:
            box_size = rect_arr[dim:] - rect_arr[:dim]
            out_lower = out_lower - box_size
            out_upper = out_upper + box_size
        return np.concatenate([out_lower, out_upper]).tolist()

    def box_map_batch(rects: Any) -> list[list[float]]:
        R = np.asarray(rects, dtype=np.float64)
        if R.size == 0:
            return []
        R = R.reshape(-1, 2 * dim)
        center = (R[:, :dim] + R[:, dim:]) / 2.0
        idx = np.floor((center - lower) / box_side).astype(np.int64)
        np.clip(idx, 0, n_per_axis - 1, out=idx)
        point_indices = idx[:, None, :] * scale + numerators[None, :, :]
        samples = ys_grid[tuple(point_indices[..., k] for k in range(dim))]
        out_lower_b = samples.min(axis=1)
        out_upper_b = samples.max(axis=1)
        if padding:
            box_size = R[:, dim:] - R[:, :dim]
            out_lower_b = out_lower_b - box_size
            out_upper_b = out_upper_b + box_size
        return np.concatenate([out_lower_b, out_upper_b], axis=1).tolist()

    return PrecomputedBoxMap(box_map, box_map_batch)


def make_adaptive_precomputed_box_map(
    f: Any,
    lower_bounds: Any,
    upper_bounds: Any,
    *,
    subdiv_max: int,
    padding: bool = True,
    eval_mode: EvalMode = "corners",
    num_pts: int = _DEFAULT_NUM_PTS,
    sample_depth: int = _DEFAULT_SAMPLE_DEPTH,
    seed: Optional[int] = 0,
    batch_points: BatchPoints = "auto",
    device: Any = "auto",
) -> Callable[[Any], list[float]]:
    """Return a precomputed ``box_map`` for CMGDB's adaptive subdivision tree.

    ``eval_mode`` selects where inside each box the map is sampled, matching
    ``CMGDB.ComputeBoxMap.BoxMap``. Because the adaptive tree queries boxes at
    every depth, non-corner offsets are exact only on a table refined by the
    offsets' dyadic depth; see :func:`evaluation_offsets`.
    """
    lower, upper, dim = _validate_bounds(lower_bounds, upper_bounds)
    subdiv_max = int(subdiv_max)
    if subdiv_max < 1:
        raise ValueError(f"subdiv_max must be positive; got {subdiv_max}")

    # CMGDB bisects coordinate ``depth % dim`` at each depth, so after
    # ``subdiv_max`` subdivisions axis j has been split
    # ``(subdiv_max - j + dim - 1) // dim`` times. Using ceil(subdiv_max/dim) on
    # every axis over-samples all but the first whenever subdiv_max % dim != 0
    # -- in 2-D at subdiv_max=29 that is a 32769^2 table instead of
    # 32769 x 16385, i.e. 16 GiB instead of 8. Verified against CMGDB's actual
    # finest box widths.
    axis_depths = [(subdiv_max - j + dim - 1) // dim for j in range(dim)]
    n_per_axis = np.array([2**t for t in axis_depths], dtype=np.int64)

    numerators, depth, padding = _resolve_eval_mode(
        eval_mode,
        dim,
        padding,
        num_pts=num_pts,
        sample_depth=sample_depth,
        seed=seed,
    )
    scale = 1 << depth
    # ``depth`` extra levels per axis; ``depth == 0`` is the plain corner
    # lattice, so corner mode pays nothing for the generalization.
    nodes_per_axis = n_per_axis * scale + 1

    finest_box_side = (upper - lower) / n_per_axis
    ys_grid, _out_dim = precompute_corner_grid(
        f,
        lower_bounds=lower,
        upper_bounds=upper,
        corners_per_axis=nodes_per_axis.tolist(),
        batch_points=batch_points,
        device=device,
    )

    def _sample_indices(i_lower: np.ndarray, i_upper: np.ndarray) -> np.ndarray:
        """Table indices of the evaluation points of one or many boxes.

        A box spanning ``i_upper - i_lower`` finest cells spans ``scale`` times
        as many table nodes, so offset ``k / scale`` sits ``k * (i_upper -
        i_lower)`` nodes above the box's lower corner -- an integer at every
        box depth, which is the whole reason the offsets are dyadic.
        """
        span = i_upper - i_lower
        if i_lower.ndim == 1:
            return i_lower[None, :] * scale + numerators * span[None, :]
        return i_lower[:, None, :] * scale + numerators[None, :, :] * span[:, None, :]

    def box_map(rect: Any) -> list[float]:
        rect_arr = np.asarray(rect, dtype=np.float64)
        if rect_arr.shape != (2 * dim,):
            raise ValueError(f"rect must have shape ({2 * dim},); got {rect_arr.shape}")
        i_lower = np.round((rect_arr[:dim] - lower) / finest_box_side).astype(np.int64)
        i_upper = np.round((rect_arr[dim:] - lower) / finest_box_side).astype(np.int64)
        np.clip(i_lower, 0, n_per_axis, out=i_lower)
        np.clip(i_upper, 0, n_per_axis, out=i_upper)
        point_indices = _sample_indices(i_lower, i_upper)
        samples = ys_grid[tuple(point_indices[:, k] for k in range(dim))]
        out_lower = samples.min(axis=0)
        out_upper = samples.max(axis=0)
        if padding:
            box_size = rect_arr[dim:] - rect_arr[:dim]
            out_lower = out_lower - box_size
            out_upper = out_upper + box_size
        return np.concatenate([out_lower, out_upper]).tolist()

    def box_map_batch(rects: Any) -> list[list[float]]:
        R = np.asarray(rects, dtype=np.float64)
        if R.size == 0:
            return []
        R = R.reshape(-1, 2 * dim)
        i_lower = np.round((R[:, :dim] - lower) / finest_box_side).astype(np.int64)
        i_upper = np.round((R[:, dim:] - lower) / finest_box_side).astype(np.int64)
        np.clip(i_lower, 0, n_per_axis, out=i_lower)
        np.clip(i_upper, 0, n_per_axis, out=i_upper)
        point_indices = _sample_indices(i_lower, i_upper)     # (m, n_points, dim)
        samples = ys_grid[tuple(point_indices[..., k] for k in range(dim))]
        out_lower_b = samples.min(axis=1)
        out_upper_b = samples.max(axis=1)
        if padding:
            box_size = R[:, dim:] - R[:, :dim]
            out_lower_b = out_lower_b - box_size
            out_upper_b = out_upper_b + box_size
        return np.concatenate([out_lower_b, out_upper_b], axis=1).tolist()

    return PrecomputedBoxMap(box_map, box_map_batch)


def make_precomputed_box_map(
    f: Any,
    lower_bounds: Any,
    upper_bounds: Any,
    *,
    subdiv_max: int,
    mode: Mode = "adaptive",
    padding: bool = True,
    eval_mode: EvalMode = "corners",
    num_pts: int = _DEFAULT_NUM_PTS,
    sample_depth: int = _DEFAULT_SAMPLE_DEPTH,
    seed: Optional[int] = 0,
    batch_points: BatchPoints = "auto",
    device: Any = "auto",
) -> Callable[[Any], list[float]]:
    """Return a CMGDB ``box_map`` using whole-lattice precomputation.

    ``mode`` picks the grid layout being served (``adaptive`` or ``uniform``).
    ``eval_mode`` picks where inside each box the map is sampled, mirroring
    ``CMGDB.ComputeBoxMap.BoxMap``: ``corners``, ``center``, or ``random``.
    The two are independent.

    ``num_pts``, ``sample_depth``, and ``seed`` apply to ``eval_mode="random"``
    only. Sampling costs table size: ``center`` refines the lattice one level
    per axis and ``random`` refines it ``sample_depth`` levels, a factor of
    ``2**dim`` and ``2**(dim * sample_depth)`` respectively.
    """
    common = dict(
        subdiv_max=subdiv_max,
        padding=padding,
        eval_mode=eval_mode,
        num_pts=num_pts,
        sample_depth=sample_depth,
        seed=seed,
        batch_points=batch_points,
        device=device,
    )
    if mode == "adaptive":
        return make_adaptive_precomputed_box_map(
            f, lower_bounds, upper_bounds, **common
        )
    if mode == "uniform":
        return make_uniform_precomputed_box_map(
            f, lower_bounds, upper_bounds, **common
        )
    raise ValueError(f"mode must be 'adaptive' or 'uniform'; got {mode!r}")
