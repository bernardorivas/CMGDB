### PrecomputedBoxMap.py
### MIT LICENSE 2026 Marcio Gameiro

import sys
import contextlib
import itertools
import numpy as np
from CMGDB.ComputeBoxMap import _nan_image_message

class PrecomputedBoxMap:
    """Box map backed by a table of map evaluations precomputed on the corner
       lattice of the finest subdivision grid.

       The constructor evaluates the map f once at every corner node of the
       dyadic grid at depth subdiv_max, in memory-bounded chunks. After that,
       calling the object with a rectangle performs no map evaluation at all:
       the corners of any box of the subdivision tree (at any depth up to
       subdiv_max) lie on the precomputed lattice, so the image rectangle is
       a table lookup followed by a componentwise min/max -- the box image
       BoxMap(f, rect) would produce, up to rounding: a lattice node and the
       box corner TreeGrid computes for it can differ in the last bit, except
       on the faces of the domain, where both are the bounds themselves. As
       in BoxMap, a box with a sample point where f is NaN raises ValueError.

       This pays off when f is expensive (neural network surrogates, Gaussian
       processes, ODE integration): each lattice point is evaluated exactly
       once even though adjacent boxes share it and it recurs across every
       subdivision level, and the evaluation happens in large uniform batches.
       For cheap analytic maps the live BoxMap / BoxMapBatch path is simpler
       and just as fast. Memory scales with the lattice (about 2**subdiv_max
       nodes), so this suits moderate depths.

       f must map an (m, dim) NumPy array of points to an (m, dim) array of
       image points (the BoxMapBatch convention). If PyTorch is in use and f
       is a torch.nn.Module, it is evaluated in eval mode on device ('auto'
       selects mps, then cuda, then cpu; mps, which has no float64, is passed
       over for a float64 module), in the dtype of its floating-point
       parameters and buffers (float32 if it has none), and afterwards
       returned to its own device and modes; torch is never imported
       otherwise.

       Typical use:
           F = CMGDB.PrecomputedBoxMap(f, lower_bounds, upper_bounds, subdiv_max)
           model = CMGDB.Model(subdiv_min, subdiv_max, lower_bounds, upper_bounds, F)
           model.set_batch_map(F.batch)
    """

    def __init__(self, f, lower_bounds, upper_bounds, subdiv_max,
                 mode='corners', padding=False, batch_points='auto', device='auto'):
        self.lower_bounds = np.asarray(lower_bounds, dtype=float)
        self.upper_bounds = np.asarray(upper_bounds, dtype=float)
        self.dim = self.lower_bounds.shape[0]
        if self.upper_bounds.shape != (self.dim,):
            raise ValueError("lower_bounds and upper_bounds must have the same length")
        if np.any(self.upper_bounds <= self.lower_bounds):
            raise ValueError("upper_bounds must be strictly greater than lower_bounds")
        subdiv_max = int(subdiv_max)
        if subdiv_max < 1:
            raise ValueError(f"subdiv_max must be positive; got {subdiv_max}")
        self.subdiv_max = subdiv_max

        if mode == 'corners':
            # Sample each box at its 2^dim corners, which lie on the plain
            # corner lattice (scale 1)
            scale = 1
            numerators = np.array([[(k >> d) & 1 for d in range(self.dim)]
                                   for k in range(2 ** self.dim)], dtype=np.int64)
        elif mode == 'center':
            # Sample each box at its center, which lies on the lattice refined
            # by one extra level (scale 2). Center mode must pad (as in BoxMap)
            padding = True
            scale = 2
            numerators = np.array([[1] * self.dim], dtype=np.int64)
        else:
            raise ValueError("PrecomputedBoxMap supports modes 'corners' and 'center'")
        self._scale = scale
        self._numerators = numerators
        self.padding = padding

        # CMGDB bisects coordinate (depth % dim) at each depth, so after
        # subdiv_max subdivisions axis j has been split ceil((subdiv_max-j)/dim)
        # times; using the per-axis counts (rather than the max) keeps the
        # table smaller whenever subdiv_max % dim != 0
        axis_depths = [(subdiv_max - j + self.dim - 1) // self.dim for j in range(self.dim)]
        self._cells_per_axis = np.array([2 ** t for t in axis_depths], dtype=np.int64)
        self._finest_box_side = (self.upper_bounds - self.lower_bounds) / self._cells_per_axis
        nodes_per_axis = self._cells_per_axis * scale + 1
        self._nodes_per_axis = nodes_per_axis

        n_total = int(np.prod(nodes_per_axis))
        if n_total > 2 ** 31:
            raise ValueError(
                f"The corner lattice at subdiv_max={subdiv_max} has {n_total} nodes, "
                "which is too large to precompute; reduce subdiv_max or use the "
                "live BoxMap / BoxMapBatch evaluation instead")
        if batch_points == 'auto':
            chunk = min(n_total, 1 << 20)
        else:
            chunk = max(1, int(batch_points))

        step = self._finest_box_side / scale
        table = np.empty((n_total, self.dim), dtype=float)
        with self._evaluator(f, device) as evaluator:
            for start in range(0, n_total, chunk):
                stop = min(start + chunk, n_total)
                flat_idx = np.arange(start, stop, dtype=np.int64)
                multi_idx = np.stack(np.unravel_index(flat_idx, tuple(nodes_per_axis)), axis=-1)
                # The last node is upper_bounds itself, where TreeGrid puts the
                # upper faces of the boundary boxes: lower_bounds + multi_idx *
                # step rounds to either side of it on many domains (to
                # 1.2000000000000002 on [-1, 1.2], outside the domain of f)
                points = np.where(multi_idx == nodes_per_axis - 1, self.upper_bounds,
                                  self.lower_bounds + multi_idx * step)
                values = np.asarray(evaluator(points), dtype=float)
                if values.shape != (stop - start, self.dim):
                    raise ValueError(
                        f"The map must return an array of shape (m, {self.dim}); "
                        f"got {values.shape} for m={stop - start}")
                table[start:stop] = values
        self._table = table.reshape(tuple(nodes_per_axis) + (self.dim,))
        # A box image with a NaN bound raises at lookup, as in BoxMap, not
        # here: f may be undefined at nodes the model never samples
        self._table_has_nan = bool(np.isnan(table).any())

    @contextlib.contextmanager
    def _evaluator(self, f, device):
        """Yield a function evaluating f on an (m, dim) array of points."""
        # Torch is used only if the caller already imported it and f is a Module
        torch = sys.modules.get('torch')
        if torch is None or not isinstance(f, torch.nn.Module):
            yield lambda points: f(points)
            return
        # The points go in the dtype of the module's floating-point tensors
        floating = next((t for t in itertools.chain(f.parameters(), f.buffers())
                         if t.is_floating_point()), None)
        dtype = torch.float32 if floating is None else floating.dtype
        if device == 'auto':
            if torch.backends.mps.is_available() and dtype != torch.float64:
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        # Module.to and Module.eval change the caller's module in place, so
        # its device and the mode of every submodule are put back afterwards
        # (a module spread over several devices comes back on the device of
        # its first tensor)
        modes = [(submodule, submodule.training) for submodule in f.modules()]
        tensor = next(itertools.chain(f.parameters(), f.buffers()), None)
        home = None if tensor is None else tensor.device
        try:
            module = f.to(device)
            module.eval()

            def evaluator(points):
                with torch.no_grad():
                    x = torch.as_tensor(points, dtype=dtype, device=device)
                    # Widened on the CPU: MPS has no float64, NumPy no bfloat16
                    return module(x).detach().cpu().to(torch.float64).numpy()
            yield evaluator
        finally:
            if home is not None:
                f.to(home)
            for submodule, training in modes:
                submodule.training = training

    @staticmethod
    def _first_bad(bad, lower, upper, *arrays):
        """The first box with a bad coordinate, as [lower..., upper...], and
           its rows of arrays (one box, or one row per box), for error
           messages."""
        i = np.flatnonzero(np.atleast_2d(bad).any(axis=1))[0]
        box = np.concatenate([np.atleast_2d(lower)[i], np.atleast_2d(upper)[i]])
        return [box.tolist()] + [np.atleast_2d(a)[i] for a in arrays]

    def _box_indices(self, lower, upper):
        """Lattice cell indices of one or many boxes. Box bounds produced by the
           subdivision tree are dyadic up to floating point rounding, so rounding
           to the nearest cell index is exact. Any other box is an error: one
           reaching outside the domain, one narrower than a finest cell
           (typically the model subdivides deeper than subdiv_max), and one
           with bounds off the lattice, which is no box of the subdivision
           grid over [lower_bounds, upper_bounds] at any depth."""
        raw_lower = (lower - self.lower_bounds) / self._finest_box_side
        raw_upper = (upper - self.lower_bounds) / self._finest_box_side
        # The table has no image for the part of a box outside the domain.
        # Written so that NaN bounds fail the test too
        inside = (raw_lower >= -0.01) & (raw_upper <= self._cells_per_axis + 0.01)
        if not np.all(inside):
            box, = self._first_bad(~inside, lower, upper)
            raise ValueError(
                f"Box {box} reaches outside the domain [lower_bounds, "
                "upper_bounds] of the precomputed table")
        narrow = raw_upper - raw_lower < 0.99
        if np.any(narrow):
            box, = self._first_bad(narrow, lower, upper)
            raise ValueError(
                f"Box {box} is finer than the subdiv_max lattice. Is "
                f"subdiv_max={self.subdiv_max} at least the model's maximum "
                "subdivision depth?")
        i_lower = np.round(raw_lower).astype(np.int64)
        i_upper = np.round(raw_upper).astype(np.int64)
        deviation = np.maximum(np.abs(raw_lower - i_lower), np.abs(raw_upper - i_upper))
        off_lattice = deviation > 0.01
        if np.any(off_lattice):
            # A table at another depth would not help: a box at least a
            # finest cell wide with bounds off the lattice is no grid box
            box, box_deviation = self._first_bad(off_lattice, lower, upper, deviation)
            raise ValueError(
                f"Box {box} is not a box of the subdivision grid over "
                "[lower_bounds, upper_bounds]: its bounds lie "
                f"{box_deviation.max():.3g} finest cells off the lattice, so "
                "the precomputed table has no image for it (BoxMap and "
                "BoxMapBatch map any rectangle)")
        return i_lower, i_upper

    def __call__(self, rect):
        rect_arr = np.asarray(rect, dtype=float)
        i_lower, i_upper = self._box_indices(rect_arr[:self.dim], rect_arr[self.dim:])
        span = i_upper - i_lower
        # Table node indices of the box's sample points: offset k/scale of the
        # box sits k*span nodes above the box's lower corner (integer at every
        # box depth because the offsets are dyadic)
        nodes = i_lower[None, :] * self._scale + self._numerators * span[None, :]
        samples = self._table[tuple(nodes[:, d] for d in range(self.dim))]
        image_lower = samples.min(axis=0)
        image_upper = samples.max(axis=0)
        # ndarray.min propagates NaN, so every NaN sample shows in image_lower
        if self._table_has_nan and np.isnan(image_lower).any():
            raise ValueError(_nan_image_message(rect_arr))
        if self.padding:
            pad = rect_arr[self.dim:] - rect_arr[:self.dim]
            image_lower = image_lower - pad
            image_upper = image_upper + pad
        return list(image_lower) + list(image_upper)

    def batch(self, rects):
        """Evaluate the box map on many rectangles at once. Suitable for use
           with Model.set_batch_map: model.set_batch_map(F.batch)."""
        rects_arr = np.asarray(rects, dtype=float)
        i_lower, i_upper = self._box_indices(rects_arr[:, :self.dim], rects_arr[:, self.dim:])
        span = i_upper - i_lower
        nodes = (i_lower[:, None, :] * self._scale +
                 self._numerators[None, :, :] * span[:, None, :])
        samples = self._table[tuple(nodes[:, :, d] for d in range(self.dim))]
        image_lower = samples.min(axis=1)
        image_upper = samples.max(axis=1)
        if self._table_has_nan:
            nan_rows = np.isnan(image_lower).any(axis=1)
            if nan_rows.any():
                raise ValueError(_nan_image_message(rects_arr[np.argmax(nan_rows)]))
        if self.padding:
            pad = rects_arr[:, self.dim:] - rects_arr[:, :self.dim]
            image_lower = image_lower - pad
            image_upper = image_upper + pad
        return np.hstack([image_lower, image_upper])


# Fork additions (bernardorivas/CMGDB). The grid-layout factories
# make_precomputed_box_map, make_adaptive_precomputed_box_map and
# make_uniform_precomputed_box_map, and their helpers, live in
# CMGDB.precomputed_grid and are re-exported here, so
# ``from CMGDB.PrecomputedBoxMap import make_precomputed_box_map`` keeps working.
#
# The package attribute CMGDB.PrecomputedBoxMap is the class above, not this
# module, and Python resolves ``import CMGDB.PrecomputedBoxMap as P`` through
# that attribute, so P is the class too. The helpers are therefore also
# attached to the class as static methods: ``P.evaluation_offsets(...)`` and
# ``CMGDB.PrecomputedBoxMap.make_precomputed_box_map(...)``, which resolved
# when the attribute was this module, still resolve.
from CMGDB import precomputed_grid as _precomputed_grid
from CMGDB.precomputed_grid import *  # noqa: E402,F401,F403
from CMGDB.precomputed_grid import BatchPoints, EvalMode, Mode  # noqa: E402,F401

__all__ = ["PrecomputedBoxMap"] + list(_precomputed_grid.__all__)

for _name in _precomputed_grid.__all__:
    setattr(PrecomputedBoxMap, _name, staticmethod(getattr(_precomputed_grid, _name)))
for _name in ("BatchPoints", "EvalMode", "Mode"):
    setattr(PrecomputedBoxMap, _name, getattr(_precomputed_grid, _name))
del _name
