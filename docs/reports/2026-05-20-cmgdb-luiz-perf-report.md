# CMGDB Fork And Luiz Performance Comparison Report

Date: 2026-05-20

## Summary

We made `archive/CMGDB` the canonical CMGDB fork for the latent-dynamics project, pushed it to `bernardorivas/CMGDB.git`, and added a reusable Python helper for chunked finest-lattice precomputed box maps. We then compared that fork against Luiz Ribeiro's `perf-optimizations` branch and report.

The comparison found real, relevant optimization work in Luiz's branch. The highest-value item is the C++ `MapGraph` eager adjacency cache with CSR storage, because it removes redundant `cover(F(cell))` recomputation in CMGDB's SCC and reachability passes. That optimization is complementary to our precomputed box-map helper: our helper reduces the cost of `F`, while Luiz's cache reduces how often CMGDB asks for adjacency data.

Recommended implementation order:

1. Add a regression-validating benchmark harness.
2. Add a batched rectangle API to `PrecomputedBoxMap`.
3. Expose `Model.set_batch_map(...)` in C++/pybind.
4. Port CSR adjacency caching and zero-copy adjacency views.
5. Wire the precomputed helper into the batched cache path.
6. Evaluate the 128-bit reachability mask separately.
7. Benchmark before considering parallel cache build or SCC trim.

## What We Did In Bernardo's Fork

Repository:

- Remote: `https://github.com/bernardorivas/CMGDB.git`
- Current pushed master after the first round: `08ec25171f528b9e5e07248d335b89a50155913f`
- Local checkout: `archive/CMGDB`
- Original upstream remote retained locally as `marcio`.

Implemented:

- Added `CMGDB.make_precomputed_box_map(...)`.
- Added `CMGDB.PrecomputedBoxMap` helper module.
- Supported `mode="adaptive"` and `mode="uniform"`.
- Supported chunked finest-corner lattice precomputation with `batch_points`.
- Supported framework-neutral batched NumPy-style callables.
- Supported optional `torch.nn.Module` maps.
- With `device="auto"`, Torch device preference is `mps > cuda > cpu`.
- Updated README and setup metadata for Bernardo's fork.
- Bumped package version to `1.3.3`.

Files added or modified:

- `src/CMGDB/PrecomputedBoxMap.py`
- `tests/test_precomputed_box_map.py`
- `src/CMGDB/__init__.py`
- `README.md`
- `setup.py`

Validation performed:

- `archive/CMGDB` package tests: `11 passed`
- latent-dynamics Morse consumer tests: `50 passed`

## Current Precomputed Helper Semantics

The helper currently treats the user's map as a point map `f` and precomputes images on a finest corner lattice. For an adaptive grid with dimension `d` and `subdiv_max`, it uses:

```text
M = ceil(subdiv_max / d)
n_per_axis = 2^M
corners_per_axis = n_per_axis + 1
```

For a queried rectangle, it snaps lower and upper bounds onto the finest lattice, gathers the `2^d` corner values, computes axis-wise lower and upper image bounds, and applies optional padding.

This exactly mirrors CMGDB's `BoxMap(..., mode="corners")` semantics for rectangles aligned to CMGDB's subdivision tree. It is naturally less direct for `mode="center"` or `mode="random"`:

- `center`: the sample point depends on the actual rectangle, not just its finest-grid corners.
- `random`: the sampled points are stochastic and rectangle-specific.
- Both modes can be accelerated, but exact compatibility requires per-rectangle sampling or a different documented semantics, such as aggregating samples from finest cells inside the rectangle.

## Luiz Branch And Report

Inspected sources:

- Branch: `https://github.com/luizribeiro/CMGDB/tree/perf-optimizations`
- Report: `https://public.thepromisedlan.club/~luiz/cmgdb-perf/`

Luiz's branch contains these relevant commits on top of the Bernardo-style fork lineage:

- `a54e941`: benchmark suite
- `55ca345`: eager `MapGraph` adjacency cache
- `8af85c1`: CSR adjacency cache and zero-copy `adjacencies_view`
- `744f1f7`: reach bitmask widened from `uint64_t` to `__uint128_t`
- `e1d91cf`: parallel eager cache for thread-safe maps
- `5363ad1`: GIL-split parallel cover sweep for Python-callback maps
- `2d64f97`: SCC parallel trim preprocessing

The branch also contains Nix/dev-shell changes, built-in maps, and prior UniformGrid-related work. Those are less immediately relevant to the latent-dynamics bottleneck.

## Why Luiz's Main Optimization Matters

Our current CMGDB `MapGraph` computes adjacencies on demand:

```cpp
std::vector<Vertex> target =
    grid_->cover((*f_)(grid_->geometry(source)));
```

That means SCC and reachability can ask for the same source cell's adjacency list repeatedly. If the map is Python-backed, each call crosses the C++/Python boundary. If the map is our precomputed helper, each call still crosses into Python for the lookup, even though point-map values are already cached.

Luiz's eager cache changes that shape:

1. Build all adjacency lists once when `MapGraph` is constructed.
2. Store them in CSR form:

```text
csr_offsets_[v] ... csr_offsets_[v+1]
csr_edges_[...]
```

3. SCC and reachability read zero-copy adjacency views.

This is relevant because it removes redundant CMGDB-level work after the box-map image rectangles are known. It is complementary to our helper:

- Our helper reduces cost of computing image rectangles.
- Luiz's CSR cache reduces duplicate adjacency queries.

## Why Not Merge Luiz's Branch Directly

A direct merge is not appropriate.

Observed merge issue:

- `Compute_Morse_Graph.hpp` conflicts.

Observed branch-level mismatch:

- Luiz's branch does not include our `PrecomputedBoxMap.py`.
- A straight branch replacement would delete our helper and tests.
- The diff also includes Nix, built-in maps, UniformGrid changes, and broad C++ build changes beyond the specific optimization we want.

Therefore the right strategy is selective porting or cherry-picking in slices, with benchmarks at each stage.

## Relevance Ranking

### Adopt First

**Benchmark suite**

Reason: We need output validation plus timing before touching core C++. Luiz's `tests/bench.py` is directly useful and should be adapted first.

**CSR adjacency cache**

Reason: This targets the main duplicated C++ work path and should improve both ordinary Python callbacks and our precomputed helper.

**Batched model map API**

Reason: Our helper can expose `batch(rects)` cheaply. A C++ `Model.set_batch_map(...)` path lets CMGDB pass many rectangles at once during adjacency-cache construction.

### Adopt After Measurement

**128-bit reach mask**

Reason: Low risk on GCC/Clang, but only helps scenarios with many Morse sets. Measure separately.

**GIL-split parallel cover sweep**

Reason: Potentially useful after CSR caching, especially if cover remains expensive. It is more complex and should not be bundled with the initial cache port.

### Defer

**SCC parallel trim**

Reason: Interesting, but more algorithmically invasive. It should come only after we know SCC itself is a bottleneck with cached adjacency views.

**Nix flake/dev shell**

Reason: Helpful for reproducibility in Luiz's environment, but not required for our current workflow.

**Built-in C++ example maps**

Reason: Useful for benchmarks, but not directly needed for latent-dynamics Torch/NumPy maps.

## Implementation Plan

The detailed implementation plan is:

```text
docs/superpowers/plans/2026-05-20-cmgdb-luiz-perf-integration.md
```

The plan is staged so each phase can be tested and committed independently:

1. Benchmark harness.
2. Batched precomputed helper API.
3. C++/pybind batch callback hook.
4. Serial CSR adjacency cache.
5. Batch cache path wired to the helper.
6. 128-bit reach mask evaluation.
7. Optional parallel work.
8. README/report update and push.

## Implementation Log

### Task 1: Benchmark Harness

Status: complete.

Added `tests/bench.py`, a correctness-validating benchmark harness adapted
from Luiz Ribeiro's suite. The initial default suite includes:

- `py_small`: 2D adaptive 6/10/4, expected 4 Morse vertices.
- `py_medium`: 2D adaptive 10/14/8, expected 4 Morse vertices.
- `uniform_2d`: fixed-depth 2D, expected 25 Morse vertices.
- `conley_2d`: 2D Conley-index regression matching `tests/test_basic.py`.

The script also contains optional heavy scenarios and is prepared to expose
`batch_medium` automatically once `Model.set_batch_map(...)` exists.

Validation command:

```bash
MPLCONFIGDIR=/private/tmp/mpl-cache \
  /Users/bdoprad/Work/Projects/latent-dynamics/.venv/bin/python tests/bench.py \
  --repeats 1 --warmup 0
```

Result on 2026-05-20:

```text
scenario          verts    build min  build med    compute min  compute med  compute stdev
------------------------------------------------------------------------------------------
py_small              4         0.2ms       0.2ms           2.3ms         2.3ms           0.0ms
py_medium             4         0.1ms       0.1ms           3.6ms         3.6ms           0.0ms
uniform_2d           25         0.1ms       0.1ms          20.9ms        20.9ms           0.0ms
conley_2d             4         0.1ms       0.1ms           3.2ms         3.2ms           0.0ms
```

### Task 2: Batched Precomputed Box-Map API

Status: complete.

Changed `make_precomputed_box_map(...)` to return a callable object with:

- `__call__(rect)`: preserves the existing CMGDB model callback behavior.
- `batch(rects)`: evaluates a sequence of rectangles and returns the same
  values as calling the object one rectangle at a time.

Validation:

```text
tests/test_precomputed_box_map.py: 10 passed
```

One compatibility detail: the class remains internal to the
`CMGDB.PrecomputedBoxMap` module and is not star-exported from `CMGDB`.
That avoids shadowing the existing `import CMGDB.PrecomputedBoxMap as precomputed`
module import style used by tests and downstream code.

### Task 3: Python Batch Callback Hook

Status: complete.

Added a virtual `Map::batch_map(...)` fallback, optional `ModelMapF` batch
callback storage, and Python-visible `Model.set_batch_map(...)`.

Validation:

```text
tests/test_batch_model_map.py tests/test_basic.py: 3 passed
```

Build note: reinstalling the editable package reports that
`latentdynamics 0.1.0` pins `CMGDB==1.3.2` while this fork is now `1.3.3`.
That packaging pin should be updated in latent-dynamics after this CMGDB
branch is pushed.

### Task 4: Serial CSR Adjacency Cache

Status: complete.

Added serial eager adjacency caching in `MapGraph` with CSR storage:

- `csr_offsets_`: one offset per vertex plus a sentinel.
- `csr_edges_`: flat concatenation of all adjacency lists.
- `adjacencies(v)`: preserves the public vector-returning behavior.
- `adjacencies_view(v)`: returns a non-owning span for graph algorithms.

`GraphTheory.hpp` now prefers `adjacencies_view(v)` when available, falling
back to the old `adjacencies(v)` API for other graph-like objects.

Baseline before CSR:

```text
scenario          verts    build min  build med    compute min  compute med  compute stdev
------------------------------------------------------------------------------------------
py_medium             4         0.1ms       0.1ms           3.3ms         3.4ms           0.1ms
uniform_2d           25         0.1ms       0.2ms          19.5ms        19.7ms           0.2ms
```

After serial CSR:

```text
scenario          verts    build min  build med    compute min  compute med  compute stdev
------------------------------------------------------------------------------------------
py_medium             4         0.1ms       0.1ms           3.2ms         3.2ms           0.0ms
uniform_2d           25         0.1ms       0.1ms          19.8ms        19.8ms           0.1ms
```

Validation:

```text
tests: 13 passed
```

### Task 5: Batched Cache Construction

Status: complete.

Added chunked batch construction to `MapGraph::initialize(...)` when
`f_->has_optimized_batch()` is true. The current chunk size is 100,000
source rectangles per batch callback. The batch path converts source grid
cells to geometries, calls `Map::batch_map(...)`, then covers each returned
image rectangle into the CSR staging buffer.

Additional tests:

- `test_precomputed_box_map_batch_can_be_installed_on_model`
- `test_map_graph_cache_uses_batch_map_callback`

The second test verifies that `ComputeMorseGraph(...)` actually calls the
Python batch callback while building the cached map graph.

Validation:

```text
tests: 15 passed
```

Benchmark after batch wiring:

```text
scenario          verts    build min  build med    compute min  compute med  compute stdev
------------------------------------------------------------------------------------------
batch_medium          4         0.1ms       0.1ms           3.5ms         3.7ms           0.2ms
uniform_2d           25         0.1ms       0.1ms          22.5ms        23.0ms           0.3ms
```

Interpretation: the batch path is now mechanically wired and tested, but it
does not help this tiny benchmark. The expected payoff is for expensive
Torch/NumPy callbacks where reducing Python call count dominates the extra
batch orchestration cost.

### Task 6: 128-Bit Reachability Mask

Status: complete.

Added optional `__uint128_t` reachability bitmasks for GCC/Clang builds,
falling back to `uint64_t` elsewhere. This doubles each reachability group
from 64 Morse sets to 128 Morse sets when the compiler supports it.

Added optional benchmark scenario:

- `reach_4d`: 4D adaptive Leslie-style map with 225 Morse vertices.

Baseline before 128-bit reach mask:

```text
scenario          verts    build min  build med    compute min  compute med  compute stdev
------------------------------------------------------------------------------------------
reach_4d            225         0.2ms       0.2ms         353.7ms       362.3ms           6.3ms
```

After 128-bit reach mask:

```text
scenario          verts    build min  build med    compute min  compute med  compute stdev
------------------------------------------------------------------------------------------
reach_4d            225         0.1ms       0.2ms         350.9ms       351.4ms           8.9ms
```

Validation:

```text
tests: 15 passed
```

Decision: keep this change. It is compiler-gated, correctness-preserving in
the test suite, and neutral-to-slightly faster in the benchmark that exercises
more than 64 Morse sets.

## Expected Performance Impact

The largest likely gain for our actual workflow is:

```text
PrecomputedBoxMap + Model.set_batch_map + CSR adjacency cache
```

Why:

- PrecomputedBoxMap avoids repeated Torch/NumPy point evaluation.
- Batch map avoids one Python call per rectangle during cache construction.
- CSR cache avoids repeated adjacency recomputation in SCC and reachability.

This should beat either optimization alone. The exact speedup is workload-dependent, especially on:

- number of grid vertices;
- number of Morse sets;
- average out-degree of `grid.cover(image_rect)`;
- cost of `F`;
- whether Conley-index computation dominates after graph construction.

## Open Design Questions

1. Should `make_precomputed_box_map(...)` continue returning a callable object, or should we expose an explicit class constructor as the preferred API?
2. Should `center` and `random` sampling preserve CMGDB's current per-rectangle semantics exactly, or should CMGDB document a new finest-grid aggregated sampling semantics?
3. Should the adjacency cache be enabled by default, or guarded by an opt-in flag until we have more high-out-degree neural-map benchmarks?
4. Should edge budgets be configurable from Python?

## Recommendation

Proceed with the staged plan. Do not port the SCC trim or broad parallel features until after the benchmark harness and CSR cache are in place. The next concrete commit should be the benchmark harness, because it gives us correctness and timing evidence before we change core C++ behavior.
