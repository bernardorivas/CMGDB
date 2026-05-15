# CMGDB — Improvement Notes

A working plan for making CMGDB faster, leaner, and easier to maintain
**without changing what it computes**. Written against the v1.3.2 source
tree at `archive/CMGDB/src/CMGDB/`.

## Hard constraint: mathematical output is fixed

Any change in this plan must preserve, bit-for-bit (modulo IEEE-754
rounding within reproducibility limits):

- The Morse graph vertex count, edge set, and vertex labels produced by
  `ComputeConleyMorseGraph` and `ComputeMorseGraph`.
- The Conley index polynomials returned in `MorseGraph::annotations`.
- The semantics of `Grid::cover`, `Map::operator()`, and
  `BoxMap`'s `mode='corners' / 'center' / 'random'` enclosures.
- The subdivision strategy parametrized by `(subdiv_init, subdiv_min,
  subdiv_max, subdiv_limit)`.

The minimum bar for any commit on this list is **a CMGDB-intrinsic
verification**: the library's own test suite, run on inputs the library
itself defines. Today that suite is the two cases in `tests/test_basic.py`
(the 4-vertex Morse graph with Conley indices `[x-1,0,0], [0,x-1,0],
[0,x-1,0], [0,0,x-1]`). That is too thin. **F1–F3 below propose the
intrinsic test fixtures the suite needs to *be* the verification bar.**
Until those land, the existing `test_basic.py` is the de facto bar.

Downstream consumers (this project's latent-dynamics pipeline, Marcio's
DSGRN-coupled work in `CMGDB_utils`, etc.) will each have their own
regression sweeps. Those are useful sanity checks for the *consumer* to
run when updating a CMGDB install, but they are not part of CMGDB's
verification contract — a CMGDB committer has no way to reproduce them
without that consumer's data and configs.

Changes that touch SCC / subdivision / chomp homology are off the
table by construction.

## Where the time goes today

From the latent-dynamics profiling (`output/leslie2d_to_2d_profile/profile_results.json`):

- `mode='corners'` BoxMap loop dominates wall-clock for trained
  neural-network maps. At `smax = 20` in 2D, the default PyTorch backend
  spends ~70% of total runtime in `box_map(rect)` calls — millions of
  scalar PyTorch forwards behind a Python `for` loop in
  `ComputeBoxMap.py:45`.
- The C++ SCC / reachability passes (`GraphTheory.hpp`) and chomp
  Conley-index passes (`ConleyIndex.h`) are not bottlenecks at these
  grid sizes.

For analytic maps with cheap evaluation, the picture flips: SCC +
chomp dominate. Most of the high-leverage improvements below target
the BoxMap path because that is what bites our project; the SCC path
is fine.

## A. BoxMap throughput

### A1. Batched `box_map(rects)` API, batched per `MorseDecomposition` node

**Architectural commitment.** Batching happens at the
`MorseDecomposition::decompose(f)` layer
(`Compute_Morse_Graph.hpp:128-138`), not per-vertex. Before invoking
`computeMorseSetsAndReachability`, collect the geometries of every
box in the node's subgrid into one batched call to the user's
Python callable. The returned per-box image rectangles are then
walked into `Grid::cover` results to build the `MapGraph`. The
existing per-vertex API path (`ModelMapF::operator()`,
`ModelMapF.h:45-72`) is preserved for users who don't opt into
batching.

**Expected effect** (extrapolated from latent-dynamics profile
data): at `smax=20` in 2D with our 2-layer 64-wide MLP latent map,
194s → ~25–40s. Roughly 5–8×. Works in **both adaptive and
uniform** mode — the natural unit of batching is the
`MorseDecomposition` node, which the algorithm produces for both
strategies. Peak extra memory is one subgrid's worth of corner
geometries and image rectangles, typically a few MB.

**Why not the alternatives we considered**:
- Per-vertex amortisation (one pybind call with 2^d corners per
  box) gives ~2× — not worth the surgery.
- Whole-phase-space precompute (uniform-mode only, mirroring our
  Python `uniform_precomputed` backend) gives ~10–20× but requires
  detecting and bypassing the per-node path entirely, and the
  corner cache can hit ~500 MB at d≥4 / deep `smax`. Out of scope
  for the C++ refactor; users wanting it can keep doing it in
  Python (as we do).

**Surface.** Add a sibling to `ModelMapF`, e.g.
`ModelMapFBatched`, keyed by a new Python parameter on
`CMGDB.Model`:

```python
CMGDB.Model(..., F=user_map, batched=False)        # current default
CMGDB.Model(..., F=user_map_batched, batched=True) # batched call
```

`MorseDecomposition::decompose(f)` checks the map type; when
batched, it calls a new `ModelMapFBatched::evaluate(rects)` pybind
entry once per `decompose()`. The output is a flat `(N, 2*d)`
contiguous array (lower bounds | upper bounds per box). SCC
iteration order does not change — only the adjacency input is
pre-materialised.

**Math invariance.** `mode='corners'` output is a deterministic
function of corner values. Batching changes only the order and
locality of the evaluations, not the values themselves.

**Effort.** Medium. New pybind binding, new `Map` subclass, refactor
of `decompose()` in `Compute_Morse_Graph.hpp`. ~300 LOC.

### A1b. DLPack zero-copy tensor handoff for GPU support

**Goal.** When the user's batched `box_map` is GPU-resident (e.g. a
PyTorch model on `cuda` or `mps`), the host roundtrip implicit in
`numpy.ndarray` ↔ pybind11 ↔ `std::vector<double>` becomes the new
bottleneck. DLPack (Python data interchange protocol, supported
natively by Torch, JAX, CuPy, TensorFlow) lets us pass a
device-memory tensor capsule between C++ and Python without copying.

**Expected effect** (extrapolated): at `smax=20` in 2D on GPU,
~25s → ~18s (one cache-warm forward pass over ~1M corners). At
`smax=24` with `d≥4`, the projected gap widens to minutes →
~tens of seconds, because corner counts scale as `(2^(k/d)+1)^d`.

**Surface.** Add an alternate `ModelMapFBatchedDLPack` that, on its
side of the pybind boundary, constructs a `DLManagedTensor` view of
the `(N, 2*d)` rect buffer and returns a `DLManagedTensor` capsule
for the image rects. On the Python side the user writes:

```python
def user_map_batched(rects_tensor):  # torch.Tensor on cuda
    X = rects_tensor.view(-1, d)     # 2^d * N rows
    Y = nn(X)                        # GPU forward
    return Y.view(N, 2*d)            # back to rect layout
```

The C++ side never touches the numerical data on the GPU. The
"select right device" logic stays on the Python side (`F` is the
user's closure; CMGDB doesn't pick).

**Catch.** Pybind11 has functional DLPack support but the
machinery for passing capsules through `Map` subclasses is not
boilerplate-free; ~2 weeks of careful work for the wiring and
fixture tests. The bug surface is exactly the "memory layout
misinterpretation" kind that tests must explicitly cover (see F1).

**Effort.** Medium-high. ~400 LOC including DLPack adapter code
and matching test fixtures.

### A2. Parallel `MorseDecomposition::decompose` across the priority queue

**Reassessed under Q1=per-node + Q2=DLPack.** Inside one
`decompose()` call, parallelism is now Python's responsibility
(via batched Torch / GPU). The remaining serial bottleneck is the
priority-queue iteration over `MorseDecomposition` nodes in
`ConstructMorseDecomposition`
(`Compute_Morse_Graph.hpp:177-229`). Nodes at the same depth are
independent: each `decompose()` only depends on its own grid.
Adjacent nodes could be processed concurrently if we partition by
depth strata.

**Catch.** The priority queue is "largest first" — a depth-stratum
parallelism reorders relative to that. Reordering does **not**
affect mathematical output (the algorithm's correctness does not
depend on which order independent nodes are processed) but it does
change which nodes might hit the `subdiv_limit` cutoff first under
some pathological size distributions. A strict comparison-mode test
fixture (see F1, "Determinism") is required before this lands.

**Effort.** Medium. ~150 LOC + an OpenMP `find_package` in
`CMakeLists.txt` + a thread-safety audit of `Grid::cover` (likely
already const-safe, needs verification).

**Verdict.** Defer until A1 + A1b ship and we measure whether
inter-node serialisation is the new bottleneck.

### A3. Specialise `Grid::cover` for fixed-depth uniform grids

**What.** `TreeGrid::cover` (`TreeGrid.h`) descends the binary
subdivision tree to find every leaf intersecting an image rectangle.
When `subdiv_init == subdiv_min == subdiv_max == k` and `k % d == 0`,
every leaf sits at depth `k` and the tree is complete — the cover
reduces to axis-wise floor/ceil arithmetic with no descent.

**Surface.** Either a `UniformCover : public TreeGrid` subclass, or
a runtime check on `tree_->depth() == fixed_k` inside the existing
`cover`. The runtime check is simpler and lower-risk.

**Math invariance.** As long as the floor/ceil convention matches
`TreeGrid::cover`'s descent semantics on the boundary case (a box
whose face lies exactly on a grid line), output is bit-identical.
See F2 for documenting and locking that convention.

**Effort.** Medium. ~200 LOC plus careful boundary testing
(deeper grids hit `int64` arithmetic edge cases, and the
out-of-domain image case has to match `TreeGrid::cover` exactly).

### A4. Per-corner deduplication in `mode='corners'`

Superseded by A1. A node-level batched call evaluates the full set
of corners in that subgrid in one shot; the user can deduplicate on
the Python side trivially (`numpy.unique` on the stacked corner
list). No C++ caching machinery needed.

**Verdict.** Skip.

## B. SCC and reachability

The current implementation (`GraphTheory.hpp`) is already
reasonable: the 64-bit-mask "Computational Groups of 64" reachability
sweep (`GraphTheory.hpp:226-307`) is a deliberate micro-optimisation.
Two minor items:

### B1. Avoid recomputing the MapGraph across subdivision passes

The hierarchical refinement loop (`Compute_Morse_Graph.hpp:177-229`)
rebuilds a `MapGraph` for each `MorseDecomposition` node. On
subdivision (`Compute_Morse_Graph.hpp:221` `child->grid()->subdivide()`),
each old box is split into `2^d` children, but **most** of the new
adjacencies are inferrable from the parent's: a parent vertex maps
to a set of parent vertices; each child vertex maps to a subset of
the children of those parents. Caching parent adjacencies and
restricting the new map evaluation to the boundary cases could
save a constant factor.

**Effort.** Medium-high. The bookkeeping is intricate and the bug
surface is exactly where math invariance is most fragile (a stale
adjacency = a wrong SCC). Defer unless A1+A2 land first and we still
need more.

## C. Memory — peak runtime at deep subdivision

The only memory item this plan addresses is **peak runtime memory
during a single CMGDB run at deep `subdiv_max`**. The other memory
levers (bookkeeping leak across runs, on-disk output size, corner
cache if A1 went whole-phase-space) are either trivial, out of
scope, or moot given the A1 architecture in §A.

The scale we care about: a `MorseDecomposition` tree at `smax=22` in
2D holds ~50–200 MB peak; projected ~5–20 GB at `smax=24` in `d=4`.
The dominant term is the **per-node `PointerGrid` retention** —
every `MorseDecomposition` node holds its grid alive until the
postorder `ConstructMorseGraph` traversal reads its decomposition
and reachability. That is the lever this section attacks.

### C1. Release per-node grids after reachability is recorded

**What.** Each `MorseDecomposition` node currently keeps its
`grid_` member alive through the entire hierarchical refinement,
even though once `decompose()` has populated
`decomposition_` (the per-Morse-set subgrids) and `reachability_`
(the inter-set edges), the parent grid is only needed to
re-construct geometry strings in `ConstructMorseGraph`. If we
serialise the minimum needed information into compact form (box
indices + box geometries for the Morse sets that survive) and drop
the raw `PointerGrid` before recursing into children, the peak
memory at any moment of the algorithm is bounded by the *current
depth* slice of the tree, not the *total* tree.

**Catch.** The `decomposition_` member already holds Grid pointers
to the SCC subgrids (`Compute_Morse_Graph.hpp:154`); those are the
same `PointerGrid` instances under the hood, so dropping the parent
doesn't free their memory by itself. To actually release memory,
each `decompose()` output subgrid must either (a) be allocated as
its own self-contained `PointerGrid` (no sharing of tree nodes with
the parent — which is what `subgrid()` already does at the cost of
one extra allocation), or (b) be eagerly converted to a
`CompressedTreeGrid` / `SuccinctGrid` once it's no longer being
subdivided.

Path (b) is the bigger win — `CompressedTreeGrid` stores the same
geometry in ~½ the memory and supports `cover()` queries. The
downside is that the *next* `decompose()` of a child cannot resume
subdivision on a `CompressedTreeGrid` directly; it would
re-inflate to a `PointerGrid` before refining. That's an
allocation cost the user pays, but only for nodes that survive to
become children — the spurious-decomposition branches get freed
without re-inflation.

**Surface.** Inside `MorseDecomposition::spawn()`
(`Compute_Morse_Graph.hpp:139-149`), after children are created
from the subdivided subgrids, compact the *parent's* surviving
subgrids to `CompressedTreeGrid` and release the parent's full
grid. Spurious nodes drop their grids entirely.

**Math invariance.** No effect on outputs — only on which
allocator owns the bytes at which time. Test fixtures: F1
determinism + a memory-pressure regression run at known depths
(reading `/proc/self/statm` peak RSS or `getrusage` `ru_maxrss`).

**Effort.** High. The intricate part is the
`PointerGrid → CompressedTreeGrid → PointerGrid` round-trip when
refining a survived subgrid; bug surface is exactly where math
invariance breaks if box-index correspondence drifts during the
compact/inflate cycle. ~600 LOC including a memory-regression
fixture.

### C2. Drop `MEMORYBOOKKEEPING` from the default build

Currently `CMGDB.cpp:12` defines `MEMORYBOOKKEEPING` unconditionally.
The bookkeeping updates two process-global counters
(`Compute_Morse_Graph.hpp:25-27, 73-76`) in every
`MorseDecomposition` constructor. Cost is small *but* the counters
never reset — over a long-lived Python session running many CMGDB
calls, they accumulate, and any code path that reads them sees
stale (cumulative) values. This is a correctness wart, not a peak-
memory issue; included here only because (a) it's an obvious cleanup
in the same area as C1, and (b) C1's memory-regression fixture
needs the bookkeeping reliably resettable per-run.

**Surface.** Make `MEMORYBOOKKEEPING` a CMake option (defaults
off), and if defined, reset the counters at the entry of
`Compute_Morse_Graph` instead of leaving them as cross-run globals.

**Effort.** Trivial. ~10 LOC + CMakeLists.

## D. API hygiene

### D1. Move `ModelMap` out of the core

`ModelMap` (`ModelMap.h:13-69`) is a **hardcoded 2D Leslie map**
(`y0 = (p0*x0 + p1*x1) * exp(a*(x0+x1))`, `y1 = b*x0`). It is dead
weight in the main library — every real user provides their own map
via `ModelMapF`. Move to `examples/` as a demo, drop the binding,
remove the dependency from `Model.h`.

**Math invariance.** Trivial — no user of `Model(...)` with a Python
`F` ever hits `ModelMap`.

**Effort.** Low.

### D2. Consolidate the five compute entry points

Today we have:

- `Compute_Morse_Graph(...)` — the template, the actual algorithm.
- `ComputeMorseGraph(model)` — Python entry, no Conley.
- `ComputeConleyMorseGraph(model)` — Python entry, with Conley.
- `computeMorseGraph(...)` — helper used only by the legacy paths.
- `MorseGraphIntvalMap(...)`, `MorseGraphMap(...)` — legacy entry
  points that also write `SingleCMG_statistics.txt` to CWD as a side
  effect.

Keep the first three; deprecate the last three. The
`SingleCMG_statistics.txt` side effect is a project-rude surprise.

### D3. Make `SingleCMG_statistics.txt` opt-in and path-configurable

If we keep it at all, an explicit `stats_file_path: Optional[str]`
parameter beats writing to CWD by default.

### D4. Pybind type stubs

Currently editors get no autocomplete for `CMGDB.Model`,
`MorseGraph`, etc. A generated `_cmgdb.pyi` covers
`num_vertices()`, `vertices()`, `adjacencies()`, `morse_set_boxes()`,
`annotations()`, `Model.__init__` overloads. Either hand-written or
via `pybind11-stubgen`.

**Effort.** Low. Hand-written stubs ~150 LOC.

## E. Build & dev experience

### E1. CMake option `-DCMG_VERBOSE=ON`

Today verbose mode requires editing `CMGDB.cpp:11`. Move to a
CMakeLists option:

```cmake
option(CMG_VERBOSE "Enable per-run progress messages" OFF)
if(CMG_VERBOSE)
    target_compile_definitions(_cmgdb PRIVATE CMG_VERBOSE)
endif()
```

Pair with the same for `MEMORYBOOKKEEPING` (see C2).

### E2. Default to `Release` build with explicit optimisation flags

`CMakeLists.txt` does not set `CMAKE_BUILD_TYPE`. If the user
configures with no flags, CMake defaults to "no optimization, no
debug symbols" on macOS and Linux — meaning the wheel-builders
elsewhere are paying for whatever default the build farm sets, not
what CMGDB requests. Add:

```cmake
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
```

Verify with the existing test suite that nothing breaks under `-O3`.

### E3. Emit `compile_commands.json`

`set(CMAKE_EXPORT_COMPILE_COMMANDS ON)` in `CMakeLists.txt`. Lets
clangd / Cursor / VSCode resolve Boost / pybind11 include paths.
Saves hours of editor-noise frustration. The recent clangd
false-positive diagnostics that showed up while editing
`Compute_Morse_Graph.hpp` are exactly this gap.

## F. Test & verification

This section is a **hard prerequisite** for everything in §A and §C.
The existing two-test suite uses adaptive mode (`init=4, min=6,
max=10`) and never exercises uniform mode at any depth — it cannot
detect the regressions A1, A1b, A3, and C1 are most likely to
introduce. The intrinsic-test bar must rise before any of those
ship.

Sequencing rule: F1, F2, F3 all land *before* any commit in §A or §C.

### F1. Expand the intrinsic test suite

`tests/test_basic.py` has two tests. Add fixtures grouped by what they
protect:

**BoxMap contract** (`ComputeBoxMap.py:30-50`):
- For a fixed `f`, assert `BoxMap(f, rect, mode='corners')` returns the
  axis-wise min/max of `f` evaluated on the `2^d` corners, with and
  without `padding`.
- Assert `mode='center'` forces `padding=True` (`ComputeBoxMap.py:35`)
  by checking that the returned rectangle is inflated by the box side
  length even when `padding=False` is passed.
- Assert `mode='random'` with `num_pts=k` evaluates `f` exactly `k`
  times (via a side-channel counter) and returns the bounding box of
  those `k` samples.
- Assert `mode='unknown_string'` returns `[]` (current silent-fail
  behaviour, `ComputeBoxMap.py:42-43`).

**Subdivision modes**:
- A "trivially contracting" system (e.g. `f(x) = 0.5 * x` on
  `[-1, 1]^2`) whose Morse graph should be a single attractor at the
  origin, at any `(init, min, max)`. Both adaptive and uniform modes
  should give the same single-vertex graph.
- A system where adaptive and uniform modes are *known* to produce the
  same Morse vertex/edge structure, run with both configurations,
  asserting equality. This is the lever for catching regressions in
  the hierarchical refinement loop.
- A 1D test. CMGDB rarely sees `d=1` so the boundary cases are
  under-exercised.
- A 3D or 4D test using `ModelMapF` and a known map (e.g. the existing
  `ConleyIndex` examples in `examples/`).

**Boundary cases**:
- A box whose lower face lies *exactly* on a grid line. `Grid::cover`
  has implicit floor/ceil conventions that the algorithm depends on;
  test the documented behaviour explicitly. See F2.
- A map whose image straddles the phase-space boundary (the `cover`
  out-of-domain handling is non-trivial — see e.g.
  `CMGDB_utils/CubicalGrid.py:67-70`, which `CubicalGrid` handles
  explicitly; CMGDB's behaviour here should be pinned in a test).
- A degenerate rectangle (zero-volume box, e.g. one axis collapsed).

**Determinism**:
- Two `ComputeConleyMorseGraph(model)` calls with identical inputs
  must produce identical outputs (vertex count, edge set, Conley
  strings). The `MEMORYBOOKKEEPING` globals (C2) are the obvious
  vector for hidden state to leak; this test catches that.
- The same `model` run with `F=user_map` and with
  `F=user_map_batched` (the A1 batched callable that returns the
  same images, batched) must produce **byte-identical** Morse
  graphs and Conley strings. This is the bar A1 has to clear.
- Same test, but with `F=user_map_dlpack` (A1b GPU-resident
  callable). The DLPack handoff round-trip must preserve numerical
  values to bit-equality for the `mode='corners'` enclosures.

**Memory regression** (gates C1):
- A fixture that runs a known problem at `smax=22` in 2D and asserts
  peak RSS (via `resource.getrusage(...).ru_maxrss` on Linux/macOS,
  or `/proc/self/statm` if available) stays under a documented
  ceiling. The ceiling is set by the current `PointerGrid` baseline
  with a margin; C1 lowers it.

**`ComputeConleyIndex` standalone** (`CMGDB.cpp:36-48`):
- A known cubical index pair `(X, A)` with a hand-computed Conley
  index. Without this we have no fixture protecting the standalone
  pybind path used by `CMGDB_utils.ComputeConleyMorseGraph`.

### F2. Document and lock in the `Grid::cover` floor/ceil convention

`Grid::cover` is the place where math-invariance most depends on
unwritten boundary conventions. The convention is currently encoded
implicitly in `CubicalGrid.grid_cover` in `CMGDB_utils`:

```python
min_coord(k) = ceil((box_lower - L) / cube_size) - 1
max_coord(k) = floor((box_upper - L) / cube_size)
```

(`archive/CMGDB_utils/src/CMGDB_utils/CubicalGrid.py:76-77`). Mirror
this in a header comment over `Grid::cover` in `TreeGrid.h` and
test it with property-based fuzzing (F1 fixture).

### F3. Add a Conley-index golden fixture

The relative-homology Conley computation is the most fragile part of
the codebase against refactoring. Pick three known-output examples
(an attracting fixed point → `[x-1, 0, 0]`, a 3-period attractor →
something definite, a saddle), capture their full Conley strings,
and assert them under repeated runs. This will catch regressions in
`chomp/RelativeMapHomology.h` and friends before they reach user
code.

## G. Output

### G1. Optional binary output for very large `morse_set_boxes`

`SaveMorseData.py` writes a CSV. At `smax=24` in 2D the result is
several million rows; CSV write + parse becomes a non-trivial
fraction of the figure-render pipeline. An HDF5 or Parquet alternate
output (column-major, no parsing cost on read) would be a clean win
for users who load the result back into NumPy/pandas. CSV stays as
the default for ASCII grep-ability.

**Effort.** Low. ~30 LOC + an `h5py` optional dependency.

### G2. Richer DOT labels

`PlotMorseGraph` emits `label="v : (conley_polynomials)"`. Adding
the Morse set's box count and a colour scale by Conley-index
nontriviality would make the graphviz output more useful at a
glance. Cosmetic; non-breaking.

## Recommended sequencing

Locked-in architecture: **per-`MorseDecomposition`-node batching** (A1),
**DLPack zero-copy** for GPU (A1b), **F1–F3 as hard prerequisite**, and
peak-runtime-memory at deep subdiv (C1) as the memory target. Sequencing:

| Phase | Items | Risk | Effort | Wall-clock impact (smax=20, 2D) |
| --- | --- | --- | --- | --- |
| 0 | C2 (bookkeeping opt-in), D1, D3, E1, E2, E3 | Low | Low (~1 wk) | None; build hygiene + reproducibility |
| 1 (gate) | F1 + F2 + F3, incl. memory-regression fixture | Low | Med (~2 wks) | None; safety net for A1, A1b, A3, C1 |
| 2 | **A1: per-node batched `box_map`** | Med | Med (~2 wks) | 194s → ~30s (**~5–8×**) |
| 3 | **A1b: DLPack zero-copy GPU handoff** | Med-high | Med-high (~2 wks) | ~30s → ~18s on GPU; bigger gap at d≥4 |
| 4 | A3 (uniform `cover` specialisation) | Med | Med (~1 wk) | Small additional win in uniform mode |
| 5 | **C1: release per-node grids on spawn** | High | High (~3 wks) | Enables smax=24 in 4D within 32 GB RAM |
| Defer | A2 (parallel `decompose` across PQ), B1 (parent-adjacency cache), D2, D4, G1 | — | — | Not perf-critical given A1+A1b+C1 |
| Off-table | SCC / chomp / subdivision algorithm changes | — | — | Math invariance |

**Hard prerequisite for Phases 2–5**: Phase 1 has landed. The
existing `test_basic.py` does not exercise uniform mode, batched
APIs, GPU dtype paths, or memory regressions; without F it is
structurally impossible to validate that A1/A1b/A3/C1 preserve the
math.

**Stop conditions**: after Phase 3 (A1 + A1b) lands, re-measure.
Wall-clock on the latent-dynamics consumer's leslie2d-to-2d
uniform sweep should be down by an order of magnitude. If so, A2
and B1 stay deferred; if SCC is now the bottleneck, revisit. The
C1 peak-memory work is independent of the perf bottleneck question.

## What we won't do

- **Replace the chomp library.** The Conley index correctness is the
  single most important guarantee CMGDB offers. The `chomp/`
  subdirectory is unrelated to the rest of the codebase by
  abstraction; rewriting it is a year of work and a complete
  trust reset.
- **Change SCC algorithm.** Tarjan is correct and fast enough; any
  parallel SCC algorithm (e.g. Forward-Backward) has a different
  serialisation behaviour and would force a regression of the test
  baselines for *no* observed bottleneck win.
- **Switch from `double` to higher-precision arithmetic.** Out of
  scope; the rectangular envelopes already absorb roundoff.
- **Refactor `subdiv_init / min / max / limit` semantics.** Those
  parameters are user-facing and load-bearing; renaming or merging
  them breaks every example notebook.
- **"Modernise" the C++.** The codebase predates C++17 idioms in
  places (`BOOST_FOREACH`, raw pointer ownership in
  `MorseDecomposition::children_`). Cleaning these up is satisfying
  but costs maintainer time we don't have; defer until C2 + E2 work
  has shown the build is reliably reproducible.

## How to verify any change preserves the math

For every commit that touches a hot path, the CMGDB-intrinsic check:

1. `pytest tests/ -v` from `archive/CMGDB/`. Until F1–F3 land this is
   just `test_basic.py` (the 4-vertex Morse graph with the four
   annotated Conley indices). After F1–F3 land it should cover all the
   modes / dimensions / mode-comparison fixtures listed there.

A change that fails (1) is reverted, not patched-over, unless we have
a written-down argument for why the new output is the correct one.
The default assumption is the previous output was correct.

**Optional downstream sanity check** (advisory, not required for
CMGDB to ship a change): after updating the editable
`archive/CMGDB` install, the latent-dynamics consumer can re-run the
project's leslie2d-to-2d uniform sweep at
`smax ∈ {10,12,14,16,17,18,19,20}` and diff
`output/leslie2d_to_2d_uniform_nopad/profile_results.json` against the
committed baseline, then run `pytest tests/` from `code/`. A failure
*here* with (1) still passing is evidence either that CMGDB's
intrinsic test coverage is missing a case the consumer exercises (file
a new fixture — see F1) or that the consumer's pipeline relies on
something other than the documented CMGDB output (a consumer bug, not
a CMGDB bug). Either way, the path forward is to land the missing
fixture inside CMGDB's own test suite before deciding what to revert.

## Anchors

- Profile data: `output/leslie2d_to_2d_profile/profile_results.json`
  and `output/leslie2d_to_2d_uniform_nopad/profile_results.json` in
  the latent-dynamics project.
- Existing baseline test: `tests/test_basic.py` (2D Henon-like, 4
  Morse sets).
- Algorithm reference: `latent-dynamics/code/docs/cmgdb_reference.md`.
- `CMG_VERBOSE` silencing patch: commit `184467e` on this archive
  branch.
