# CMGDB

Conley Morse Graph Database — combinatorial-topological computation of the
global dynamics of discrete dynamical systems.

> **This is a fork of [CMGDB](https://github.com/marciogameiro/CMGDB) by Marcio Gameiro.**
> For the official, PyPI-released package, install upstream with `pip install CMGDB`.
> This fork is kept merged with upstream (it currently includes upstream
> v1.5.2) and adds a few performance and analysis features (see
> [What this fork adds](#what-this-fork-adds)). It is not published to PyPI;
> prebuilt wheels are attached to the fork's GitHub releases, and the current
> tree installs from source as shown below. The mathematical output of the
> inherited CMGDB algorithms is unchanged from upstream, except for the checks
> that the fork adds to `ComputeConleyIndexForCells`.

## Overview

CMGDB uses combinatorial and topological methods to compute the dynamics of
discrete dynamical systems. Given a map and a phase space, it builds the Morse
graph (the partial order of recurrent components) and computes the Conley index
of each Morse set.

## What this fork adds

Relative to upstream, this fork adds the following. Apart from the checks in
`ComputeConleyIndexForCells` described below, none of it changes the Morse
graph, Conley indices, or subdivision semantics that upstream computes.
Batched map evaluation (`Model.set_batch_map`), the transition-graph cache,
the native reachability queries, `ComputeConleyIndexForCells`, and
`PrecomputedBoxMap` are upstream features; see
[Performance options](#performance-options-and-the-transition-graph-cache) and
[Precomputed box maps](#precomputed-box-maps).

- **Index-pair check in `ComputeConleyIndexForCells`** — for a cell set S,
  upstream's version computes the index from the pair X = cover(F(S)),
  A = X \ S whether or not that is an index pair. The fork's raises
  `ValueError` when a cell of A maps into X \ A, since (X, A) is then not
  an index pair. This also refuses some Morse sets of adaptive runs, whose
  `annotations` come from the same pair without the check, as in upstream.
  It also raises `ValueError` for a Model without a map, on which upstream's
  crashes, or with another dimension than the Morse graph's grid, and it
  takes the `batch_chunk_size` keyword of `ComputeConleyMorseGraph`. See
  [Performance options](#performance-options-and-the-transition-graph-cache).
- **Grid-layout precomputed box maps** — `CMGDB.make_precomputed_box_map(...)`
  complements upstream's `CMGDB.PrecomputedBoxMap` class with a uniform-grid
  layout, a reproducible `random` sampling mode, and memory-aware chunking of
  the lattice evaluation.
- **Compact MapGraph checkpoints** — `MapGraph.csr_view()` exposes the native
  cached CSR as read-only zero-copy NumPy arrays, and
  `write_map_graph_csr_checkpoint(...)` atomically persists mmap-ready `int64`
  offsets plus `int32`/`int64` targets with explicit caps and strict
  configuration/content fingerprints. See
  [the MapGraph CSR documentation](docs/map_graph_csr.md).
- **A cached `map_graph` by default** — `ComputeMorseGraph` and
  `ComputeConleyMorseGraph` return a cached `map_graph` unless
  `cache_map_graph=False`, `CMGDB_MAPGRAPH_CACHE=0`, or a `max_cached_edges`
  limit is exceeded (upstream returns a lazy one by default), so `csr_view`,
  the checkpoints and the native reachability queries work on the default
  result. The default result therefore holds the graph's full CSR (see
  [Cache sizing](#cache-sizing)), and building it and copying it once on
  return raise the call's peak memory: in one measurement with the `henon3d`
  benchmark model, whose CSR is 316 MB, the peak was about 1.2 GB, against
  0.27 GB with `cache_map_graph=False`. Runs that do not use the `map_graph`
  should pass `cache_map_graph=False` or call the `*Only` variants below.
- **Environment controls for the MapGraph cache** — allocation hints, opt-in
  hard limits, and `CMGDB_MAPGRAPH_CACHE`, which sets the default of the cache
  keyword arguments; see [Cache sizing](#cache-sizing).
- **Regions of attraction** — `CMGDB.cmgdb_roa` and `CMGDB.morse_graph_parser`
  provide exact region-of-attraction labels computed on the `MapGraph` returned
  during the Morse stage, plus a standalone parser for CMGDB's DOT output.
- **Morse-graph lattices** — `CMGDB.morse_lattice` builds the lattices of
  attractors and repellers and the nontrivial Conley-Morse graph from a parsed
  Morse graph, and `CMGDB.plot_derived_graph` renders them with graphviz.
- **Explicit relative-chain-complex bridge** —
  `CMGDB.ComputeRelativeHomologyShiftClass(...)` accepts a finite based
  relative chain complex and an explicit chain endomorphism without pretending
  that noncubical cells are boxes. It validates the chain-complex and chain-map
  equations over F_5, computes the induced homology maps, and applies CMGDB's
  existing Frobenius/shift-class reduction. `CMGDB.ComputeRelativeShiftClass`
  takes the same arguments and computes the shift class by linear algebra. See
  [the explicit-chain API documentation](docs/explicit_relative_chain_maps.md).
- **Carrier chain maps** — `CMGDB.ComputeCarrierChainMap(...)` computes the
  chain map over F_5 induced by an acyclic carrier between simplicial
  complexes, after checking that every carrier is nonempty and acyclic and that
  the exit subcomplex is mapped into the exit subcomplex.
- **Tagged finite-union box maps** — `CMGDB.AtlasModel(...)` runs the native
  adaptive SCC/Morse pipeline on a disjoint union of rectangular charts. Its
  callback returns separately covered `(target_chart, bounds)` pieces, which
  avoids convexifying a hybrid image across reset seams. Atlas charts are not
  a quotient cell complex: glued-face representatives and quotient incidence
  remain the adapter's responsibility. `set_active_subgrid` constructs a
  selected tagged dyadic antichain directly as a compressed tree, without first
  allocating its full rectangular refinement; inactive targets are explicit
  exits with no implicit cemetery vertex. The legacy TreeGrid/CHOMP Conley path
  is disabled. See [the AtlasModel documentation](docs/atlas_model.md).
- **Fixed-subdivision Morse-set reachability verification** —
  `CMGDB.ComputeMorseSetReachability(model, morse_graph, phase_subdiv=s, ...)`
  independently verifies the reachability relation of an adaptive
  `MorseGraph` on the conceptual uniform grid at a fixed subdivision depth,
  without materializing a complete `TreeGrid` or `MapGraph`. Each Morse
  set's forward closure is exhausted independently; every ordered pair is
  classified `REACHABLE` / `NOT_REACHABLE` / `INCOMPLETE`, and
  `absent_adaptive_edges()` lists exactly the adaptive edges certified
  absent at the tested subdivision. The result reports mutual-reachability
  (coalescing) groups and non-transitivity witnesses instead of silently
  reducing an invalid relation, supports per-source resource limits with
  resumable checkpoints, and carries a versioned provenance record.
  `CMGDB.ComputeMorseSetReachabilityStudy(...)` repeats the verification at
  several subdivisions and classifies pairs as agreeing, unstable, or
  unresolved. The input `MorseGraph` is never mutated.
- **Morse graphs without the returned MapGraph** —
  `CMGDB.ComputeMorseGraphOnly(model)` and
  `CMGDB.ComputeConleyMorseGraphOnly(model)` skip the extra box-map pass that
  builds the returned `MapGraph`, for runs that do not use it.

## Installation

Prebuilt wheels are attached to each
[fork release](https://github.com/bernardorivas/CMGDB/releases) for CPython
3.11-3.13 on manylinux x86_64 and macOS arm64. To install the most recent
release, `1.3.3+fork.6`:

	pip install cmgdb==1.3.3+fork.6 \
	  --find-links https://github.com/bernardorivas/CMGDB/releases/expanded_assets/v1.3.3%2Bfork.6

The version pin is what selects this fork; `--find-links` only tells pip where
to look. Release `1.3.3+fork.6` predates the merge with upstream v1.5.2; until
a newer release is published, build from source to get the upstream features.

To build the current tree from source you need a C++17 compiler and
[Boost](https://www.boost.org/) 1.66 or later (chrono, thread and
serialization). CMake finds Boost through the `BoostConfig.cmake` that Boost
installs from 1.70 on, and older releases through its FindBoost module.
sdsl-lite is vendored, and GMP is not needed. Install the current tree
directly with:

	pip install git+https://github.com/bernardorivas/CMGDB.git

Alternatively, clone the repository and install with:

	git clone https://github.com/bernardorivas/CMGDB.git
	cd CMGDB
	pip install .

In a [uv](https://docs.astral.sh/uv/) environment, use `uv pip install .`
instead. Rendering the `graphviz.Source` objects returned by `PlotMorseGraph`
and `plot_derived_graph` needs Graphviz's `dot` program on `PATH`.

To uninstall:

	pip uninstall CMGDB

> This fork uses the same import name (`CMGDB`) as the upstream package, so it
> replaces upstream in your environment rather than installing alongside it.

## Documentation and examples

To get started, see the Jupyter notebooks in the [examples](examples) folder.
[Examples.ipynb](examples/Examples.ipynb),
[Gaussian_Process_Example.ipynb](examples/Gaussian_Process_Example.ipynb), and
[Conley_Index_Examples.ipynb](examples/Conley_Index_Examples.ipynb) cover the
basic workflow and are a good starting point.
[Precomputed_vs_OnDemand_BoxMap.ipynb](examples/Precomputed_vs_OnDemand_BoxMap.ipynb)
and [Regions_of_Attraction.ipynb](examples/Regions_of_Attraction.ipynb)
demonstrate the fork-specific features, and
[Lattice_and_Nontrivial_CMGraph.ipynb](examples/Lattice_and_Nontrivial_CMGraph.ipynb)
and [Attractor_Cell_Sets.ipynb](examples/Attractor_Cell_Sets.ipynb) cover the
Morse-graph lattice / nontrivial-graph / attractor-cell helpers.

For background, see this
[survey](http://chomp.rutgers.edu/Projects/survey/cmdbSurvey.pdf) and
[talk](http://chomp.rutgers.edu/Projects/Databases_for_the_Global_Dynamics/software/LorentzCenterAugust2014.pdf).

## Performance options and the transition-graph cache

`ComputeMorseGraph` and `ComputeConleyMorseGraph` return a pair `(morse_graph, map_graph)` and accept keyword arguments controlling the transition-graph machinery (the defaults are right for most runs):

* `cache_transition_graph` (default `None`: cache unless `CMGDB_MAPGRAPH_CACHE=0`) — cache the per-level transition graph used internally by the SCC/reachability passes, halving the map evaluations per subdivision level. Set `False` for a memory-lean run that re-evaluates the map on demand; since the returned `map_graph` is cached by default, such a run also needs `cache_map_graph=False` (or `CMGDB_MAPGRAPH_CACHE=0`, which makes both flags default to `False`).
* `batch_chunk_size` (default `65536`) — rectangles per batched map call when a batch map is attached with `model.set_batch_map` (`0` means one call for the whole grid). The Conley-index phase of `ComputeConleyMorseGraph` also gathers its map evaluations into these chunks, evaluating each rectangle exactly once — without a batch map the evaluations are scalar but still deduplicated, so attaching a batch map speeds up every phase, not just the transition graph. `ComputeConleyIndexForCells` takes the same keyword for its batched evaluations.
* `max_cached_edges` (default `0` = unlimited) — abandon a cache as soon as it would exceed this many edges and fall back to on-demand evaluation. The limit is checked before each row of the graph is stored, so the edge array never holds more edges than that, whatever `batch_chunk_size` is. It bounds the returned `map_graph`'s cache too: with an explicit `cache_map_graph=True`, a returned graph over the limit comes back lazy with a `RuntimeWarning`. It does not bound a later `map_graph.build_cache()`, which takes its own `max_cached_edges` (default `0` = unlimited) and raises `RuntimeError` when the graph exceeds it.
* `reserve_edges` / `reserve_min_edges` (defaults `0` / `2**24`) — up-front sizing of the flat edge array. By default the final edge count is projected from the first chunk and twice that is reserved, which avoids the reallocation spikes of multi-gigabyte graphs on deep grids; a positive `reserve_edges` reserves exactly that many instead. Reservation only engages once the projection reaches `reserve_min_edges`.
* `cache_map_graph` (default `None`: cache unless `CMGDB_MAPGRAPH_CACHE=0`; upstream's default is `False`) — eagerly cache the **returned** `map_graph` (one extra full batched map pass over the final grid, after which its adjacency queries are O(1) array lookups). `False` returns a lazy `map_graph` that evaluates the map per `adjacencies` query; `map_graph.build_cache()` upgrades it to the cached form later. `map_graph.has_cache()` and `map_graph.num_cached_edges()` report the state.

An explicit `True` or `False` for either cache flag always wins over `CMGDB_MAPGRAPH_CACHE`. `ComputeMorseGraphOnly` and `ComputeConleyMorseGraphOnly` take the same keyword arguments except `cache_map_graph`, since they return no `map_graph`. The `AtlasModel` overloads take no keyword arguments; their caches follow `CMGDB_MAPGRAPH_CACHE`.

A batch map attached with `model.set_batch_map(g)` receives a read-only NumPy array of shape `(count, 2*dim)`, one rectangle per row (lower bounds, then upper bounds), and must return the image rectangles in the same layout, as an array or a list of lists. It must agree with the model's scalar map on every rectangle.

A cached `map_graph` also unlocks the native post-processing queries (all of which release the GIL and refuse a lazy graph):

* `MorseReachabilityMasks(map_graph, morse_graph, cells)` — for each queried cell, a `uint64` bitmask of the Morse nodes reachable through the box dynamics (bit `i` = Morse node `i`); the basin-of-attraction primitive.
* `MorseSingletonReachability(map_graph, morse_graph, cells)` — per cell, the single reachable Morse node id, `-1` for none, `-2` for several.
* `MorseDirectedPathCells(map_graph, morse_graph, sources, targets)` — the cells lying on some directed path from the source Morse nodes to the target Morse nodes (candidate connecting-orbit regions).
* `ComputeConleyIndexForCells(model, morse_graph, cells, batch_chunk_size=65536)` — the homological Conley index of a set S of cells of the final grid, computed as `ComputeConleyMorseGraph` computes it for its Morse sets, from the pair X = cover(F(S)), A = X \ S. This is an index pair exactly when no cell of A maps into X \ A; otherwise the function raises `ValueError`. Sets that contain every cell on a path between two of their cells pass, such as the Morse sets of a run with equal initial, minimum and maximum subdivision and the cells from `MorseDirectedPathCells`; a Morse set of an adaptive run can fail when a sampled box map is not monotone under refinement. For cells of different depths the homology is computed at the finest depth in S: coarser cells are subdivided, and deeper cells of A are replaced by their ancestors at that depth; the check is made on the cells, not on these cubes.

The three reachability queries are exact on the cells of `map_graph`: a cell reaches a Morse node when a directed path in `map_graph` leads from it to a cell of that node's Morse set, and a cell lies on a directed path when a cell of a source Morse set reaches it and it reaches a cell of a target Morse set. They are computed by a strongly connected component sweep of the CSR and do not use the Morse graph's edges. On a hierarchical run (`phase_subdiv_init < phase_subdiv_min`, as with the `Model(subdiv_min, subdiv_max, lower_bounds, upper_bounds, F)` constructors) the Morse graph comes from grids coarser than `map_graph`, so the two can disagree: a Morse edge need not be realized by any cell path, and with a box map that is not monotone under refinement (such as corner-sampled `BoxMap`), the cells of a Morse set can reach Morse sets that the Morse graph does not place below it, and `map_graph` can have cycles outside the Morse sets. Each call allocates arrays over all cells of `map_graph` and traverses every cell and edge reachable from its query cells (the cells of the source Morse sets for `MorseDirectedPathCells`), Morse sets included; the query cells of one call share a single traversal, so pass many cells to one call rather than one cell per call.

The C++ core is silent by default; rebuild with the `CMG_VERBOSE` preprocessor define (uncomment it at the top of `src/CMGDB/_cmgdb/CMGDB.cpp`) to restore the progress and diagnostic prints.

## Precomputed box maps

For maps that are expensive to evaluate one box at a time, CMGDB can evaluate
a batched map once on the finest lattice, in bounded chunks, and then serve
every box query from that table. There are two entry points.

Upstream's `CMGDB.PrecomputedBoxMap` class serves the adaptive subdivision tree
with corner or center sampling:

```python
F = CMGDB.PrecomputedBoxMap(f, lower_bounds, upper_bounds, subdiv_max,
                            mode="corners",   # sampling rule: corners | center
                            padding=False)
model = CMGDB.Model(subdiv_min, subdiv_max, lower_bounds, upper_bounds, F)
model.set_batch_map(F.batch)
```

This fork's `CMGDB.make_precomputed_box_map` also serves uniform grids and
random sampling:

```python
box_map = CMGDB.make_precomputed_box_map(
    f,  # batched NumPy callable or torch.nn.Module
    lower_bounds,
    upper_bounds,
    subdiv_max=28,
    mode="adaptive",       # grid layout: adaptive | uniform
    eval_mode="corners",   # sampling rule: corners | center | random
    padding=False,
    batch_points="auto",
    device="auto",   # Torch only: mps, then cuda, then cpu
)

model = CMGDB.Model(
    subdiv_min,
    subdiv_max,
    subdiv_init,
    subdiv_limit,
    lower_bounds,
    upper_bounds,
    box_map,
)
model.set_batch_map(box_map.batch)
```

On the adaptive grid with corner or center sampling the two build the same
table and return the same image boxes. They differ as follows:

| | `PrecomputedBoxMap` | `make_precomputed_box_map` |
|---|---|---|
| `mode` | sampling rule (`corners`, `center`) | grid layout (`adaptive`, `uniform`); the sampling rule is `eval_mode` |
| `padding` default | `False` | `True` |
| box off the `subdiv_max` lattice, or finer than it | raises `ValueError` | snapped to the lattice |
| `batch(rects)` returns | `(N, 2*dim)` NumPy array | list of lists |
| `batch_points="auto"` | chunks of `2**20` points | sized from available memory (SLURM aware) |
| Torch | used only if `torch` is already imported | imported when installed |

The helpers `precompute_corner_grid`, `evaluation_offsets`,
`resolve_batch_points`, `as_batched_evaluator` and `select_torch_device` are
importable from `CMGDB` and from `CMGDB.PrecomputedBoxMap`.

For a finite tagged chart family, use the separate Atlas lookup.  It preserves
every tagged target piece (including an explicit empty union), rejects exact
duplicate source rectangles, and raises on a cache miss rather than silently
turning that miss into an open exit:

```python
atlas_box_map = CMGDB.precompute_atlas_box_map(
    tagged_callback,
    [(chart_id, bounds), ...],
    batch_size=4096,
    batch_callback=optional_batched_tagged_callback,
    provenance_callback=optional_source_provenance,
)
atlas_model.set_map(atlas_box_map)
```

`PrecomputedAtlasBoxMap` is a verbatim callback-value cache; it does not turn
sampled values into a certified continuous-image enclosure.  Its `batch`
method is available to Python callers, although the current native
`AtlasModel` consumes the scalar tagged-union callback interface.

### Cache sizing

By default, the eager CSR cache has **no size ceiling**. A graph is built for
whatever grid it is given; a run too large for the host fails where it actually
runs out of memory, rather than being refused up front on a guess. Sizing the
run is the caller's decision. Explicit hard limits are available when an
application has a real resource budget, as described below.

Budgeting is still worth doing: offsets use approximately `8 * (vertices + 1)`
bytes and cached edges use `8 * edges` bytes, excluding temporary batch objects
and `std::vector` growth overhead. A `2^24`-cell graph needs about 128 MiB for
offsets; 64 edges per cell would add 8 GiB of edge storage.

Two optional environment variables tune allocation. Neither refuses anything:

- `CMGDB_MAPGRAPH_RESERVE_EDGES` is unset by default. Set it to reserve the
  edge buffer up front, before the first box is evaluated, instead of growing
  it geometrically. A reserve smaller than the real edge count is not an
  error; the buffer simply grows past it. A call with `max_cached_edges` set
  caps this reserve at that many edges.
- `CMGDB_MAPGRAPH_RESERVE_MIN_VERTICES` defaults to `16777216`. The explicit
  reserve applies only to graphs at least this large, so coarse intermediate
  MapGraphs do not each take a multi-gigabyte allocation.

Both must be positive base-10 integers; a malformed value is an error rather
than being silently ignored, since ignoring it would drop the hint you asked
for. A malformed value of these variables, or of the hard limits below, fails
before the first map evaluation of any call that builds a cache. The projected
reservation of the `reserve_edges` / `reserve_min_edges` keyword arguments (see
[Performance options](#performance-options-and-the-transition-graph-cache))
still applies after the first chunk: once the projected edge count reaches
`reserve_min_edges`, it enlarges a smaller buffer to twice the projection, or
to exactly `reserve_edges` when that is set.

Three separate opt-in variables stop native CSR growth before its next reserve
or append:

- `CMGDB_MAPGRAPH_HARD_MAX_VERTICES` limits `V`;
- `CMGDB_MAPGRAPH_HARD_MAX_EDGES` limits retained edges;
- `CMGDB_MAPGRAPH_HARD_MAX_CACHE_BYTES` limits the `int64` offset array plus
  native `uint64` edge-buffer capacity.

They are unset by default and accept nonnegative base-10 integers (zero is a
real limit). The byte cap does not include the grid, Morse graph, callback
state, or a transient scalar row / batch returned before CMGDB can count it.
Checkpoint caps remain independent because an on-disk checkpoint can use
`int32` targets even though the live native graph uses `uint64` targets.

For the measured 3-D level-24 graph (~1.096 billion edges) on a 48-GiB host:

```bash
CMGDB_MAPGRAPH_RESERVE_EDGES=1200000000 \
python ...
```

The 1.2-billion-edge reserve is about 8.94 GiB. The projected reservation
described above can still enlarge the buffer to twice the projected edge count
after the first chunk; a `reserve_min_edges` above any projected edge count
(for example `2**62`) turns it off.

To trade speed for memory, disable the cache outright:

```bash
CMGDB_MAPGRAPH_CACHE=0 python ...
```

The lazy path recomputes adjacencies through the map on every query -- far
slower, but it never materializes the edge array. Accepted values are
`0`/`1`, `off`/`on`, `false`/`true`. `CMGDB_MAPGRAPH_CACHE` only sets the
default: an explicit `cache_transition_graph=True` or `cache_map_graph=True`,
or `map_graph.build_cache()`, still builds the cache. `max_cached_edges` bounds
the caches a call builds, but not a later `map_graph.build_cache()`. The keyword
arguments `cache_transition_graph=False`, `cache_map_graph=False` and
`max_cached_edges` described under
[Performance options](#performance-options-and-the-transition-graph-cache)
trade speed for memory per call. A memory-lean call needs both cache flags
`False`: `cache_transition_graph=False` alone still caches the returned
`map_graph`.

> Removed in this fork: the old `CMGDB_MAPGRAPH_MAX_VERTICES` and
> `CMGDB_MAPGRAPH_MAX_EDGES`. They are read by nothing and setting them has no
> effect. The explicitly named `CMGDB_MAPGRAPH_HARD_*` limits above have
> fail-fast semantics and never select a lazy fallback. The `max_table_points`
> argument of `make_precomputed_box_map` is likewise gone.

### Evaluation modes

`eval_mode` selects where inside each box `make_precomputed_box_map` samples
the map, mirroring `CMGDB.ComputeBoxMap.BoxMap`. It is independent of `mode`,
which selects the grid layout:

```python
box_map = CMGDB.make_precomputed_box_map(
    f, lower_bounds, upper_bounds,
    subdiv_max=28,
    mode="adaptive",      # grid layout: adaptive | uniform
    eval_mode="center",   # sampling rule: corners | center | random
    num_pts=10,           # random only
    sample_depth=4,       # random only
    seed=0,               # random only
)
```

Non-corner sampling needs a finer table. A box at depth `t` on an axis refined
`T` times has its center at `(i + 1/2) * 2^(T - t)` in units of the finest
corner spacing -- not an integer when `t == T`, so the centers of the finest
boxes fall exactly between corner-lattice nodes. Refining the table by `d`
extra levels per axis makes every offset `k / 2^d` a node, at every box depth
at once, which is what lets boxes share evaluation points:

| `eval_mode` | extra levels | table size factor |
|---|---:|---|
| `corners` | 0 | 1 |
| `center` | 1 | `2^dim` |
| `random` | `sample_depth` | `2^(dim * sample_depth)` |

Two consequences worth knowing:

- `center` forces `padding=True`, as upstream `BoxMap` does. One sample gives a
  degenerate image box, which encloses nothing without padding.
- `random` draws its offsets **once** and reuses them for every box, so sibling
  boxes are probed at the same relative positions. Upstream `BoxMap` instead
  calls `np.random.uniform` afresh on each invocation, which makes its box map a
  non-deterministic function of the rectangle and its Morse graphs
  irreproducible. Fixed offsets are both precomputable and reproducible.

For exact basin membership on selected cells, use the native CSR query on a
cached `map_graph`:

```python
summary = CMGDB.MorseSingletonReachability(
    map_graph, morse_graph, query_cell_ids
)
in_basin_a = summary == a
```

The returned C-contiguous `int32` array is the unique reachable Morse-node id
when the complete reachable set is a singleton, `-1` when no Morse node is
reachable, and `-2` when two or more Morse nodes are reachable. The routine
requires a cached graph, never calls the map, and uses the existing forward CSR
without constructing a reverse edge array. `MorseReachabilityMasks(...)`
additionally returns exact all-node `uint64` masks when the Morse graph has at
most 64 nodes.

Torch is not a required dependency. If Torch is installed and `f` is a
`torch.nn.Module`, both entry points evaluate it on `mps`, then `cuda`, then
`cpu` when `device="auto"`.

## Benchmarks

Upstream's benchmark suite validates each scenario's Morse sets, reachability
and Conley indices against frozen references before it reports timings, so a
change that alters the computed dynamics fails:

```bash
python benchmarks/benchmark.py              # quick suite
python benchmarks/benchmark.py --heavy
python benchmarks/benchmark.py --scenario leslie2d_python --repeat 5
```

See [benchmarks/README.md](benchmarks/README.md) for the scenarios. It covers
Python, batched (`set_batch_map`), data-driven and interval box maps and the
Conley phase, so it replaces the fork's former `tests/bench.py`.

Upstream's scenarios call `ComputeMorseGraph` and `ComputeConleyMorseGraph`
without the cache keyword arguments, so under this fork's cached `map_graph`
default each of these runs makes one extra map pass over its final grid and
caches the returned graph. The suite's map-call counts, timings and peak
memory (the MB column) therefore include that pass and are not comparable with
upstream's results, such as the committed `baseline_857ec8b.json`,
`optimized_857ec8b.json` and `version_compare.md`. Measured on one machine,
`henon3d` peaked at about 1.2 GB, against 0.4 GB for upstream v1.5.2. The
~10 GB that [benchmarks/README.md](benchmarks/README.md) gives for
`chafee3d_uniform_24` is upstream's figure; the same computation at
subdivision 22 peaked at 5.8 GB, against 3.7 GB for upstream. No
`CMGDB_MAPGRAPH_CACHE` setting restores upstream's defaults, because
`CMGDB_MAPGRAPH_CACHE=0` also turns off the transition-graph cache.

## License

MIT, Copyright (c) 2020 Marcio Gameiro (see [LICENSE](LICENSE)). This fork is
maintained by Bernardo Rivas and retains the upstream license. The extension
also compiles in the vendored header-only
[sdsl-lite](src/CMGDB/_cmgdb/third_party/sdsl-lite) v3, which is BSD-3-Clause
licensed (see [licenses/sdsl-lite-xxsds-LICENSE](licenses/sdsl-lite-xxsds-LICENSE)),
and [CHOMP](src/CMGDB/_cmgdb/include/chomp), which is MIT licensed, Copyright
(c) 2015 Shaun Harker (see [licenses/chomp-LICENSE](licenses/chomp-LICENSE)).
Both notices ship with every wheel.
