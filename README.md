# CMGDB

Conley Morse Graph Database — combinatorial-topological computation of the
global dynamics of discrete dynamical systems.

> **This is a fork of [CMGDB](https://github.com/marciogameiro/CMGDB) by Marcio Gameiro.**
> For the official, PyPI-released package, install upstream with `pip install CMGDB`.
> This fork adds a few performance and analysis features (see
> [What this fork adds](#what-this-fork-adds)). It is not published to PyPI;
> install the current development tree from source as shown below. Prebuilt
> wheels remain available for the older `fork.3` feature baseline. The
> mathematical output of the inherited CMGDB algorithms is unchanged from
> upstream.

## Overview

CMGDB uses combinatorial and topological methods to compute the dynamics of
discrete dynamical systems. Given a map and a phase space, it builds the Morse
graph (the partial order of recurrent components) and computes the Conley index
of each Morse set.

## What this fork adds

Relative to the upstream release, this fork adds the following. None of it
changes the Morse graph, Conley indices, or subdivision semantics that upstream
computes — these are additive helpers and a build-flag change.

- **Precomputed / batched box maps** — `CMGDB.make_precomputed_box_map(...)`
  evaluates an expensive map on the finest corner lattice in bounded chunks and
  returns a standard `box_map(rect)` callable, amortizing the per-box evaluation
  cost of maps that are slow to call one box at a time (e.g. neural-network or
  GPU-resident maps).
- **Batched adjacency construction** — `Model.set_batch_map(...)` plus a CSR
  adjacency cache in `MapGraph` let CMGDB build the map graph with far fewer
  Python calls.
- **Compact MapGraph checkpoints** — `MapGraph.csr_view()` exposes the native
  cached CSR as read-only zero-copy NumPy arrays, and
  `write_map_graph_csr_checkpoint(...)` atomically persists mmap-ready `int64`
  offsets plus `int32`/`int64` targets with explicit caps and strict
  configuration/content fingerprints. The scalar construction path used by
  `AtlasModel` now appends directly to CSR instead of retaining and copying a
  second per-row edge store. See
  [the MapGraph CSR documentation](docs/map_graph_csr.md).
- **Regions of attraction** — `CMGDB.cmgdb_roa` and `CMGDB.morse_graph_parser`
  provide exact region-of-attraction labels computed on the `MapGraph` returned
  during the Morse stage, plus a standalone parser for CMGDB's DOT output.
- **Conley indices for connection-complete cell sets** —
  `CMGDB.ComputeConleyIndexForCells(model, morse_graph, cells)` applies the
  same TreeGrid/CHOMP computation used for ordinary Morse-node annotations to
  an arbitrary phase-space cell subset. This supports Conley-index
  recomputation after collapsing an order-convex collection of Morse nodes and
  adding the cells on connections internal to that quotient fiber.
- **Explicit relative-chain-complex bridge** —
  `CMGDB.ComputeRelativeHomologyShiftClass(...)` accepts a finite based
  relative chain complex and an explicit chain endomorphism without pretending
  that noncubical cells are boxes. It validates the chain-complex and chain-map
  equations over F_5, computes the induced homology maps, and applies CMGDB's
  existing Frobenius/shift-class reduction. See
  [the explicit-chain API documentation](docs/explicit_relative_chain_maps.md).
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
- **A correctness-validating benchmark harness** — `tests/bench.py` checks the
  expected Morse-graph output before reporting timings.
- **Quieter default output** — per-run progress prints are gated behind a
  `CMG_VERBOSE` build flag (off by default).
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

## Installation

The Atlas, compact-CSR, and explicit-chain APIs documented above are currently
on `master` and have not yet been packaged in a release wheel. Build the
current source with a C++ compiler and
[Boost](https://www.boost.org/) (>= 1.56), [GMP](https://gmplib.org/), and
[SDSL](https://github.com/xxsds/sdsl-lite) v3:

	git clone https://github.com/bernardorivas/CMGDB.git
	cd CMGDB
	./install.sh

Or install the current tree directly with pip:

	pip install --force-reinstall --no-deps --no-cache-dir git+https://github.com/bernardorivas/CMGDB.git@master

The most recent prebuilt release, `fork.3`, predates the Atlas, compact-CSR,
and explicit-chain APIs. It remains available for the earlier batched-map and
reachability features on CPython 3.11-3.13, manylinux x86_64, and macOS arm64:

	pip install cmgdb==1.3.3+fork.3 \
	  --find-links https://github.com/bernardorivas/CMGDB/releases/expanded_assets/v1.3.3%2Bfork.3

The version pin is what selects this fork; `--find-links` only tells pip where
to look.

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

## Precomputed box maps

For maps that are expensive to evaluate one box at a time,
`CMGDB.make_precomputed_box_map` evaluates a batched map on the finest lattice
in bounded chunks, then returns a standard `box_map(rect)` callable for
`CMGDB.Model`.

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
```

The returned object is still callable, and it also exposes `batch(rects)`. When
a batched rectangle callback is available, install it on the model so CMGDB can
build cached adjacencies with fewer Python calls:

```python
model.set_batch_map(box_map.batch)
```

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

- `CMGDB_MAPGRAPH_RESERVE_EDGES` is unset by default. Set it to allocate the
  edge buffer once instead of growing it geometrically, avoiding a transient
  capacity peak. A reserve smaller than the real edge count is not an error;
  the buffer simply grows past it.
- `CMGDB_MAPGRAPH_RESERVE_MIN_VERTICES` defaults to `16777216`. The explicit
  reserve applies only to graphs at least this large, so coarse intermediate
  MapGraphs do not each take a multi-gigabyte allocation.

Both must be positive base-10 integers; a malformed value is an error rather
than being silently ignored, since ignoring it would drop the hint you asked
for.

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

The 1.2-billion-edge reserve is about 8.94 GiB, allocated once.

To trade speed for memory, disable the cache outright:

```bash
CMGDB_MAPGRAPH_CACHE=0 python ...
```

The lazy path recomputes adjacencies through the map on every query -- far
slower, but it never materializes the edge array. This is the supported way to
ask for a memory-lean run; earlier versions required setting an artificially
low edge cap to provoke the same fallback, which also refused unrelated runs
that would have fit. Accepted values are `0`/`1`, `off`/`on`, `false`/`true`.

> Removed in this fork: the old `CMGDB_MAPGRAPH_MAX_VERTICES` and
> `CMGDB_MAPGRAPH_MAX_EDGES`. They are read by nothing and setting them has no
> effect. The explicitly named `CMGDB_MAPGRAPH_HARD_*` limits above have
> fail-fast semantics and never select a lazy fallback. The Python
> `max_table_points` argument is likewise gone.

### Evaluation modes

`eval_mode` selects where inside each box the map is sampled, mirroring
`CMGDB.ComputeBoxMap.BoxMap`. It is independent of `mode`, which selects the
grid layout:

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
  boxes are probed at the same relative positions. Upstream instead calls
  `np.random.uniform` afresh on each invocation, which makes its box map a
  non-deterministic function of the rectangle and its Morse graphs
  irreproducible. Fixed offsets are both precomputable and reproducible.

For exact Marcio-style basin membership on selected cells, use the native CSR
query:

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
`torch.nn.Module`, the helper evaluates it on `mps`, then `cuda`, then `cpu`
when `device="auto"`.

## Benchmarks

The fork includes a correctness-validating benchmark harness:

```bash
python tests/bench.py
python tests/bench.py --heavy
python tests/bench.py --scenarios py_medium,reach_4d --repeats 5 --warmup 1
```

The harness validates expected Morse-graph outputs before reporting timings. It
is useful for checking changes to `MapGraph`, reachability, and Python map
callback paths.

## License

MIT, Copyright (c) 2020 Marcio Gameiro (see [LICENSE](LICENSE)). This fork is
maintained by Bernardo Rivas and retains the upstream license.
