# CMGDB

Conley Morse Graph Database — combinatorial-topological computation of the
global dynamics of discrete dynamical systems.

> **This is a fork of [CMGDB](https://github.com/marciogameiro/CMGDB) by Marcio Gameiro.**
> For the official, PyPI-released package, install upstream with `pip install CMGDB`.
> This fork adds a few performance and analysis features (see
> [What this fork adds](#what-this-fork-adds)) and is **installed from source**;
> it is not published to PyPI. The mathematical output is unchanged from upstream.

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
- **Regions of attraction** — `CMGDB.cmgdb_roa` and `CMGDB.morse_graph_parser`
  provide exact region-of-attraction labels computed on the `MapGraph` returned
  during the Morse stage, plus a standalone parser for CMGDB's DOT output.
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

This fork is not on PyPI, but every release carries prebuilt Linux wheels for
CPython 3.11-3.13, so no compiler or dependency is needed to use it:

	pip install cmgdb==1.3.3+fork.1 \
	  --find-links https://github.com/bernardorivas/CMGDB/releases/expanded_assets/v1.3.3+fork.1

The version pin is what selects this fork; `--find-links` only tells pip where
to look. To build from source instead you need a C++ compiler and
[Boost](https://www.boost.org/) (>= 1.56), [GMP](https://gmplib.org/), and
[SDSL](https://github.com/xxsds/sdsl-lite) v3, which is header-only.

Clone and install:

	git clone https://github.com/bernardorivas/CMGDB.git
	cd CMGDB
	./install.sh

Or install directly with pip:

	pip install --force-reinstall --no-deps --no-cache-dir git+https://github.com/bernardorivas/CMGDB.git

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
`CMGDB.make_precomputed_box_map` evaluates a batched map on the finest corner
lattice in bounded chunks, then returns a standard `box_map(rect)` callable for
`CMGDB.Model`.

```python
box_map = CMGDB.make_precomputed_box_map(
    f,  # batched NumPy callable or torch.nn.Module
    lower_bounds,
    upper_bounds,
    subdiv_max=28,
    mode="adaptive",  # or "uniform"
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

The eager CSR cache is bounded by two process-level environment variables:

- `CMGDB_MAPGRAPH_MAX_VERTICES` defaults to `16777216` (`2^24`), inclusive.
- `CMGDB_MAPGRAPH_MAX_EDGES` defaults to `200000000`.
- `CMGDB_MAPGRAPH_RESERVE_EDGES` is unset by default. Set it to a positive
  value no larger than `CMGDB_MAPGRAPH_MAX_EDGES` to allocate the final edge
  buffer once and avoid a transient capacity-growth peak.
- `CMGDB_MAPGRAPH_RESERVE_MIN_VERTICES` defaults to `16777216`; the explicit
  edge reserve is used only for graphs at least this large, avoiding a
  multi-gigabyte allocation for every coarse intermediate MapGraph.

Both values must be positive base-10 integers. A model with an installed batch
map fails with a clear error when either limit is exceeded instead of silently
falling back to one Python callback per cell. Increase the limits explicitly
for a larger run only after budgeting memory: offsets use approximately
`8 * (vertices + 1)` bytes and cached edges use `8 * edges` bytes, excluding
temporary batch objects and `std::vector` growth overhead. For example, a
`2^24`-cell graph needs about 128 MiB for offsets; 64 edges per cell would add
8 GiB of final edge storage.

For the measured 3-D level-24 graph (~1.096 billion edges), a bounded
48-GiB-host launch can use:

```bash
CMGDB_MAPGRAPH_MAX_VERTICES=16777216 \
CMGDB_MAPGRAPH_MAX_EDGES=1200000000 \
CMGDB_MAPGRAPH_RESERVE_EDGES=1200000000 \
python ...
```

The 1.2-billion-edge reserve is about 8.94 GiB, allocated once.

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
