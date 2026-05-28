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

## Installation

This fork is not on PyPI and must be built from source. You need a C++ compiler
and the following dependencies: [Boost](https://www.boost.org/),
[GMP](https://gmplib.org/), and the
[Succinct Data Structure Library (SDSL)](https://github.com/simongog/sdsl-lite).

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
demonstrate the fork-specific features.

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
