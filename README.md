# CMGDB
Conley Morse Graph Database

## Overview

This project uses combinatorial and topological methods to compute dynamics of discrete dynamical systems.

## Installation

Install the latest tagged version:

	pip install CMGDB

To uninstall:

	pip uninstall CMGDB

## Documentation and examples

To get started on how to run the code see the examples in the Jupyter notebooks in the [examples](examples) folder.

In particular the notebooks [Examples.ipynb](examples/Examples.ipynb), [Gaussian\_Process\_Example.ipynb](examples/Gaussian_Process_Example.ipynb), and [Conley\_Index\_Examples.ipynb](examples/Conley_Index_Examples.ipynb) present basic examples on how to run the code and are a good starting point.

Here is an old [survey](http://chomp.rutgers.edu/Projects/survey/cmdbSurvey.pdf) and a
[talk](http://chomp.rutgers.edu/Projects/Databases_for_the_Global_Dynamics/software/LorentzCenterAugust2014.pdf) that might be useful.

## Precomputed box maps

This fork includes optional Python helpers for maps that are expensive to
evaluate one box at a time. `CMGDB.make_precomputed_box_map` evaluates a
batched map on the finest corner lattice in bounded chunks, then returns a
standard `box_map(rect)` callable for `CMGDB.Model`.

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

The returned object is still callable, and it also exposes `batch(rects)`.
When a batched rectangle callback is available, install it on the model so
CMGDB can build cached adjacencies with fewer Python calls:

```python
model.set_batch_map(box_map.batch)
```

Torch is not a required dependency. If Torch is installed and `f` is a
`torch.nn.Module`, the helper evaluates it on `mps`, then `cuda`, then `cpu`
when `device="auto"`.

## Benchmarks

This fork includes a correctness-validating benchmark harness:

```bash
python tests/bench.py
python tests/bench.py --heavy
python tests/bench.py --scenarios py_medium,reach_4d --repeats 5 --warmup 1
```

The harness validates expected Morse-graph outputs before reporting timings.
It is useful for checking changes to `MapGraph`, reachability, and Python map
callback paths.

## Installing from source and dependencies

To install from source you need a C++ compiler and the following dependencies installed: [Boost](https://www.boost.org/), [GMP](https://gmplib.org/), and the [Succinct Data Structure Library (SDSL)](https://github.com/simongog/sdsl-lite). Assuming you have these dependencies installed in your system, you can install from source with the command:

	pip install --force-reinstall --no-deps --no-cache-dir git+https://github.com/bernardorivas/CMGDB.git

Alternatively, you can clone the GitHub repository and install with:

	git clone https://github.com/bernardorivas/CMGDB.git
	cd CMGDB
	./install.sh
