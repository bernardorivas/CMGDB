# Conley indices of Morse sets on adaptive grids in CMGDB

Report: [report.pdf](report.pdf), source [report.tex](report.tex). Scripts are in [scripts/](scripts/), their stored outputs in [outputs/](outputs/), and the figures in [figures/](figures/). The names of the scripts, outputs and figures of Part I, on the bounds of the cubical complex, start with `bounds_`, and those of Part II, on the Leslie map, with `leslie_`.

## Builds

| tag | build | install |
|---|---|---|
| `cmgdb-1.5.2` | upstream CMGDB 1.5.2, before the fixes | `pip install CMGDB==1.5.2` |
| `cmgdb-1.3.3_fork.6` | fork release 1.3.3+fork.6, before the fixes | `pip install cmgdb==1.3.3+fork.6 --find-links https://github.com/bernardorivas/CMGDB/releases/expanded_assets/v1.3.3%2Bfork.6` |
| `cmgdb-1.5.3_fork.7.dev0-0e21503` | merge 0e21503 of upstream 1.5.2, before the fixes | `pip install git+https://github.com/bernardorivas/CMGDB@0e21503` |
| `cmgdb-1.5.3_fork.7.dev0-96642f2` | master 96642f2, after the fixes | `pip install git+https://github.com/bernardorivas/CMGDB@96642f2` |

The last two are built from source. 0e21503 needs a C++20 compiler and 96642f2 a C++17 compiler, and both need Boost with the chrono, thread and serialization libraries. Both report the version 1.5.3+fork.7.dev0, so their outputs are told apart by the tag, which the scripts that use CMGDB take as `--tag`. For the first two builds the default tag is the version.

## Environments

One Python 3.13 environment per build, with numpy, matplotlib and pytest, in a directory `V` outside the repository:

```bash
V=$HOME/venvs-cmgdb
python3.13 -m venv $V/upstream && $V/upstream/bin/pip install numpy matplotlib pytest CMGDB==1.5.2
python3.13 -m venv $V/fork && $V/fork/bin/pip install numpy matplotlib pytest cmgdb==1.3.3+fork.6 \
    --find-links https://github.com/bernardorivas/CMGDB/releases/expanded_assets/v1.3.3%2Bfork.6
python3.13 -m venv $V/merge && $V/merge/bin/pip install numpy matplotlib pytest git+https://github.com/bernardorivas/CMGDB@0e21503
python3.13 -m venv $V/master && $V/master/bin/pip install numpy matplotlib pytest git+https://github.com/bernardorivas/CMGDB@96642f2

UP=$V/upstream/bin/python
FK=$V/fork/bin/python
MG=$V/merge/bin/python
MA=$V/master/bin/python
```

The stored outputs were computed with Python 3.13.16, numpy 2.5.3, matplotlib 3.11.2 and pytest 9.1.1 on macOS arm64. The figure scripts and `bounds_table.py`, `leslie_dynamics.py` and `leslie_before_after.py` do not use CMGDB and run in any of these environments. All commands below run from this directory, in bash or zsh, and the option `-B` keeps Python from writing bytecode files into `scripts/`. The directory must lie in a checkout of the fork at 577d988 or later, since `bounds_regression_tests.py` runs `tests/test_conley_geometry_regressions.py` of that checkout.

The stored outputs assume that no `CMGDB_MAPGRAPH_*` variable is set, since `CMGDB_MAPGRAPH_CACHE` changes the counts of box map evaluations. `bounds_run_all.py`, `leslie_run_all.py` and `bounds_regression_tests.py` remove these variables, `bounds_mechanism.py` sets `CMGDB_MAPGRAPH_CACHE=0` itself, and the other scripts that use CMGDB list any such variable in the header of their outputs. Unset these variables before running the scripts directly.

## Part I: the bounds of the cubical complex

All outputs and figures, about 10 s:

```bash
$MA -B scripts/bounds_run_all.py \
    --build cmgdb-1.5.2=$UP \
    --build cmgdb-1.3.3_fork.6=$FK \
    --build cmgdb-1.5.3_fork.7.dev0-0e21503=$MG \
    --build cmgdb-1.5.3_fork.7.dev0-96642f2=$MA
```

For each `--build`, this runs `bounds_annotations.py`, `bounds_annotations.py --boxmap`, `bounds_references.py` and `bounds_mechanism.py` with the interpreter of that build, about 1 s each. Then it runs `bounds_table.py` and `bounds_figures.py` with its own interpreter, which needs numpy and matplotlib. A second run reproduced every output and figure byte for byte.

Regression tests, about 2 s each, on master and on the merge:

```bash
$MA -B scripts/bounds_regression_tests.py --tag cmgdb-1.5.3_fork.7.dev0-96642f2
$MG -B scripts/bounds_regression_tests.py --tag cmgdb-1.5.3_fork.7.dev0-0e21503
```

| script | computes | outputs | in the report |
|---|---|---|---|
| `bounds_annotations.py [--tag T] [--boxmap] [--cases E1 ...]` | annotations of the Morse sets that contain the fixed points of E1 to E6, and the evaluations off the bisection tree | `outputs/bounds_annotations_<tag>.*`, with `--boxmap` `outputs/bounds_annotations-boxmap_<tag>.*` | Tables 2 and 3 |
| `bounds_references.py [--tag T]` | the linearization with conditions (a) and (b) of Lemma 3.3, and the uniform runs | `outputs/bounds_references_<tag>.*` | Section 3.3, Table 2 |
| `bounds_mechanism.py [--tag T] [--cases E1 ...]` | complexes, bounds before and after the fix, boxes received by the box map, maps on cubes, `ComputeConleyIndex` on them, lift on the line | `outputs/bounds_mechanism_<tag>.*` | Tables 4 and 5, Sections 3.5 to 3.9 |
| `bounds_table.py [--tags T ...]` | tables across builds, from the outputs above | `outputs/bounds_table.*` | Tables 2, 3 and 4 |
| `bounds_figures.py [--before T] [--after T]` | figures from `bounds_mechanism_<tag>.json` of 1.5.2 and 96642f2 | `figures/bounds_e1_number_line.pdf`, `figures/bounds_e1_graph.pdf`, `figures/bounds_e4_saddle.pdf` | Figures 1, 2 and 3 |
| `bounds_regression_tests.py [--tag T]` | `tests/test_conley_geometry_regressions.py` with pytest | `outputs/bounds_regression-tests_<tag>.txt` | Section 3.9 |

`bounds_common.py` holds the examples and the rules of the code, and `bounds_lift1d.py` the lift on the line used by `bounds_mechanism.py`.

## Part II: the hierarchy with box maps that are not monotone

All outputs and figures, 30 jobs, about 60 s:

```bash
$MA -B scripts/leslie_run_all.py \
    --build cmgdb-1.5.3_fork.7.dev0-96642f2=$MA \
    --build cmgdb-1.5.2=$UP \
    --build cmgdb-1.3.3_fork.6=$FK \
    --build cmgdb-1.5.3_fork.7.dev0-0e21503=$MG \
    --heavy
```

The first `--build` is the reference: it also runs `leslie_dynamics.py`, `leslie_before_after.py` and the figure scripts, and `leslie_morse.py` gets `--boxes` there. The wall-clock time of each job goes to `outputs/leslie_run_all_timings.txt`, the only output that changes between runs. Two complete runs gave byte-identical outputs and figures otherwise.

Step by step:

```bash
$MA -B scripts/leslie_dynamics.py
for pair in "${MA}:cmgdb-1.5.3_fork.7.dev0-96642f2" "${UP}:cmgdb-1.5.2" "${FK}:cmgdb-1.3.3_fork.6" "${MG}:cmgdb-1.5.3_fork.7.dev0-0e21503"; do
    PY=${pair%%:*}; TAG=${pair##*:}
    $PY -B scripts/leslie_morse.py --tag $TAG --heavy
    $PY -B scripts/leslie_depth_one.py --tag $TAG
    $PY -B scripts/leslie_cycle.py --tag $TAG
    $PY -B scripts/leslie_isolation.py --tag $TAG
    $PY -B scripts/leslie_monotonicity.py --tag $TAG
    $PY -B scripts/leslie_origin.py --tag $TAG
done
$MA -B scripts/leslie_morse.py --tag cmgdb-1.5.3_fork.7.dev0-96642f2 --heavy --boxes
$MA -B scripts/leslie_before_after.py --tags cmgdb-1.5.3_fork.7.dev0-96642f2 cmgdb-1.5.2 cmgdb-1.3.3_fork.6 cmgdb-1.5.3_fork.7.dev0-0e21503
$MA -B scripts/leslie_fig_phase_portrait.py
$MA -B scripts/leslie_fig_depth_one.py --tag cmgdb-1.5.3_fork.7.dev0-96642f2
$MA -B scripts/leslie_fig_cycle.py --tag cmgdb-1.5.3_fork.7.dev0-96642f2
$MA -B scripts/leslie_fig_morse_sets.py --tag cmgdb-1.5.3_fork.7.dev0-96642f2
```

| script | computes | outputs | in the report | time |
|---|---|---|---|---|
| `leslie_dynamics.py` | fixed points, periodic orbits, the circle, basins, unstable manifolds, without CMGDB | `outputs/leslie_dynamics.*` | Section 4.1, data of Figure 4 | 14 s |
| `leslie_morse.py --tag T [--heavy] [--boxes]` | Morse sets, annotations, pairs, `ComputeConleyIndexForCells`, strongly connected components of the graph of the final grid. `--heavy` adds the uniform run of depth 18, and `--boxes` stores the boxes of Figure 5 | `outputs/leslie_morse_<tag>.*` | Table 7, last column of Table 11 | 3 s, 5 s with `--heavy` |
| `leslie_depth_one.py --tag T` | the nodes of depths 1 and 2, the chain of boxes that contain p | `outputs/leslie_depth_one_<tag>.*` | Section 4.3, Table 8 | under 1 s |
| `leslie_cycle.py --tag T` | the cycle through C, the pair of M, homology ranks, the 612-cell component, the run 16/18 | `outputs/leslie_cycle_<tag>.*` | Section 4.4, Tables 9 and 10 | 1 s |
| `leslie_isolation.py --tag T` | the hypotheses of Proposition 4.6 with the hull of f(B) | `outputs/leslie_isolation_<tag>.*` | Section 4.5 | 2 s |
| `leslie_monotonicity.py --tag T` | outer approximation and monotonicity on every evaluated box | `outputs/leslie_monotonicity_<tag>.*` | Table 11, Section 4.7 | 1.5 s |
| `leslie_origin.py --tag T` | the Morse set of the origin on six domains, its exit set, the cells of the exit set on the segment of the unstable line, and a point of the cell of the origin that f maps out of the domain | `outputs/leslie_origin_<tag>.*` | Section 4.8, Table 12 | 1 s |
| `leslie_before_after.py --tags REF T ...` | comparison of the outputs of the builds, without CMGDB | `outputs/leslie_before_after.*` | Section 4.9, Table 10 | under 1 s |
| `leslie_fig_phase_portrait.py` | figure | `figures/leslie_phase_portrait.pdf` | Figure 4 | under 1 s |
| `leslie_fig_morse_sets.py --tag T` | figure, from `leslie_morse.py --boxes` | `figures/leslie_morse_sets_16.pdf` | Figure 5 | 1.5 s |
| `leslie_fig_depth_one.py --tag T` | figure | `figures/leslie_depth_one.pdf` | Figure 6 | under 1 s |
| `leslie_fig_cycle.py --tag T` | figure | `figures/leslie_cycle_12_14.pdf` | Figure 7 | under 1 s |

`leslie_common.py` holds the map, the three box maps and the grid utilities, and `leslie_plotstyle.py` the plot style.

## Short programs

E1 on a single build. On 1.5.2, 1.3.3+fork.6 and 0e21503 the Morse set `[[0.9375, 1.015625]]` gets `['0', '0']`, and on 96642f2 `['0', 'x-1']`:

```python
import CMGDB

def F(rect):
    # f(x) = x/(2 - x) is increasing, so f([a, b]) = [f(a), f(b)]
    return [t / (2.0 - t) for t in rect]

model = CMGDB.Model(4, 8, 0, 10000, [0.0], [1.25], F)
morse_graph, map_graph = CMGDB.ComputeConleyMorseGraph(model)
for v in range(morse_graph.num_vertices()):
    print(morse_graph.morse_set_boxes(v), morse_graph.annotations(v))
```

The Leslie run 12/14 with the box spanned by f at the corners. Every build prints `0 525 ['0', 'x+1', '0']`, and `ComputeConleyIndexForCells` returns `['0', 'x+1', '0']` on 1.5.2, 1.3.3+fork.6 and 0e21503 and raises `ValueError` on 96642f2:

```python
import math
import CMGDB

def f(x):
    s = x[0] + x[1]
    return [(19.6 * x[0] + 23.68 * x[1]) * math.exp(-0.1 * s), 0.7 * x[0]]

def F(rect):
    return CMGDB.BoxMap(f, rect)  # f at the corners of rect, no padding

model = CMGDB.Model(12, 14, [-0.001, -0.001], [90.0, 70.0], F)
morse_graph, map_graph = CMGDB.ComputeConleyMorseGraph(model)
for v in range(morse_graph.num_vertices()):
    print(v, len(morse_graph.morse_set(v)), morse_graph.annotations(v))
try:
    print(CMGDB.ComputeConleyIndexForCells(model, morse_graph, morse_graph.morse_set(0)))
except ValueError as error:
    print("ValueError:", str(error)[:72])
```

Each takes under a second.

## The report

```bash
latexmk -pdf report.tex && latexmk -c report.tex
```

The second command removes the auxiliary files and keeps `report.pdf`.
