# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

CMGDB (Conley Morse Graph Database): computes Morse graphs and Conley indices of maps on rectangular phase spaces by adaptive cubical subdivision. A header-only C++ core (`include/database/` plus the CHomP-derived `include/chomp/`) is exposed to Python through pybind11 as `CMGDB._cmgdb`.

## Build and test

```bash
./install.sh                          # rm -rf build dist; pip install . --force-reinstall --no-deps --no-cache-dir
uv pip install --no-deps .            # in a uv venv (it has no pip); always rebuilds the given tree
python -m pytest tests -q
python -m pytest tests/test_basic.py::test_name -q
python tests/bench.py                 # quick suite; --heavy, --scenarios a,b --repeats 5 --warmup 1, --list
```

- The install is non-editable, and the tests import the installed package (src layout, no conftest). Reinstall after any change, Python or C++. A clean build takes about 30-40 s (one translation unit) and the suite about 7 s. `pip install -e .` also works: Python edits are then live, and C++ edits need the install rerun.
- Requirements: Boost (chrono, thread, serialization), GMP headers (`chomp/Ring.h` includes `gmpxx.h`; libgmp is not linked) and SDSL v3 (xxsds). When `sdsl/cereal.hpp` is not found, CMake fetches a pinned SDSL commit at configure time, so a clean build needs network access; `CMAKE_ARGS=-DSDSL_INCLUDE_DIR=<v3>/include` skips the fetch. SDSL v2 (simongog) is GPL and must never be used; Homebrew installs it under `/opt/homebrew/include`, which is why the v3 headers are added with `BEFORE`.
- `setup.py` passes `pybind11_DIR` (the build environment's pybind11) and `Python_EXECUTABLE`. Keep that spelling: `PYBIND11_FINDPYTHON` uses CMake's FindPython, which ignores `PYTHON_EXECUTABLE`. The `CMAKE_ARGS` environment variable is forwarded, but `-O3 -DNDEBUG` is appended after it and overrides `CMAKE_CXX_FLAGS_RELEASE`.
- A tree copied together with its `build/` fails to configure ("source ... does not match"). Remove `build/` first.
- Two tests compile `tests/cpp/*.cpp` with the system `c++` against the checkout's headers, not the installed ones, so they see header edits without a reinstall. Keep `MorseSetReachabilityCore.h` and `CarrierChainMap.h` free of Boost, SDSL and pybind11 includes. `tests/cpp/test_atlas_active_subgrid.cpp` is not run by pytest; it needs Boost, SDSL v3 (ahead of `/opt/homebrew/include`) and the Python headers and libraries.
- `tests/test_derived_graph_plot.py` needs graphviz `dot` on `PATH`; the 2 skips are tests that need torch.
- `CMG_VERBOSE` turns on progress output. Uncomment `// #define CMG_VERBOSE` at the top of `CMGDB.cpp` (it must stay above the includes) or pass `-DCMG_VERBOSE`. Without it a failed Conley-index computation is silent: `annotations(v)` returns `[]`, the same as "not computed".

## Architecture

### C++ core (`src/CMGDB/_cmgdb/`)

- `CMGDB.cpp` is the only translation unit. Besides the bindings it holds about 1100 lines of non-inline wrappers, and several headers define non-inline globals and functions, so a second `.cpp` would fail to link.
- To expose a function: write it `inline` in a header under `include/database/`, include it in `CMGDB.cpp`, and add an `m.def` (or a `FooBinding(m)` call) in `PYBIND11_MODULE`. Pattern: copy the arguments, run the C++ work inside `py::gil_scoped_release`, build Python objects after. Names starting with `_` are skipped by the star-import in `__init__.py`.
- `Model` holds a `Configuration` (bounds, periodic flags, `subdiv_min/max/init/limit`) and the map. The phase grid is `PointerGrid`, a `TreeGrid`. `model.phaseSpace()` returns a fresh unsubdivided grid, never the computed one. `Model` does no validation: `init > min` or `max < min` silently yields an empty Morse graph (`AtlasModel` does check).
- `Compute_Morse_Graph.hpp`: subdivide `init` times, then process decomposition nodes from a priority queue. Each node builds a `MapGraph` over its grid, takes SCCs (Tarjan) and reachability, and spawns one child per Morse set (its subgrid subdivided once) until depth `max - init`. Morse-graph vertices are created at depth `min - init`; deeper levels only decide spuriousness (a Morse set is dropped when every refinement branch ends in an empty recurrent set). `limit` stops refining nodes larger than it. The results are joined into one non-uniform grid.
- One subdivision bisects every leaf along axis `depth % dim`. Box indices are DFS leaf order (Morton order), not row-major.
- `ComputeMorseGraph` and `ComputeConleyMorseGraph` return `(MorseGraph, MapGraph)`; that `MapGraph` costs a second box-map pass over the final grid, which the `*Only` variants skip. `MapGraph` vertex ids, `morse_set(v)` entries and `phase_space_box(i)` indices all refer to the same grid object (the comment near `CMGDB.cpp:280` saying otherwise is wrong). `phase_space_box` does not bounds-check.
- Morse-graph vertices are numbered so that every edge `u -> v` has `u > v`. `edges()` and `adjacencies(v)` recompute the transitive reduction on every call.
- Box-map callbacks take and return a flat `[lo_0..lo_{d-1}, hi_0..hi_{d-1}]`. The scalar path does not check the returned length (a short list is undefined behavior); the batch path (`Model.set_batch_map`) does. The batch map is used by the eager `MapGraph` cache (chunks of 100000) and `ComputeMorseSetReachability`. The scalar map is always used by the Conley index, `ComputeConleyIndexForCells` and lazy graphs, so the two must compute the same map.
- Covers clip images to the domain: escape is dropped, not modeled as a sink. Periodic axes are wrapped by at most one period, so the callback must reduce periodic coordinates.
- The `CMGDB_MAPGRAPH_*` environment variables are read at every `MapGraph` construction, including the intermediate per-node graphs. `CMGDB_MAPGRAPH_CACHE=0` gives lazy graphs, which the native reachability functions and `csr_view` refuse. See the README ("Cache sizing") and `docs/map_graph_csr.md`.
- Conley indices (`include/chomp/`): everything is over F_5: `Zp<5>` in `chomp/Ring.h`, the tables in `GF5.h`, `coefficient_field = 5` in results, and `ComputeCarrierChainMap` rejects other moduli. Change them together. Morse nodes store induced matrices; `annotations(v)` recomputes the shift-class strings on every call (Frobenius form in a `boost::thread`, 3600 s timeout per degree). There is one string per degree `0..dim`: the invariant factors with powers of `x` removed, concatenated; `"0"` means nilpotent.
- `Zp::operator<` always returns false (a hack for `Bezout`), so never sort ring values, and `SmithNormalForm` over Z_5 can fail to terminate. CHomP calls `exit(1)` on some internal errors, which kills the Python process.
- Known defect, shared with upstream: the `SmithSolve` step in `RelativeMapHomology.h` gives wrong induced maps when the coreduction Morse complex is not minimal. It affects `ComputeConleyIndex` and `ComputeConleyIndexForCells`. For explicit chain complexes use `ComputeRelativeShiftClass`, which solves over the field, not `ComputeRelativeHomologyShiftClass`. Details: `~/Work/Projects/computational-hybrid/notes/2026-09-29-cmgdb-upstream-report/`.
- `ComputeConleyIndexForCells` uses X = cover(F(S)) and A = X \ S; this is the pair (S ∪ F(S), F(S) \ S) only when S ⊆ F(S), as for Morse sets.
- The fork's algebra kernels (`ExplicitChainComplex.h`, `RelativeShiftClass.h`, `CarrierChainMap.h`, `GF5.h`) use only the standard library. `RelativeShiftClass.h` re-implements the shift-class string format (`FormatPolynomial`, `ShiftClassString`); keep it in sync with `chomp/PolyRing.h` and `conleyIndexString.h`, since tests compare strings.
- `AtlasModel` (`Atlas*.h`, `docs/atlas_model.md`) is a disjoint union of `PointerGrid` charts. The call order is `add_chart`, then optionally `set_active_subgrid`, then `set_map`. Vertex ids are contiguous per-chart blocks ordered by chart id over the nonempty charts. It has no Conley path (the overloads throw), and `phaseSpace()` is the initial grid; use `phase_space_chart_box` and `morse_set_chart_boxes`.
- `ComputeMorseSetReachability` (`MorseSetReachability*.h`) re-implements `TreeGrid` geometry and cover in `ImplicitFixedTreeGrid`; mirror any change to `TreeGrid` there.

### Python layer (`src/CMGDB/`)

- `__init__.py` star-imports `_cmgdb` first and then each module. `CMGDB.MorseGraph` is therefore the DOT-parser dataclass from `morse_graph_parser`, shadowing the native class (`CMGDB._cmgdb.MorseGraph`). A new module needs an `__all__`, a line in `__init__.py`, and names that do not collide with native ones.
- Box-map builders, passed to `Model` or `AtlasModel`: `ComputeBoxMap.BoxMap` (pointwise `f`), `BoxMapData` (from sample pairs), `PrecomputedBoxMap.make_precomputed_box_map` (batched `f` or a `torch.nn.Module`, evaluated once on a lattice table; returns a callable with `.batch`, which the caller must install with `model.set_batch_map(bm.batch)`), and `PrecomputedAtlasBoxMap` (exact cache keyed by `float.hex`).
- `make_precomputed_box_map` pitfalls: `f` maps `(n, d)` to `(n, d)`; `padding` defaults to True (False in `BoxMap`); `mode="uniform"` is valid only when `init == min == max == subdiv_max`.
- Output: `PlotMorseGraph`, `PlotMorseSets`, `SaveMorseData`/`LoadMorseSetFile`. The CSV files have 6 significant digits; use `morse_set_boxes(v)` for exact geometry.
- Graph post-processing works on the DOT-parsed DAG: `morse_graph_parser`, `morse_lattice`, `derived_graph_plot`, `cmgdb_roa`. The bridge is to write `PlotMorseGraph(mg).source` to a file and call `MorseGraph.from_dot(path)`. Conley labels appear only for graphs from `ComputeConleyMorseGraph`.

## Fork and releases

`origin` is `bernardorivas/CMGDB`, a fork of `marciogameiro/CMGDB` (remote `upstream`), which publishes to PyPI as `CMGDB` under the same import name.

- The version lives only in `setup.py`. Development versions are `X+fork.N.dev0`; a release commit `release: X+fork.N` sets `X+fork.N` and is tagged `vX+fork.N`.
- `.github/workflows/wheels.yml` builds on a tag push and attaches the wheels to a GitHub release. It builds cp311-cp313 wheels for manylinux_2_28 x86_64 and macOS arm64 (configured under `[tool.cibuildwheel]` in `pyproject.toml`), and runs `tools/wheel_smoke_test.py` plus the test suite. Pushes to `master` or `wheels-ci` that touch build inputs build without releasing.
- The SDSL commit is pinned in both `CMakeLists.txt` and `tools/cibw_before_all_linux.sh`, and its BSD notice ships as `licenses/sdsl-lite-xxsds-LICENSE`; update all three together.
- Downstream projects pin the local version, which is what selects the fork: `pip install cmgdb==1.3.3+fork.6 --find-links https://github.com/bernardorivas/CMGDB/releases/expanded_assets/v1.3.3%2Bfork.6`.
