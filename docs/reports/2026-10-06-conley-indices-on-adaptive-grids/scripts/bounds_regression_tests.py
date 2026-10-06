"""Run the repository's regression tests for the bounds of the complex on the installed build.

The tests are tests/test_conley_geometry_regressions.py, added with the fix
(577d988).  They check the Conley indices of the fixed points of x/(2-x) on
E1, E2, E4 and E5 and on the model of
examples/Lattice_and_Nontrivial_CMGraph.ipynb through PrecomputedBoxMap, and
that every rectangle the map receives in ComputeConleyMorseGraph is a box of
the bisection tree.  The script runs pytest without its cache and writes the
summary to outputs/bounds_regression-tests_<TAG>.txt.

Usage:  python bounds_regression_tests.py [--tag TAG]
Needs pytest in the environment of the build.  Runtime: about 2 s.
"""

import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bounds_common as C  # noqa: E402

TESTS = C.HERE.parents[3] / "tests" / "test_conley_geometry_regressions.py"


def main():
    args = C.parse_args(__doc__)
    out = C.Output("bounds_regression-tests", args.tag)
    out(*C.header_lines(__file__, args.tag, env=False))
    out(f"tests: {TESTS.relative_to(C.HERE.parents[3])}", "")
    env = {k: v for k, v in os.environ.items() if not k.startswith("CMGDB_MAPGRAPH_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    res = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rA", str(TESTS)],
                         capture_output=True, text=True, env=env, cwd=str(C.HERE))
    lines = [l for l in res.stdout.splitlines() if l.startswith(("PASSED", "FAILED", "ERROR"))]
    for l in lines:
        status, _, test = l.partition(" ")
        out(f"  {status} {test.split(' - ')[0].split('::', 1)[-1]}")
    summary = [l for l in res.stdout.splitlines() if " passed" in l or " failed" in l]
    out("", "summary: " + (summary[-1].strip("= ").split(" in ")[0] if summary else res.stdout[-200:]))
    out.close()


if __name__ == "__main__":
    main()
