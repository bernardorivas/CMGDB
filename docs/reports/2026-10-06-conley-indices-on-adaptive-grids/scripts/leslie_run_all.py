"""Run every Leslie script (Part II) and record how long each one takes.

Each --build gives a tag and the Python interpreter of an environment with
one CMGDB build.  The first build is the reference: it also runs the
dynamics, the comparison across builds and the figures.

    python leslie_run_all.py \\
        --build cmgdb-1.5.3_fork.7.dev0-96642f2=/path/to/venv-master/bin/python \\
        --build cmgdb-1.5.2=/path/to/venv-upstream/bin/python \\
        --build cmgdb-1.3.3_fork.6=/path/to/venv-fork/bin/python \\
        --build cmgdb-1.5.3_fork.7.dev0-0e21503=/path/to/venv-merge/bin/python \\
        [--heavy] [--no-figures]

--heavy adds the uniform depth-18 decomposition to leslie_morse.py, which
also stores the boxes of the figures (--boxes) for the first build.  The
wall-clock times go to outputs/leslie_run_all_timings.txt.  The other
outputs do not depend on the run (no times, dates or paths in them).  The
scripts run without the CMGDB_MAPGRAPH_* variables of the calling
environment, which change the number of box map evaluations.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Default options: no CMGDB_MAPGRAPH_* variable from the calling environment,
# as in bounds_run_all.py.
ENV = {k: v for k, v in os.environ.items() if not k.startswith("CMGDB_MAPGRAPH_")}
PER_BUILD = ["leslie_morse.py", "leslie_depth_one.py", "leslie_cycle.py", "leslie_isolation.py",
             "leslie_monotonicity.py", "leslie_origin.py"]
FIGURES = ["leslie_fig_phase_portrait.py", "leslie_fig_depth_one.py", "leslie_fig_cycle.py",
           "leslie_fig_morse_sets.py"]


def main():
    p = argparse.ArgumentParser(description="Run every Leslie script.")
    p.add_argument("--build", action="append", required=True, metavar="TAG=PYTHON")
    p.add_argument("--heavy", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()
    builds = [tuple(b.split("=", 1)) for b in args.build]
    ref_tag, ref_py = builds[0]
    jobs = [(ref_py, "leslie_dynamics.py", [])]
    for tag, py in builds:
        for s in PER_BUILD:
            extra = []
            if s == "leslie_morse.py":
                extra += ["--heavy"] if args.heavy else []
                extra += ["--boxes"] if tag == ref_tag else []
            jobs.append((py, s, ["--tag", tag] + extra))
    jobs.append((ref_py, "leslie_before_after.py", ["--tags"] + [t for t, _ in builds]))
    if not args.no_figures:
        jobs += [(ref_py, s, ["--tag", ref_tag]) for s in FIGURES]
    lines = []
    for py, script, extra in jobs:
        t0 = time.perf_counter()
        r = subprocess.run([py, "-B", str(HERE / script)] + extra, cwd=HERE, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, text=True, env=ENV)
        dt = time.perf_counter() - t0
        line = f"{script} {' '.join(extra)}: {dt:.1f} s" + ("" if r.returncode == 0 else f"  FAILED ({r.returncode})")
        print(line, flush=True)
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
        lines.append(line)
    (HERE.parent / "outputs" / "leslie_run_all_timings.txt").write_text(
        "wall-clock time of each script (this machine)\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
