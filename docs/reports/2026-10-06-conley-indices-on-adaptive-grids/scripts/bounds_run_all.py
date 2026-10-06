"""Run every script of Part I on several CMGDB builds, then the tables and the figures.

Usage:

    python bounds_run_all.py --build TAG=PYTHON [--build TAG=PYTHON ...]
                          [--before TAG] [--after TAG]

Each --build names a build tag and the Python interpreter of a virtual
environment in which that CMGDB build is installed, for example

    --build cmgdb-1.5.2=/path/to/venv-upstream/bin/python

With that interpreter the script runs bounds_annotations.py (with and without
--boxmap), bounds_references.py and bounds_mechanism.py, each with --tag TAG.  Then,
with the interpreter that runs this script (which needs matplotlib, not
CMGDB), it runs bounds_table.py on the tags in the order given, and
bounds_figures.py with the rectangles recorded on the builds --before (default
cmgdb-1.5.2) and --after (default cmgdb-1.5.3_fork.7.dev0-96642f2).

The builds of the report and how to install them:

    cmgdb-1.5.2                       pip install CMGDB==1.5.2
    cmgdb-1.3.3_fork.6                pip install cmgdb==1.3.3+fork.6 --find-links
        https://github.com/bernardorivas/CMGDB/releases/expanded_assets/v1.3.3%2Bfork.6
    cmgdb-1.5.3_fork.7.dev0-0e21503   pip install git+https://github.com/bernardorivas/CMGDB@0e21503
    cmgdb-1.5.3_fork.7.dev0-96642f2   pip install git+https://github.com/bernardorivas/CMGDB@96642f2

The last two report the same version string, so their tags carry the commit.
Runtime: about 10 s for the four builds.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Default options: no CMGDB_MAPGRAPH_* variable from the calling environment
# (bounds_mechanism.py sets CMGDB_MAPGRAPH_CACHE itself).
ENV = {k: v for k, v in os.environ.items() if not k.startswith("CMGDB_MAPGRAPH_")}
ENV["PYTHONDONTWRITEBYTECODE"] = "1"


def run(python, script, *args):
    cmd = [python, str(HERE / script), *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, env=ENV)


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--build", action="append", required=True, metavar="TAG=PYTHON")
    ap.add_argument("--before", default="cmgdb-1.5.2")
    ap.add_argument("--after", default="cmgdb-1.5.3_fork.7.dev0-96642f2")
    args = ap.parse_args()
    builds = [b.split("=", 1) for b in args.build]
    for tag, python in builds:
        run(python, "bounds_annotations.py", "--tag", tag)
        run(python, "bounds_annotations.py", "--boxmap", "--tag", tag)
        run(python, "bounds_references.py", "--tag", tag)
        run(python, "bounds_mechanism.py", "--tag", tag)
    run(sys.executable, "bounds_table.py", "--tags", *[t for t, _ in builds])
    run(sys.executable, "bounds_figures.py", "--before", args.before, "--after", args.after)


if __name__ == "__main__":
    main()
