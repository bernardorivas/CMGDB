"""Regression tests for the benchmark tools (review findings C54, C55)."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def load_benchmark_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "benchmarks" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Put on PYTHONPATH, this makes ComputeMorseGraph return a Morse graph whose
# largest Morse set has its last box swapped for a box outside every Morse
# set, between that set's first and last boxes: its size and its minimum box
# stay the same.
SWAP_A_BOX = """
import CMGDB

_compute = CMGDB.ComputeMorseGraph


class SwappedBox:
    def __init__(self, morse_graph):
        self._graph = morse_graph
        sets = [sorted(morse_graph.morse_set(v)) for v in range(morse_graph.num_vertices())]
        self._vertex = max(range(len(sets)), key=lambda v: len(sets[v]))
        taken = set().union(*map(set, sets))
        boxes = sets[self._vertex]
        outside = next(b for b in range(boxes[0] + 1, boxes[-1]) if b not in taken)
        self._boxes = sorted(boxes[:-1] + [outside])

    def morse_set(self, v):
        return self._boxes if v == self._vertex else self._graph.morse_set(v)

    def __getattr__(self, name):
        return getattr(self._graph, name)


def ComputeMorseGraph(*args, **kwargs):
    morse_graph, map_graph = _compute(*args, **kwargs)
    return SwappedBox(morse_graph), map_graph


CMGDB.ComputeMorseGraph = ComputeMorseGraph
"""


def run_benchmark(scenario, *arguments, pythonpath=None,
                  script=REPO / "benchmarks" / "benchmark.py"):
    env = dict(os.environ)
    if pythonpath is not None:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(pythonpath), env.get("PYTHONPATH")]))
    return subprocess.run([sys.executable, str(script),
                           "--scenario", scenario, "--repeat", "1", *arguments],
                          capture_output=True, text=True, env=env)


def git(repo, *arguments):
    return subprocess.run(["git", "-C", str(repo), *arguments], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_compare_versions_builds_the_revisions_it_reports(tmp_path, monkeypatch):
    # compare_versions.py kept one old worktree and two venvs whatever
    # --old-rev named and however the working tree had changed, while its
    # table named the requested revision and the current tree (C55).
    compare_versions = load_benchmark_script("compare_versions")
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    for revision in ("revA", "revB"):
        (repo / "REV").write_text(revision)
        git(repo, "add", "REV")
        git(repo, "-c", "user.name=test", "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false", "commit", "-q", "-m", revision)
        git(repo, "tag", revision)

    installs = []

    def run(command, *args, **kwargs):
        # git runs; making a venv and installing into it are only recorded,
        # with the revision of the tree that pip would install.
        if command[0] == "git":
            return subprocess.run(command, *args, **kwargs)
        if command[1:3] == ["-m", "venv"]:
            (Path(command[3]) / "bin").mkdir(parents=True, exist_ok=True)
            (Path(command[3]) / "bin" / "python").touch()
        elif "install" in command:
            source = Path(command[-1])
            installs.append(("new" if source == repo else "old", (source / "REV").read_text()))
        else:
            raise AssertionError(f"unexpected command {command}")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(compare_versions, "REPO", repo)
    monkeypatch.setattr(compare_versions, "subprocess",
                        types.SimpleNamespace(run=run, STDOUT=subprocess.STDOUT))
    work = tmp_path / "work"
    work.mkdir()

    compare_versions.build_versions("revA", work)
    assert installs == [("old", "revA"), ("new", "revB")]

    installs.clear()
    (repo / "REV").write_text("edited")
    compare_versions.build_versions("revB", work)
    assert installs == [("old", "revB"), ("new", "edited")]

    # A revision already built is reused; the working tree is not.
    installs.clear()
    compare_versions.build_versions("revA", work)
    assert installs == [("new", "edited")]


def test_benchmark_gate_fails_a_morse_set_with_other_boxes(tmp_path):
    # The gate compared only the size and the minimum box of each Morse set,
    # so a Morse set with other boxes, but as many and the same minimum,
    # passed as OK (C54). benchmarks/references.digests.json pins the boxes.
    result = run_benchmark("leslie2d_python_batch")
    assert result.returncode == 0, result.stdout + result.stderr
    (tmp_path / "sitecustomize.py").write_text(SWAP_A_BOX)
    result = run_benchmark("leslie2d_python_batch", pythonpath=tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "VALIDATION FAILED for leslie2d_python_batch" in result.stdout
    assert "morse_set_digests: other boxes in Morse sets [1]" in result.stdout


def test_benchmark_keeps_the_digests_of_other_references_beside_them(tmp_path):
    # The Morse set digests of a references file named by --refs live beside
    # it, so --update-refs --refs leaves benchmarks/references.digests.json
    # alone and the gate does not check another file's references against
    # it. The script runs from a copy of the benchmarks whose digests of the
    # scenario are spoiled, so a gate that reads them fails.
    bench = tmp_path / "benchmarks"
    bench.mkdir()
    for name in ("benchmark.py", "references.json", "references.digests.json"):
        shutil.copy(REPO / "benchmarks" / name, bench / name)
    script = bench / "benchmark.py"
    default_digests = bench / "references.digests.json"
    digests = json.loads(default_digests.read_text())
    digests["product2d_adaptive"] = ["0" * 64] * len(digests["product2d_adaptive"])
    default_digests.write_text(json.dumps(digests, indent=1))
    spoiled = default_digests.read_bytes()

    alt = tmp_path / "alt.json"
    result = run_benchmark("product2d_adaptive", "--update-refs", "--refs", str(alt),
                           script=script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert default_digests.read_bytes() == spoiled
    assert list(json.loads((tmp_path / "alt.digests.json").read_text())) == ["product2d_adaptive"]

    result = run_benchmark("product2d_adaptive", "--refs", str(alt), script=script)
    assert result.returncode == 0, result.stdout + result.stderr
    result = run_benchmark("product2d_adaptive", script=script)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "VALIDATION FAILED for product2d_adaptive" in result.stdout
