"""Regression tests for the benchmark tools (review finding C55)."""

import importlib.util
import shutil
import subprocess
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def load_benchmark_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "benchmarks" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

