import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_cubical_fiber_geometry(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++17 compiler")
    repo = Path(__file__).resolve().parents[1]
    source = repo / "tests" / "cpp" / "test_conley_fiber_geometry.cpp"
    headers = repo / "src" / "CMGDB" / "_cmgdb" / "include"
    executable = tmp_path / "test_conley_fiber_geometry"
    include_paths = [headers, Path("/opt/homebrew/include")]
    for name in ("BOOST_ROOT", "CONDA_PREFIX"):
        if root := os.environ.get(name):
            include_paths.append(Path(root) / "include")
    if boost_include := os.environ.get("BOOST_INCLUDEDIR"):
        include_paths.append(Path(boost_include))
    includes = [f"-I{path}" for path in include_paths if path.is_dir()]
    build = subprocess.run(
        [compiler, "-std=c++17", "-O2", *includes, str(source), "-o", str(executable)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    if "boost/" in build.stderr and (
        "No such file or directory" in build.stderr or "file not found" in build.stderr
    ):
        pytest.skip("requires Boost headers for the standalone C++ test")
    assert build.returncode == 0, build.stderr
    subprocess.run([str(executable)], check=True, timeout=10)
