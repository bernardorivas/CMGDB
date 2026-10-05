"""Regression tests for the build configuration (review findings C50-C58)."""

import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def pyproject():
    with open(REPO / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)


def cmake_commands(path):
    """(name, arguments) of each command in a CMake listfile, comments removed."""
    text = re.sub(r"#[^\n]*", "", path.read_text())
    return [(match.group(1).lower(), match.group(2).split())
            for match in re.finditer(r"\b(\w+)\s*\(([^()]*)\)", text)]


def configure(build_dir, *arguments):
    """Configure the project into build_dir as scikit-build-core would, with
    these arguments in place of the cmake.args of pyproject.toml."""
    command = [shutil.which("cmake"), "-S", str(REPO), "-B", str(build_dir),
               "-DSKBUILD_PROJECT_NAME=CMGDB", "-DSKBUILD_PROJECT_VERSION=0.0",
               f"-DPython_EXECUTABLE={sys.executable}", *arguments]
    try:
        import pybind11
    except ImportError:
        pass  # CMake may still find a pybind11 installed elsewhere
    else:
        command.append(f"-Dpybind11_DIR={pybind11.get_cmake_dir()}")
    return subprocess.run(command, capture_output=True, text=True)


@pytest.fixture(scope="module")
def configured(tmp_path_factory):
    """A build directory configured with the cmake.args of pyproject.toml.

    The tests that configure the project skip when this fails: the machine
    then lacks CMake, a compiler, pybind11 or Boost."""
    if shutil.which("cmake") is None:
        pytest.skip("cmake is not on PATH")
    build_dir = tmp_path_factory.mktemp("configured")
    result = configure(build_dir, *pyproject()["tool"]["scikit-build"]["cmake"]["args"])
    if result.returncode != 0:
        pytest.skip("the project does not configure here:\n" + result.stderr[-2000:])
    return build_dir


def test_every_cmake_policy_is_set_only_where_cmake_knows_it():
    # scikit-build-core accepts any CMake from cmake_minimum_required's
    # minimum on, and a CMake that does not know a policy stops at
    # cmake_policy(SET) with an error, so each one sits under if(POLICY).
    # CMP0144 is CMake 3.27's: unguarded, Ubuntu 22.04's 3.22 and EL8's
    # 3.26 could not configure the project (C50).
    conditions = []
    unguarded = []
    for name, arguments in cmake_commands(REPO / "CMakeLists.txt"):
        if name == "if":
            conditions.append(arguments)
        elif name in ("elseif", "else"):
            conditions[-1] = arguments if name == "elseif" else []
        elif name == "endif":
            conditions.pop()
        elif name == "cmake_policy" and arguments[:1] == ["SET"]:
            if ["POLICY", arguments[1]] not in conditions:
                unguarded.append(arguments[1])
    assert unguarded == []


def test_configures_when_the_cmake_args_of_pyproject_are_overridden(configured, tmp_path):
    # -C cmake.args=... and SKBUILD_CMAKE_ARGS replace the cmake.args list of
    # pyproject.toml, which held the only USER_PROJECT_PATH; configure then
    # looked for the module source at //CMGDB.cpp and failed (C57).
    result = configure(tmp_path, "-DCMAKE_CXX_FLAGS=-DCMG_VERBOSE")
    assert result.returncode == 0, result.stderr[-2000:]
