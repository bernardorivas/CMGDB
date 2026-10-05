"""Regression tests for the build configuration (review findings C50-C58)."""

import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

REPO = Path(__file__).resolve().parents[1]


def pyproject():
    with open(REPO / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)


def pyproject_cmake_args():
    return pyproject()["tool"]["scikit-build"]["cmake"]["args"]


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
    then lacks CMake, a compiler, pybind11 or Boost. The test extra installs
    pybind11."""
    if shutil.which("cmake") is None:
        pytest.skip("cmake is not on PATH")
    build_dir = tmp_path_factory.mktemp("configured")
    # Ask CMake's file API for the code model, which reports the language
    # standard of each target.
    query = build_dir / ".cmake" / "api" / "v1" / "query" / "codemodel-v2"
    query.parent.mkdir(parents=True)
    query.touch()
    result = configure(build_dir, *pyproject_cmake_args())
    if result.returncode != 0:
        pytest.skip("the project does not configure here:\n" + result.stderr[-2000:])
    return build_dir


def stand_in_boost(root):
    """A Boost 1.66 as FindBoost sees one, with no CMake package files (as
    EL8's boost-devel): version.hpp and, for chrono, thread, serialization
    and the components they depend on, the header FindBoost checks and the
    library."""
    headers = ["config.hpp", "atomic.hpp", "chrono.hpp", "date_time/date.hpp",
               "serialization/serialization.hpp", "system/config.hpp", "thread.hpp"]
    for header in headers:
        (root / "include" / "boost" / header).parent.mkdir(parents=True, exist_ok=True)
        (root / "include" / "boost" / header).touch()
    (root / "include" / "boost" / "version.hpp").write_text(
        '#define BOOST_VERSION 106600\n#define BOOST_LIB_VERSION "1_66"\n')
    (root / "lib").mkdir()
    # MSVC links Boost statically (CMakeLists.txt), from libboost_*.lib.
    suffix = ".lib" if sys.platform == "win32" else ".a"
    for component in ("atomic", "chrono", "date_time", "serialization", "system", "thread"):
        (root / "lib" / f"libboost_{component}{suffix}").touch()
    return root


def cache_entry(build_dir, name):
    """The value of a CMakeCache.txt entry, or None."""
    match = re.search(rf"^{name}:[A-Z]+=(.*)$",
                      (build_dir / "CMakeCache.txt").read_text(), re.MULTILINE)
    return match.group(1) if match else None


def language_standards(build_dir, target):
    """The language standards that CMake's file API reports for a target."""
    reply = build_dir / ".cmake" / "api" / "v1" / "reply"
    index = json.loads(max(reply.glob("index-*.json")).read_text())
    codemodel = next(entry for entry in index["objects"] if entry["kind"] == "codemodel")
    model = json.loads((reply / codemodel["jsonFile"]).read_text())
    entry = next(entry for entry in model["configurations"][0]["targets"]
                 if entry["name"] == target)
    groups = json.loads((reply / entry["jsonFile"]).read_text()).get("compileGroups", [])
    return {group["languageStandard"]["standard"]
            for group in groups if "languageStandard" in group}


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


def test_compiles_as_cxx17(configured):
    # C++20 removes members of std::allocator that the ublas and property_tree
    # headers of older Boost releases use, so Ubuntu 22.04's Boost 1.74 and
    # EL8's 1.66 did not compile; the vendored sdsl-lite needs C++17 (C58).
    standards = language_standards(configured, "_cmgdb")
    if not standards:
        pytest.skip("this CMake's file API reports no language standard")
    assert standards == {"17"}


def test_finds_a_boost_that_has_no_cmake_package_file(configured, tmp_path):
    # Boost installs BoostConfig.cmake only from 1.70 on, and a CONFIG-only
    # find_package rejected older releases, such as EL8's 1.66, at configure
    # (C58). A BoostConfig.cmake that reports Boost as not found shadows
    # any this machine has, and Boost_ROOT, which CMake searches before any
    # BOOST_ROOT of the environment, points FindBoost at a stand-in 1.66.
    # (CMAKE_IGNORE_PATH cannot hide a file that CMake reaches again through
    # a symlinked prefix, as /lib64 is on manylinux_2_28.)
    boost = stand_in_boost(tmp_path / "boost-1.66")
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    (hidden / "BoostConfig.cmake").write_text("set(Boost_FOUND FALSE)\n")
    build_dir = tmp_path / "build"
    result = configure(build_dir, *pyproject_cmake_args(), f"-DBoost_DIR={hidden.as_posix()}",
                       f"-DBoost_ROOT={boost.as_posix()}", "-DBoost_NO_SYSTEM_PATHS=ON")
    assert result.returncode == 0, result.stderr[-2000:]
    include_dir = cache_entry(build_dir, "Boost_INCLUDE_DIR")
    assert include_dir and Path(include_dir).resolve() == (boost / "include").resolve()


def test_test_extra_installs_pybind11_for_the_configure_tests():
    # The tests above that configure the project skip when CMake finds no
    # pybind11. The wheel test environments (cibuildwheel's test-extras)
    # have none but what the test extra installs, so without it those tests
    # never ran there (C57, C58).
    extra = pyproject()["project"]["optional-dependencies"]["test"]
    assert "pybind11" in {Requirement(line).name for line in extra}


def test_matplotlib_requirement_excludes_releases_without_poly3d_shade():
    # PlotMorseSets3D passes Poly3DCollection the shade keyword, which
    # matplotlib 3.7 added; pyproject.toml admitted 3.6, where every call
    # raised AttributeError (C03, C56). Every plotting test passes on 3.7.0.
    requirement = next(Requirement(line) for line in pyproject()["project"]["dependencies"]
                       if Requirement(line).name == "matplotlib")
    assert not requirement.specifier.contains("3.6.0")
    assert not requirement.specifier.contains("3.6.3")
    assert requirement.specifier.contains("3.7.0")


def test_build_backend_requirement_excludes_releases_without_pep639():
    # project.license is an SPDX expression (PEP 639), which scikit-build-core
    # parses from 0.11 on; under 0.10.x every build failed before CMake ran
    # (C51, fixed by the merge).
    requirement = next(Requirement(line) for line in pyproject()["build-system"]["requires"]
                       if Requirement(line).name == "scikit-build-core")
    assert not requirement.specifier.contains("0.10.7")
    assert requirement.specifier.contains("0.11.0")


def test_installed_package_carries_the_notices_of_the_code_compiled_in():
    # The extension compiles in CHOMP (MIT) and sdsl-lite (BSD-3-Clause), whose
    # notices wheel.exclude keeps out of the wheel with the rest of their
    # source; license-files ships copies of them (C52, fixed by the merge).
    files = importlib.metadata.files("CMGDB")
    if files is None:
        pytest.skip("CMGDB is not installed with a file list")
    notices = {
        "licenses/chomp-LICENSE": "src/CMGDB/_cmgdb/include/chomp/LICENSE",
        "licenses/sdsl-lite-xxsds-LICENSE": "src/CMGDB/_cmgdb/third_party/sdsl-lite/LICENSE",
    }
    for shipped, source in notices.items():
        copy = next((path for path in files
                     if path.as_posix().endswith(".dist-info/licenses/" + shipped)), None)
        assert copy is not None, shipped
        assert copy.read_text() == (REPO / source).read_text(), shipped
        assert (REPO / shipped).read_text() == (REPO / source).read_text(), shipped
