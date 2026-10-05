"""Regression tests for the build configuration (review findings C50-C58)."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def cmake_commands(path):
    """(name, arguments) of each command in a CMake listfile, comments removed."""
    text = re.sub(r"#[^\n]*", "", path.read_text())
    return [(match.group(1).lower(), match.group(2).split())
            for match in re.finditer(r"\b(\w+)\s*\(([^()]*)\)", text)]


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
