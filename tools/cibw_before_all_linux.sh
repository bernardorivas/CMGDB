#!/usr/bin/env bash
# System dependencies for building CMGDB wheels inside a manylinux container.
#
#   Boost   chrono / thread / serialization (plus header-only ublas and
#           property_tree). EPEL's boost1.78-devel installs
#           /usr/lib64/cmake/Boost-1.78.0 on CMake's default search path, so
#           CMakeLists.txt finds it through its BoostConfig.cmake. The
#           image's own boost-devel, 1.66, has no such file; CMakeLists.txt
#           would find it through FindBoost, but the wheels take the newer
#           release.
#   graphviz supplies the `dot` binary that the plotting tests shell out to.
#
# sdsl-lite v3 is vendored under src/CMGDB/_cmgdb/third_party, and GMP is
# optional (USE_GMP), so neither is installed here.
set -euxo pipefail

dnf install -y boost1.78-devel graphviz

# Fail loudly here rather than at CMake's find_package.
test -f /usr/lib64/cmake/Boost-1.78.0/BoostConfig.cmake
test -d /usr/include/boost1.78/boost/serialization
