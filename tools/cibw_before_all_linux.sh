#!/usr/bin/env bash
# System dependencies for building CMGDB wheels inside a manylinux container.
#
#   Boost   chrono / thread / serialization (plus header-only ublas and
#           property_tree). CMakeLists.txt finds Boost in CONFIG mode, which
#           needs the BoostConfig.cmake that Boost ships from 1.70 on; the
#           image's own boost-devel is 1.66, so take EPEL's boost1.78-devel,
#           which installs /usr/lib64/cmake/Boost-1.78.0 on CMake's default
#           search path.
#   graphviz supplies the `dot` binary that the plotting tests shell out to.
#
# sdsl-lite v3 is vendored under src/CMGDB/_cmgdb/third_party, and GMP is
# optional (USE_GMP), so neither is installed here.
set -euxo pipefail

dnf install -y boost1.78-devel graphviz

# Fail loudly here rather than at CMake's find_package.
test -f /usr/lib64/cmake/Boost-1.78.0/BoostConfig.cmake
test -d /usr/include/boost1.78/boost/serialization
