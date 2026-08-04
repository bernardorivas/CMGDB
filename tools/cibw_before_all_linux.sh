#!/usr/bin/env bash
# System dependencies for building CMGDB wheels inside a manylinux container.
#
#   Boost   chrono / thread / serialization / ublas / property_tree
#           (>= 1.56: boost/serialization/unordered_set.hpp and shared_ptr
#           serialization are used, so the CentOS 7 era 1.53 is too old)
#   GMP     chomp/Ring.h includes <gmpxx.h> unconditionally, and nothing in
#           CMakeLists.txt asks for it, so a missing header fails deep in the
#           compile rather than at configure time
#   SDSL    v3 is header-only, so it is dropped into /usr/local/include, which
#           CMakeLists.txt already searches; no library is built or linked
set -euxo pipefail

# xxsds/sdsl-lite master, 2026-07-01. This is the maintained BSD-3 successor to
# simongog/sdsl-lite (unmaintained since 2019, GPLv3, static library). Header
# only, so the wheel stays MIT rather than inheriting a copyleft obligation.
SDSL_COMMIT=e6c417391f55476c6946c6fcf76c7315354e1af9

dnf install -y boost-devel gmp-devel gmp-c++

tmp="$(mktemp -d)"
curl -fsSL "https://github.com/xxsds/sdsl-lite/archive/${SDSL_COMMIT}.tar.gz" | tar xz -C "$tmp"
cp -r "$tmp/sdsl-lite-${SDSL_COMMIT}/include/sdsl" /usr/local/include/
cp "$tmp/sdsl-lite-${SDSL_COMMIT}/LICENSE" /usr/local/include/sdsl/LICENSE
rm -rf "$tmp"

# Fail loudly here rather than 300 lines into a template error.
test -f /usr/local/include/sdsl/bit_vectors.hpp
test -f /usr/include/gmpxx.h
test -d /usr/include/boost/serialization
