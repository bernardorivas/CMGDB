cmake_minimum_required(VERSION 3.15...3.26)

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

# CMP0144 (CMake 3.27) lets find_package honor an upper-case BOOST_ROOT,
# which the CI workflows set. scikit-build-core accepts any CMake from the
# minimum above, and an older one stops at a policy it does not know.
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

# C++17, which the vendored sdsl-lite needs. C++20 removes members of
# std::allocator (construct, destroy, rebind) that the ublas and
# property_tree headers of older Boost releases still use, so Ubuntu
# 22.04's Boost 1.74 and EL8's 1.66 do not compile under it.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Note on optimization flags: the Release build already compiles with -O3.
# LTO was tried and measured to have no effect -- the project builds as a
# single translation unit (CMGDB.cpp includes the header-only codebase), so
# there are no cross-TU boundaries for LTO to optimize.

# Visual Studio C++ does not support keywords such
# as "and", "not", etc. Settting the /permissive-
# flag for the compiler makes it support them.
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive-")
  # Python extension modules must use the dynamic CRT (/MD, CMake's MSVC
  # default): CPython itself is /MD, and mixing C runtimes across the
  # extension boundary is unsafe. Boost still links statically into the
  # module -- via its dynamic-runtime static libraries, so
  # Boost_USE_STATIC_RUNTIME must stay off to select them.
  set(Boost_USE_STATIC_LIBS ON)
endif()

if(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)

# With the settings above pybind11 finds Python
# find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Boost installs a BoostConfig.cmake from 1.70 on. Older releases, such as
# EL8's 1.66, are found by CMake's FindBoost module instead, which also
# honors an upper-case BOOST_ROOT. Policy CMP0167 (CMake 3.30) removes that
# module; setting it to OLD keeps the module available.
find_package(Boost CONFIG QUIET COMPONENTS chrono thread serialization)
if(NOT Boost_FOUND)
  if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
  endif()
  find_package(Boost REQUIRED COMPONENTS chrono thread serialization)
endif()

if(Boost_FOUND)
  message(STATUS "Boost version: ${Boost_VERSION}")
  message("Boost_ROOT is ${Boost_ROOT}")
  message("Boost_INCLUDE_DIRS is ${Boost_INCLUDE_DIRS}")
  message("Boost_LIBRARY_DIRS is ${Boost_LIBRARY_DIRS}")
  message("Boost_LIBRARIES is ${Boost_LIBRARIES}")
endif()

# Set Boost include and lib directories
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

if(APPLE)
  # Homebrew directories for Apple silicon
  # Homebrew include and library dirs (for GMP)
  include_directories(/opt/homebrew/include)
  link_directories(/opt/homebrew/lib)
endif()

# pyproject.toml passes USER_PROJECT_PATH in cmake.args, but a cmake.args
# given at build time (-C cmake.args=... or SKBUILD_CMAKE_ARGS) replaces
# that list instead of extending it, so default it here.
if(NOT DEFINED USER_PROJECT_PATH)
  set(USER_PROJECT_PATH src/CMGDB/_cmgdb)
endif()
include_directories(${CMAKE_SOURCE_DIR}/${USER_PROJECT_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/${USER_PROJECT_PATH}/include/database)
# include_directories(${CMAKE_SOURCE_DIR}/src/CMGDB/_cmgdb/include)
# include_directories(${CMAKE_SOURCE_DIR}/src/CMGDB/_cmgdb/include/database)

pybind11_add_module(_cmgdb MODULE ${USER_PROJECT_PATH}/CMGDB.cpp)
# pybind11_add_module(_cmgdb MODULE src/CMGDB/_cmgdb/CMGDB.cpp)

# Boost's include directory (and the Homebrew one above) may hold another
# pybind11 (Homebrew installs one in /opt/homebrew/include). Search the
# headers of the pybind11 package CMake found first, so that headers and
# CMake helpers always come from the same pybind11 release.
target_include_directories(_cmgdb SYSTEM BEFORE PRIVATE ${pybind11_INCLUDE_DIR})

# Vendored header-only sdsl-lite v3 (see src/CMGDB/_cmgdb/third_party/sdsl-lite/),
# compiled when CMGDB_USE_SUCCINCT (the optional succinct grid; the phase
# grid is PointerGrid either way) or CMGDB_USE_ATLAS is defined. This fork
# always defines CMGDB_USE_ATLAS (below), so sdsl is always compiled in.
# BEFORE, so that an sdsl v2 installation under a system prefix (e.g.
# /usr/local or Homebrew) cannot shadow these headers.
target_include_directories(_cmgdb BEFORE PRIVATE
  ${CMAKE_SOURCE_DIR}/${USER_PROJECT_PATH}/third_party/sdsl-lite/include)

# This fork binds AtlasModel (CMGDB.AtlasModel, ComputeMorseGraph on an
# AtlasModel), so the Atlas grid is always compiled: Atlas.h uses the
# sdsl-backed RankSelect from the vendored headers above, and
# Compute_Morse_Graph joins Atlas grids through joinImpl<Atlas>, which
# join.h defines only under CMGDB_USE_ATLAS. Without it every Atlas Morse
# graph computation throws "joinImpl specialization not written". The
# sdsl-lite BSD-3-Clause notice therefore ships with every wheel
# (license-files in pyproject.toml).
target_compile_definitions(_cmgdb PRIVATE CMGDB_USE_ATLAS)

target_link_libraries(_cmgdb PRIVATE Boost::chrono Boost::thread Boost::serialization)

install(TARGETS _cmgdb DESTINATION ${SKBUILD_PROJECT_NAME})
