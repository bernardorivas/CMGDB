cmake_minimum_required(VERSION 3.15...3.26)

project(CMGDB)

cmake_policy(SET CMP0054 NEW)
cmake_policy(SET CMP0127 NEW)

# SDSL v3 (xxsds, header-only) requires C++17; the sources themselves are
# C++11-clean and compile unchanged at this standard.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Visual Studio C++ does not support keywords such
# as "and", "not", etc. Settting the /permissive-
# flag for the compiler makes it support them.
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive-")
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++latest /permissive-")
  set(Boost_USE_STATIC_LIBS ON)
  set(Boost_USE_STATIC_RUNTIME ON)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
  # Set compiler flag for sdsl-lite
  add_definitions("/DMSVC_COMPILER")
endif()

if(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)

# set(CMAKE_TOOLCHAIN_FILE C:/vcpkg/scripts/buildsystems/vcpkg.cmake)

find_package(Boost COMPONENTS chrono thread serialization REQUIRED)

if(Boost_FOUND)
  message(STATUS "Boost version: ${Boost_VERSION}")
  message("BOOST_ROOT is ${BOOST_ROOT}")
  message("Boost_INCLUDE_DIRS is ${Boost_INCLUDE_DIRS}")
  message("Boost_LIBRARY_DIRS is ${Boost_LIBRARY_DIRS}")
  message("Boost_LIBRARIES is ${Boost_LIBRARIES}")
endif()

# Set Boost include and lib directories
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

if(MSVC)
  # Set GMP and SDSL include and link directories
  message("GMP_ROOT is $ENV{GMP_ROOT}")
  include_directories($ENV{GMP_ROOT}/include)
  link_directories($ENV{GMP_ROOT}/lib)
  message("SDSL_ROOT is $ENV{SDSL_ROOT}")
  include_directories($ENV{SDSL_ROOT}/include)
  link_directories($ENV{SDSL_ROOT}/lib/Release)
endif()

if(APPLE)
  # Homebrew directories for Apple silicon
  include_directories(/opt/homebrew/include)
  link_directories(/opt/homebrew/lib)
endif()

include_directories (
  ${USER_INCLUDE_PATH}/database
  ${USER_INCLUDE_PATH}
  /usr/local/include )

link_directories (
  ${USER_LIBRARY_PATH}
  /usr/local/lib )

pybind11_add_module(_cmgdb src/CMGDB/_cmgdb/CMGDB.cpp)

target_link_libraries(_cmgdb PRIVATE Boost::chrono Boost::thread Boost::serialization)

# SDSL v3 (xxsds) is header-only and BSD-3 licensed; v2 (simongog) is a GPLv3
# static library whose symbols would make any redistributed binary a
# GPL-combined work. v3 is therefore required, not merely preferred, and is
# fetched when the build environment does not already provide it.
# sdsl/cereal.hpp exists only in v3, which is what tells the two apart.
find_path(SDSL_INCLUDE_DIR sdsl/cereal.hpp)
if(NOT SDSL_INCLUDE_DIR)
  message(STATUS "SDSL v3 not found; fetching the pinned release")
  include(FetchContent)
  FetchContent_Declare(sdsl_lite
    URL https://github.com/xxsds/sdsl-lite/archive/e6c417391f55476c6946c6fcf76c7315354e1af9.tar.gz
    URL_HASH SHA256=5c4f5d5f9d7c4e46019711000eeaaea756d181389abe76e597931e6cb69ae785
    # No CMakeLists.txt lives here, which keeps MakeAvailable from configuring
    # SDSL's own test tree (it needs submodules we do not fetch).
    SOURCE_SUBDIR headers-only)
  FetchContent_MakeAvailable(sdsl_lite)
  set(SDSL_INCLUDE_DIR ${sdsl_lite_SOURCE_DIR}/include)
endif()
message(STATUS "SDSL v3 headers: ${SDSL_INCLUDE_DIR}")
# BEFORE, so that a v2 installation under a system prefix (Homebrew, say)
# cannot shadow these headers.
target_include_directories(_cmgdb BEFORE PRIVATE ${SDSL_INCLUDE_DIR})
