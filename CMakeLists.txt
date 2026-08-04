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

# SDSL v3 (xxsds) is header-only, so there is nothing to link. SDSL v2
# (simongog) ships a static libsdsl and must be linked when it is what the
# build environment provides.
find_library(SDSL_LIBRARY NAMES sdsl)
if(SDSL_LIBRARY)
  message(STATUS "SDSL: linking ${SDSL_LIBRARY} (v2)")
  target_link_libraries(_cmgdb PRIVATE ${SDSL_LIBRARY})
else()
  message(STATUS "SDSL: header-only (v3)")
endif()
