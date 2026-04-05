cmake_minimum_required(VERSION 3.20)
project(cortical_compute LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── geometry-central (submodule) ──────────────────────────────
# This also downloads/finds Eigen3 automatically
add_subdirectory(deps/geometry-central)

# ── Spectra (header-only) ────────────────────────────────────
# Spectra needs Eigen3, but geometry-central already provides it.
# Use FetchContent to get Spectra source, but skip its CMake
# (which tries find_package(Eigen3) independently).
# Instead, just add Spectra's include directory.
include(FetchContent)
FetchContent_Declare(
  spectra
  GIT_REPOSITORY https://github.com/yixuan/spectra.git
  GIT_TAG        v1.0.1
  GIT_SHALLOW    TRUE
)
FetchContent_Populate(spectra)
# Don't call FetchContent_MakeAvailable — skip Spectra's CMakeLists.txt
# Just use its headers directly

# ── Compute library (static, no Node.js dependency) ──────────
add_library(cortical_compute STATIC
  src/mesh_operators.cpp
  src/eigensolver.cpp
  src/normalize.cpp
  src/poisson_solver.cpp
  src/geodesic.cpp
  src/hodge.cpp
)

target_include_directories(cortical_compute PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${spectra_SOURCE_DIR}/include
)

target_link_libraries(cortical_compute PUBLIC
  geometry-central
)

# ── Test executable ───────────────────────────────────────────
add_executable(test_eigen test/test_eigen.cpp)
target_link_libraries(test_eigen PRIVATE cortical_compute)
