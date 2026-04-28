cmake_minimum_required(VERSION 3.20)
project(nxr_compute LANGUAGES CXX)

# ── Workspace defaults ───────────────────────────────────────
# C++17 floor. geometry-central + Eigen 3.4 under Emscripten Clang
# trip a template-deduction bug at C++20 (visible in
# boundary_first_flattening.cpp). Until Eigen is upgraded or the
# trip site patched, the entire workspace stays at C++17.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Dependencies ─────────────────────────────────────────────
# Per CLAUDE.md, the dependency policy is locked: Eigen + Spectra
# + geometry-central, nothing else. Both are header-only or
# vendored, so no system packages are required.

add_subdirectory(deps/geometry-central)

include(FetchContent)
FetchContent_Declare(
  spectra
  GIT_REPOSITORY https://github.com/yixuan/spectra.git
  GIT_TAG        v1.0.1
  GIT_SHALLOW    TRUE
)
FetchContent_Populate(spectra)

# ── nxr_compute static library ───────────────────────────────
# The pure compute kernel. Depends only on Eigen (transitively
# via geometry-central), Spectra (header-only), and
# geometry-central (vendored submodule). All consumer targets
# (the N-API addon, MATLAB MEX, WASM, CLI) link against this
# library and add nothing more.
#
# CMake target: nxr_compute
# Headers:      include/nxr/{compute,errors,cancellation,progress}.h
# Namespace:    nxr::compute

add_library(nxr_compute STATIC
  src/mesh_operators.cpp
  src/eigensolver.cpp
  src/normalize.cpp
  src/cholesky_cache.cpp
  src/poisson_solver.cpp
  src/geodesic.cpp
  src/hodge.cpp
  src/field_generators.cpp
  src/curvatures.cpp
  src/normals.cpp
  src/vector_field.cpp
  src/isolines.cpp
  src/direction_field.cpp
  src/face_frames.cpp
  src/geodesic_path.cpp
  src/parametrization.cpp
)

target_include_directories(nxr_compute PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${spectra_SOURCE_DIR}/include
)

target_compile_features(nxr_compute PUBLIC cxx_std_17)

target_link_libraries(nxr_compute PUBLIC
  geometry-central
)

# AVX2 intentionally NOT enabled — /arch:AVX2 on nxr_compute alone
# causes an Eigen alignment ABI mismatch with geometry-central.
# Re-enable only by either propagating the flag through
# geometry-central or defining EIGEN_MAX_ALIGN_BYTES=16 globally.

# ── Consumer-facing options ─────────────────────────────────
# Defaults are set for the standalone build (everything ON).
# Downstream consumers that just want the static library — e.g.
# cortical-flow, brainstorm, or any host app embedding nxr-compute
# as a submodule — should set both OFF before add_subdirectory()
# to avoid target name collisions with their own bindings:
#
#   set(NXR_COMPUTE_BUILD_BINDINGS OFF CACHE BOOL "" FORCE)
#   set(NXR_COMPUTE_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
#   add_subdirectory(vendor/nxr-compute)
#
# Standalone users (running the repo's own scripts/build.sh) get
# all four bindings + 7 tests with no extra configuration.

option(NXR_COMPUTE_BUILD_BINDINGS
    "Build N-API / WASM / MATLAB MEX / CLI binding shells" ON)
option(NXR_COMPUTE_BUILD_TESTS
    "Build standalone test executables (test_eigen, etc.)"  ON)

# ── Bindings ─────────────────────────────────────────────────
# Each binding is its own CMakeLists.txt under bindings/, building
# one shell around the nxr_compute static library. Builds are
# conditional on the toolchain: WASM under Emscripten, native
# bindings otherwise.

if(NXR_COMPUTE_BUILD_BINDINGS)
    if(EMSCRIPTEN)
        add_subdirectory(bindings/wasm)
    else()
        add_subdirectory(bindings/node)  # N-API for Node.js / Electron
        add_subdirectory(bindings/cli)   # Standalone CLI smoke
        add_subdirectory(bindings/mex)   # MATLAB MEX
    endif()
endif()

# ── Standalone tests (no Node.js) ────────────────────────────
# Skipped under Emscripten — these are native-only smoke tests;
# the WASM build has its own JS-side smoke (scripts/_smoke-wasm.mjs).

if(NOT EMSCRIPTEN AND NXR_COMPUTE_BUILD_TESTS)
    enable_testing()

    add_executable(test_eigen test/test_eigen.cpp)
    target_link_libraries(test_eigen PRIVATE nxr_compute)
    add_test(NAME test_eigen COMMAND test_eigen)

    add_executable(test_cholesky_cache test/test_cholesky_cache.cpp)
    target_link_libraries(test_cholesky_cache PRIVATE nxr_compute)
    add_test(NAME test_cholesky_cache COMMAND test_cholesky_cache)

    add_executable(test_field_generators test/test_field_generators.cpp)
    target_link_libraries(test_field_generators PRIVATE nxr_compute)
    add_test(NAME test_field_generators COMMAND test_field_generators)

    add_executable(test_visualization_primitives test/test_visualization_primitives.cpp)
    target_link_libraries(test_visualization_primitives PRIVATE nxr_compute)
    add_test(NAME test_visualization_primitives COMMAND test_visualization_primitives)

    add_executable(test_cancellation test/test_cancellation.cpp)
    target_link_libraries(test_cancellation PRIVATE nxr_compute)
    add_test(NAME test_cancellation COMMAND test_cancellation)

    add_executable(test_progress test/test_progress.cpp)
    target_link_libraries(test_progress PRIVATE nxr_compute)
    add_test(NAME test_progress COMMAND test_progress)

    add_executable(test_graph_agnostic test/test_graph_agnostic.cpp)
    target_link_libraries(test_graph_agnostic PRIVATE nxr_compute)
    add_test(NAME test_graph_agnostic COMMAND test_graph_agnostic)
endif()
