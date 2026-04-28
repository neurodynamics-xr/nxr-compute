cmake_minimum_required(VERSION 3.20)
project(nxr_compute_workspace LANGUAGES CXX)

# C++17 is the workspace default — geometry-central + Eigen 3.4
# under Emscripten Clang trip a template-deduction bug at C++20
# (see boundary_first_flattening.cpp). nxr_compute itself bumps
# its own target to C++17 via target_compile_features in
# nxr-compute/CMakeLists.txt so its consumers (addon, wasm, mex,
# cli, tests) inherit the requirement without forcing
# geometry-central up.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Shared third-party dependencies ──────────────────────────
# These are visible to every target (nxr_compute library, addon,
# future cli/mex/wasm wrappers). Per CLAUDE.md §10, the dependency
# policy is locked: Eigen + Spectra + geometry-central, nothing else.

add_subdirectory(deps/geometry-central)

include(FetchContent)
FetchContent_Declare(
  spectra
  GIT_REPOSITORY https://github.com/yixuan/spectra.git
  GIT_TAG        v1.0.1
  GIT_SHALLOW    TRUE
)
FetchContent_Populate(spectra)

# ── Targets ──────────────────────────────────────────────────
# Each is its own CMakeLists.txt, building one binding shell
# around the nxr_compute static library.

add_subdirectory(nxr-compute) # PURE compute — Eigen+Spectra+gc, builds for any target

if(EMSCRIPTEN)
    # WASM build (via emcmake): only nxr_compute + wasm wrapper. The
    # other bindings either don't apply (mex needs MATLAB, addon needs
    # N-API) or aren't useful in the browser (cli needs filesystem).
    add_subdirectory(wasm)
else()
    # Native build (cmake-js / standard CMake): everything else.
    add_subdirectory(cxf-io)     # File-format adapters (FreeSurfer, Zarr)
    add_subdirectory(addon)      # N-API addon — links nxr_compute + cxf-io
    add_subdirectory(cli)        # Standalone executable — links nxr_compute + cxf-io
    add_subdirectory(mex)        # MATLAB MEX — links nxr_compute only
endif()
