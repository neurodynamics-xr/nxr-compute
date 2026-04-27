cmake_minimum_required(VERSION 3.20)
project(cxf_workspace LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Shared third-party dependencies ──────────────────────────
# These are visible to every target (cxf library, addon, future
# cli/mex/wasm wrappers). Per CLAUDE.md §10, the dependency policy
# is locked: Eigen + Spectra + geometry-central, nothing else.

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
# around the cxf static library.

add_subdirectory(cxf)        # PURE compute — Eigen+Spectra+gc, builds for any target

if(EMSCRIPTEN)
    # WASM build (via emcmake): only cxf + wasm wrapper. The other
    # bindings either don't apply (mex needs MATLAB, addon needs N-API)
    # or aren't useful in the browser (cli needs filesystem).
    add_subdirectory(wasm)
else()
    # Native build (cmake-js / standard CMake): everything else.
    add_subdirectory(cxf-io)     # File-format adapters (FreeSurfer, Zarr)
    add_subdirectory(addon)      # N-API addon — links cxf + cxf-io
    add_subdirectory(cli)        # Standalone executable — links cxf + cxf-io
    add_subdirectory(mex)        # MATLAB MEX — links cxf only
endif()
