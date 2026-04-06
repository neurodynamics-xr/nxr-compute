cmake_minimum_required(VERSION 3.20)
project(cortical_compute LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── geometry-central (submodule) ──────────────────────────────
add_subdirectory(deps/geometry-central)

# ── Spectra (header-only) ────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
  spectra
  GIT_REPOSITORY https://github.com/yixuan/spectra.git
  GIT_TAG        v1.0.1
  GIT_SHALLOW    TRUE
)
FetchContent_Populate(spectra)

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

# ── Test executable (standalone, no Node.js) ─────────────────
add_executable(test_eigen test/test_eigen.cpp)
target_link_libraries(test_eigen PRIVATE cortical_compute)

# ── N-API Addon (bioctree_addon.node) ────────────────────────
# Build only when invoked through cmake-js (which sets CMAKE_JS_INC)
if(DEFINED CMAKE_JS_INC)
    add_library(bioctree_addon SHARED
        src/addon.cpp
        ${CMAKE_JS_SRC}
    )
    set_target_properties(bioctree_addon PROPERTIES
        PREFIX ""
        SUFFIX ".node"
    )
    target_include_directories(bioctree_addon PRIVATE
        ${CMAKE_JS_INC}
        ${CMAKE_CURRENT_SOURCE_DIR}/../node_modules/node-addon-api
    )
    target_link_libraries(bioctree_addon PRIVATE
        cortical_compute
        ${CMAKE_JS_LIB}
    )
    # Disable C++ exceptions in N-API (use NAPI C exceptions)
    target_compile_definitions(bioctree_addon PRIVATE
        NAPI_VERSION=9
        NAPI_CPP_EXCEPTIONS
    )
    if(MSVC)
        target_compile_options(bioctree_addon PRIVATE /EHsc)
    endif()
endif()
