cmake_minimum_required(VERSION 3.20)
project(bioctree_addon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Dependencies (uncomment when ready to build) ---

# geometry-central (git submodule in deps/)
# add_subdirectory(deps/geometry-central)

# Eigen 3.4+ (via vcpkg or system)
# find_package(Eigen3 3.4 REQUIRED)

# SuiteSparse / CHOLMOD (via vcpkg)
# find_package(SuiteSparse REQUIRED)

# Spectra 1.0+ (header-only, in deps/spectra/include)
# include_directories(deps/spectra/include)

# Node.js N-API (via cmake-js)
# include_directories(${CMAKE_JS_INC})

# --- Sources ---
# set(ADDON_SOURCES
#   src/addon.cpp
#   src/mesh_quantities.cpp
#   src/eigensolver.cpp
#   src/normalize.cpp
#   src/flow_solver.cpp
#   src/dec_operators.cpp
#   src/helmholtz.cpp
#   src/synthetic_generators.cpp
# )

# add_library(bioctree_addon SHARED ${ADDON_SOURCES})
# set_target_properties(bioctree_addon PROPERTIES PREFIX "" SUFFIX ".node")
# target_include_directories(bioctree_addon PRIVATE include ${CMAKE_JS_INC})
# target_link_libraries(bioctree_addon
#   geometry-central
#   Eigen3::Eigen
#   ${SUITESPARSE_LIBRARIES}
#   ${CMAKE_JS_LIB}
# )
