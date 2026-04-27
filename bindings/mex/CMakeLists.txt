# ── cxf-mex — MATLAB MEX wrapper around the cxf library ──────
#
# Single mexFunction with command dispatch. Built only when MATLAB
# is detected at configure time; absent MATLAB the rest of the
# native workspace (cxf, addon, cli) builds unaffected.
#
# Usage from MATLAB:
#   ops = cxf('assembleMeshOperators', V, F);
#   eig = cxf('solveEigenmodes', ops.stiffness, ops.mass, 300);

find_package(Matlab COMPONENTS MX_LIBRARY MAIN_PROGRAM)

if(NOT Matlab_FOUND)
    message(STATUS "cxf-mex: MATLAB not found, skipping MEX target.")
    return()
endif()

message(STATUS "cxf-mex: building against MATLAB at ${Matlab_ROOT_DIR}")

matlab_add_mex(
    NAME cxf_mex                # CMake target name (file is cxf.mexw64)
    SRC src/cxf_mex.cpp
    LINK_TO cxf
    OUTPUT_NAME cxf
    R2018a                      # use the new mxArray API (interleaved complex etc.)
)

# matlab_add_mex doesn't propagate target_compile_options for C++17,
# so set it explicitly. cxf already requires C++17.
target_compile_features(cxf_mex PRIVATE cxx_std_17)

# The MEX wrapper depends on cxf's public headers but isn't itself a
# consumer that needs Spectra paths exposed — those propagate through
# cxf's PUBLIC interface link.
