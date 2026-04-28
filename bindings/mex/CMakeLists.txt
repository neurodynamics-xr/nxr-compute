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

# matlab_add_mex doesn't always honour transitive compile features
# from LINK_TO, so set the standard explicitly.  cxf is C++17 today
# (see cxf/CMakeLists.txt for the Emscripten/Eigen reasoning).
target_compile_features(cxf_mex PRIVATE cxx_std_17)

# libut.lib provides utIsInterruptPending() — the canonical Ctrl-C
# bridge inside MEX functions. Undocumented but stable across MATLAB
# releases for ~20 years (used by libigl-matlab, gptoolbox). Without
# it, long-running cxf operations can't be interrupted from MATLAB.
# FindMatlab doesn't expose libut by default, so we resolve it manually.
if(WIN32)
    set(_matlab_libut_dir "${Matlab_ROOT_DIR}/extern/lib/win64/microsoft")
elseif(APPLE)
    set(_matlab_libut_dir "${Matlab_ROOT_DIR}/bin/maci64")
else()
    set(_matlab_libut_dir "${Matlab_ROOT_DIR}/bin/glnxa64")
endif()
find_library(Matlab_UT_LIBRARY
    NAMES libut ut
    PATHS "${_matlab_libut_dir}"
    NO_DEFAULT_PATH
)
if(Matlab_UT_LIBRARY)
    message(STATUS "cxf-mex: libut at ${Matlab_UT_LIBRARY}")
    # Plain signature — matlab_add_mex internally uses plain
    # target_link_libraries(cxf_mex …) without keywords, so all later
    # links to the same target must be plain too.
    target_link_libraries(cxf_mex "${Matlab_UT_LIBRARY}")
else()
    message(WARNING
        "cxf-mex: libut not found in ${_matlab_libut_dir} — "
        "Ctrl-C will not interrupt cxf solvers from MATLAB. "
        "Install path may differ on your platform.")
endif()

# The MEX wrapper depends on cxf's public headers but isn't itself a
# consumer that needs Spectra paths exposed — those propagate through
# cxf's PUBLIC interface link.
