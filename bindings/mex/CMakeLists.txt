# ── nxr-compute-mex — MATLAB MEX wrapper around the nxr-compute library ──────
#
# Single mexFunction with command dispatch. Built only when MATLAB
# is detected at configure time; absent MATLAB the rest of the
# native workspace (nxr-compute, addon, cli) builds unaffected.
#
# Usage from MATLAB:
#   ops = nxr_compute('assembleMeshOperators', V, F);
#   eig = nxr_compute('solveEigenmodes', ops.stiffness, ops.mass, 300);

find_package(Matlab COMPONENTS MX_LIBRARY MAIN_PROGRAM)

if(NOT Matlab_FOUND)
    message(STATUS "nxr-compute-mex: MATLAB not found, skipping MEX target.")
    return()
endif()

message(STATUS "nxr-compute-mex: building against MATLAB at ${Matlab_ROOT_DIR}")

matlab_add_mex(
    NAME nxr_compute_mex                # CMake target name (file is nxr_compute.mexw64)
    SRC src/nxr_compute_mex.cpp
    LINK_TO nxr_compute
    OUTPUT_NAME nxr_compute
    R2018a                      # use the new mxArray API (interleaved complex etc.)
)

# matlab_add_mex doesn't always honour transitive compile features
# from LINK_TO, so set the standard explicitly.  nxr-compute is C++17 today
# (see nxr-compute/CMakeLists.txt for the Emscripten/Eigen reasoning).
target_compile_features(nxr_compute_mex PRIVATE cxx_std_17)

# libut.lib provides utIsInterruptPending() — the canonical Ctrl-C
# bridge inside MEX functions. Undocumented but stable across MATLAB
# releases for ~20 years (used by libigl-matlab, gptoolbox). Without
# it, long-running nxr-compute operations can't be interrupted from MATLAB.
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
    message(STATUS "nxr-compute-mex: libut at ${Matlab_UT_LIBRARY}")
    # Plain signature — matlab_add_mex internally uses plain
    # target_link_libraries(nxr_compute_mex …) without keywords, so all later
    # links to the same target must be plain too.
    target_link_libraries(nxr_compute_mex "${Matlab_UT_LIBRARY}")
else()
    message(WARNING
        "nxr-compute-mex: libut not found in ${_matlab_libut_dir} — "
        "Ctrl-C will not interrupt nxr-compute solvers from MATLAB. "
        "Install path may differ on your platform.")
endif()

# The MEX wrapper depends on nxr-compute's public headers but isn't itself a
# consumer that needs Spectra paths exposed — those propagate through
# nxr-compute's PUBLIC interface link.
