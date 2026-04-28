# ── nxr_compute_cli — standalone command-line smoke ──────────
#
# A minimal executable shell around the nxr-compute library:
# `nxr_compute smoke` runs an icosahedron eigensolve and asserts
# the canonical spectrum, providing a fast install-time sanity
# check without dragging file I/O into the library.
#
# Linked against nxr_compute only — no Node.js, no MATLAB, no
# Emscripten. Real precompute pipelines (FreeSurfer + Zarr) live
# in the host application that needs them.

add_executable(nxr_compute_cli src/main.cpp)
set_target_properties(nxr_compute_cli PROPERTIES OUTPUT_NAME nxr_compute)
target_link_libraries(nxr_compute_cli PRIVATE nxr_compute)
