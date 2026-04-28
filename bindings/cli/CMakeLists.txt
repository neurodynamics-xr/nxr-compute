# ── nxr_compute_cli — standalone command-line tool ───────────────────
#
# Wraps the nxr-compute library as a single executable. Currently exposes
# one subcommand: `nxr-compute precompute` (offline operator + eigenmode
# precompute, writes a Zarr v2 store the renderer can load).
#
# Linked against nxr-compute only — no Node.js, no MATLAB, no Emscripten.
# This is intentionally the smallest binding shell, validating the
# multi-target architecture before MEX/WASM follow.

# CMake target name `nxr_compute_cli` (the library target is already `nxr-compute`).
# OUTPUT_NAME makes the produced binary `nxr-compute` so users invoke it as
# `nxr-compute precompute …` regardless of the internal target name.
add_executable(nxr_compute_cli src/main.cpp)
set_target_properties(nxr_compute_cli PROPERTIES OUTPUT_NAME nxr_compute)
target_link_libraries(nxr_compute_cli PRIVATE nxr_compute cxf-io)
