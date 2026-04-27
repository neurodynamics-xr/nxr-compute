# ── cxf-cli — standalone command-line tool ───────────────────
#
# Wraps the cxf library as a single executable. Currently exposes
# one subcommand: `cxf precompute` (offline operator + eigenmode
# precompute, writes a Zarr v2 store the renderer can load).
#
# Linked against cxf only — no Node.js, no MATLAB, no Emscripten.
# This is intentionally the smallest binding shell, validating the
# multi-target architecture before MEX/WASM follow.

# CMake target name `cxf-cli` (the library target is already `cxf`).
# OUTPUT_NAME makes the produced binary `cxf` so users invoke it as
# `cxf precompute …` regardless of the internal target name.
add_executable(cxf-cli src/main.cpp)
set_target_properties(cxf-cli PROPERTIES OUTPUT_NAME cxf)
target_link_libraries(cxf-cli PRIVATE cxf cxf-io)
