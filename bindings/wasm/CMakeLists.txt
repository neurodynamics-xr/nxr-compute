# ── nxr-compute-wasm — WebAssembly binding for browser apps ──────────
#
# Compiled via Emscripten using emcmake / emmake. Produces:
#   build_wasm/nxr_compute.js     — ES-module factory function
#   build_wasm/nxr_compute.wasm   — WebAssembly binary
#
# Usage from JS:
#   import createNxrComputeModule from './nxr_compute.js'
#   const nxr-compute = await createNxrComputeModule()
#   const ctx = new nxr-compute.ComputeContext(verts, faces)
#
# This target only builds when CMake is configured with the
# Emscripten toolchain (i.e. via emcmake). It links the nxr-compute
# static library — *not* cxf-io, since file I/O is a JS-side
# concern in the browser.

if(NOT EMSCRIPTEN)
    message(WARNING "nxr-compute-wasm: EMSCRIPTEN not defined — skipping target. "
                    "Build with: emcmake cmake -B build_wasm && emmake make -C build_wasm")
    return()
endif()

add_executable(nxr_compute_wasm src/nxr_compute_wasm.cpp)
set_target_properties(nxr_compute_wasm PROPERTIES
    OUTPUT_NAME nxr_compute
    SUFFIX ".js"        # Emscripten executable → .js + .wasm pair
)
target_link_libraries(nxr_compute_wasm PRIVATE nxr_compute)

# ── Emscripten link options ─────────────────────────────────
#
# --bind                       Enable Embind (C++ ↔ JS class binding)
# MODULARIZE                   Emit factory function rather than runtime IIFE
# EXPORT_NAME=createNxrComputeModule  Name of the factory
# EXPORT_ES6                   Emit ES module (`export default`)
# ALLOW_MEMORY_GROWTH          Linear memory grows on demand (no fixed cap)
# INITIAL_MEMORY=64MB          Comfortable starting heap
# MAXIMUM_MEMORY=2GB           Browser-side cap
# ENVIRONMENT=web,node,worker  Run in any of these
# NO_EXIT_RUNTIME=1            Keep the runtime alive after main() (so the
#                              ComputeContext doesn't get torn down)
# DISABLE_EXCEPTION_CATCHING=0 Allow C++ exceptions (nxr-compute throws on bad input)
# EXPORTED_RUNTIME_METHODS     Heap typed-array views the JS wrapper uses
target_link_options(nxr_compute_wasm PRIVATE
    "--bind"
    "-sMODULARIZE=1"
    "-sEXPORT_NAME=createNxrComputeModule"
    "-sEXPORT_ES6=1"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sINITIAL_MEMORY=67108864"
    "-sMAXIMUM_MEMORY=2147483648"
    "-sENVIRONMENT=web,node,worker"
    "-sNO_EXIT_RUNTIME=1"
    "-sDISABLE_EXCEPTION_CATCHING=0"
    "-sEXPORTED_RUNTIME_METHODS=['HEAPF32','HEAPF64','HEAP32','HEAPU32']"
)

# Release build: -O3 + LTO. Adds a few minutes to link but produces a
# noticeably smaller / faster .wasm. Override by passing -DCMAKE_BUILD_TYPE
# at configure time.
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
    target_compile_options(nxr_compute_wasm PRIVATE -O3)
    target_link_options(nxr_compute_wasm    PRIVATE -O3 -flto)
endif()

# Place outputs at build_wasm/ root for predictable paths.
set_target_properties(nxr_compute_wasm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
