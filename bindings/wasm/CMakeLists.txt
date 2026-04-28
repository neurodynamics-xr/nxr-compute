# ── cxf-wasm — WebAssembly binding for browser apps ──────────
#
# Compiled via Emscripten using emcmake / emmake. Produces:
#   build_wasm/cxf.js     — ES-module factory function
#   build_wasm/cxf.wasm   — WebAssembly binary
#
# Usage from JS:
#   import createCxfModule from './cxf.js'
#   const cxf = await createCxfModule()
#   const ctx = new cxf.ComputeContext(verts, faces)
#
# This target only builds when CMake is configured with the
# Emscripten toolchain (i.e. via emcmake). It links the cxf
# static library — *not* cxf-io, since file I/O is a JS-side
# concern in the browser.

if(NOT EMSCRIPTEN)
    message(WARNING "cxf-wasm: EMSCRIPTEN not defined — skipping target. "
                    "Build with: emcmake cmake -B build_wasm && emmake make -C build_wasm")
    return()
endif()

add_executable(cxf_wasm src/cxf_wasm.cpp)
set_target_properties(cxf_wasm PROPERTIES
    OUTPUT_NAME cxf
    SUFFIX ".js"        # Emscripten executable → .js + .wasm pair
)
target_link_libraries(cxf_wasm PRIVATE nxr_compute)

# ── Emscripten link options ─────────────────────────────────
#
# --bind                       Enable Embind (C++ ↔ JS class binding)
# MODULARIZE                   Emit factory function rather than runtime IIFE
# EXPORT_NAME=createCxfModule  Name of the factory
# EXPORT_ES6                   Emit ES module (`export default`)
# ALLOW_MEMORY_GROWTH          Linear memory grows on demand (no fixed cap)
# INITIAL_MEMORY=64MB          Comfortable starting heap
# MAXIMUM_MEMORY=2GB           Browser-side cap
# ENVIRONMENT=web,node,worker  Run in any of these
# NO_EXIT_RUNTIME=1            Keep the runtime alive after main() (so the
#                              ComputeContext doesn't get torn down)
# DISABLE_EXCEPTION_CATCHING=0 Allow C++ exceptions (cxf throws on bad input)
# EXPORTED_RUNTIME_METHODS     Heap typed-array views the JS wrapper uses
target_link_options(cxf_wasm PRIVATE
    "--bind"
    "-sMODULARIZE=1"
    "-sEXPORT_NAME=createCxfModule"
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
    target_compile_options(cxf_wasm PRIVATE -O3)
    target_link_options(cxf_wasm    PRIVATE -O3 -flto)
endif()

# Place outputs at build_wasm/ root for predictable paths.
set_target_properties(cxf_wasm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
