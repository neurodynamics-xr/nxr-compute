# ── nxr-compute-addon — N-API binding for native desktop apps ────────
#
# Wraps the nxr-compute library for use from Node.js / Electron. This is
# the binding shell consumed by cortical-flow's Electron renderer
# via electron/src/services/nxr-compute-native.ts.
#
# Built only when invoked through cmake-js (which sets CMAKE_JS_INC).
# The nxr-compute static library it links against is shared with all other
# target wrappers (CLI, MEX, WASM) — only this file is addon-specific.

if(DEFINED CMAKE_JS_INC)
    add_library(nxr_compute_addon SHARED
        src/addon.cpp
        ${CMAKE_JS_SRC}
    )
    set_target_properties(nxr_compute_addon PROPERTIES
        PREFIX ""
        SUFFIX ".node"
    )
    target_include_directories(nxr_compute_addon PRIVATE
        ${CMAKE_JS_INC}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../node_modules/node-addon-api
    )
    target_link_libraries(nxr_compute_addon PRIVATE
        nxr_compute
        cxf-io
        ${CMAKE_JS_LIB}
    )
    # Disable C++ exceptions in N-API (use NAPI C++ exception bridge).
    target_compile_definitions(nxr_compute_addon PRIVATE
        NAPI_VERSION=9
        NAPI_CPP_EXCEPTIONS
    )
    if(MSVC)
        target_compile_options(nxr_compute_addon PRIVATE /EHsc)
    endif()
endif()
