#!/usr/bin/env bash
# Build the cxf WASM binding via Emscripten.
#
# Outputs:
#   native/build_wasm/cxf.js   — ES-module factory function (createCxfModule)
#   native/build_wasm/cxf.wasm — WebAssembly binary
#
# Prerequisites:
#   - emsdk installed and activated (see https://emscripten.org/docs/getting_started/downloads.html)
#   - This script auto-sources /c/emsdk/emsdk_env.sh if EMSDK is unset
#
# Usage (from repo root):
#   bash scripts/build-wasm.sh [Release|Debug]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
NATIVE_DIR="$PROJECT_ROOT/native"
BUILD_DIR="$NATIVE_DIR/build_wasm"
CONFIG="${1:-Release}"

# Resolve emsdk binaries. On Windows the wrappers are .bat files, which
# bash's `command -v` doesn't find via PATH lookup; we call them by
# absolute path to side-step the issue.
if [ -z "$EMSDK" ]; then
    if [ -f /c/emsdk/emsdk_env.sh ]; then
        EMSDK="/c/emsdk"
    else
        echo "ERROR: EMSDK env var unset and /c/emsdk not found."
        echo "       Install: https://emscripten.org/docs/getting_started/downloads.html"
        exit 1
    fi
fi

EMCC_BIN="$EMSDK/upstream/emscripten/emcc.bat"
EMCMAKE_BIN="$EMSDK/upstream/emscripten/emcmake.bat"
EMMAKE_BIN="$EMSDK/upstream/emscripten/emmake.bat"

if [ ! -f "$EMCC_BIN" ]; then
    echo "ERROR: $EMCC_BIN not found. Did you run \`./emsdk install latest\`?"
    exit 1
fi

# emcmake invokes cmake + needs a Unix-style generator (ninja or
# mingw32-make), since Emscripten can't drive MSBuild directly.
# VS bundles both cmake and ninja; add them to PATH.
VS_CMAKE_DIR="/c/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin"
VS_NINJA_DIR="/c/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja"
if [ -d "$VS_CMAKE_DIR" ]; then export PATH="$VS_CMAKE_DIR:$PATH"; fi
if [ -d "$VS_NINJA_DIR" ]; then export PATH="$VS_NINJA_DIR:$PATH"; fi

echo "Building cxf WASM (config: $CONFIG)..."
echo "  Project: $PROJECT_ROOT"
echo "  Build:   $BUILD_DIR"
"$EMCC_BIN" --version | head -1

cd "$NATIVE_DIR"

# Configure (rerun if BUILD_DIR doesn't exist or the user passed --reconfigure)
if [ ! -d "$BUILD_DIR" ] || [ "$2" = "--reconfigure" ]; then
    rm -rf "$BUILD_DIR"
    "$EMCMAKE_BIN" cmake -B "$BUILD_DIR" -S . \
        -DCMAKE_BUILD_TYPE="$CONFIG"
fi

# Build
"$EMMAKE_BIN" cmake --build "$BUILD_DIR" --config "$CONFIG"

# Sanity check the outputs
if [ -f "$BUILD_DIR/cxf.js" ] && [ -f "$BUILD_DIR/cxf.wasm" ]; then
    echo ""
    echo "✓ Built cxf.js   ($(du -h "$BUILD_DIR/cxf.js"   | cut -f1))"
    echo "✓ Built cxf.wasm ($(du -h "$BUILD_DIR/cxf.wasm" | cut -f1))"
else
    echo "✗ cxf.js / cxf.wasm not produced"
    exit 1
fi
