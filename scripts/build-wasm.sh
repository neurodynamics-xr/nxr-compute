#!/usr/bin/env bash
# Build the nxr-compute WASM binding via Emscripten.
#
# Outputs:
#   build_wasm/nxr_compute.js   — ES-module factory function (createNxrComputeModule)
#   build_wasm/nxr_compute.wasm — WebAssembly binary
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
BUILD_DIR="$PROJECT_ROOT/build_wasm"
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

echo "Building nxr-compute WASM (config: $CONFIG)..."
echo "  Project: $PROJECT_ROOT"
echo "  Build:   $BUILD_DIR"
"$EMCC_BIN" --version | head -1

cd "$PROJECT_ROOT"

# Configure (rerun if BUILD_DIR doesn't exist or the user passed --reconfigure)
if [ ! -d "$BUILD_DIR" ] || [ "$2" = "--reconfigure" ]; then
    rm -rf "$BUILD_DIR"
    "$EMCMAKE_BIN" cmake -B "$BUILD_DIR" -S . \
        -DCMAKE_BUILD_TYPE="$CONFIG"
fi

# Build
"$EMMAKE_BIN" cmake --build "$BUILD_DIR" --config "$CONFIG"

# Sanity check the outputs
if [ -f "$BUILD_DIR/nxr_compute.js" ] && [ -f "$BUILD_DIR/nxr_compute.wasm" ]; then
    echo ""
    echo "✓ Built nxr_compute.js   ($(du -h "$BUILD_DIR/nxr_compute.js"   | cut -f1))"
    echo "✓ Built nxr_compute.wasm ($(du -h "$BUILD_DIR/nxr_compute.wasm" | cut -f1))"
else
    echo "✗ nxr_compute.js / nxr_compute.wasm not produced"
    exit 1
fi

# Refresh the committed prebuilt artifacts. dist/wasm/ is what the
# shim (bindings/wasm/js/index.mjs) and downstream git-installed
# consumers (e.g. nxr-design-system) actually load.
DIST_DIR="$PROJECT_ROOT/dist/wasm"
mkdir -p "$DIST_DIR"
cp "$BUILD_DIR/nxr_compute.js"   "$DIST_DIR/nxr_compute.js"
cp "$BUILD_DIR/nxr_compute.wasm" "$DIST_DIR/nxr_compute.wasm"
echo "✓ Refreshed $DIST_DIR/{nxr_compute.js,nxr_compute.wasm}"
