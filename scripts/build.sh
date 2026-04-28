#!/usr/bin/env bash
# Build all native targets in the nxr-compute workspace.
#
# Outputs (under build/):
#   nxr_compute.lib / .a      — pure compute static library
#   bindings/node/Release/nxr_compute_addon.node   — N-API addon (if cmake-js available)
#   bindings/cli/Release/nxr_compute.exe           — CLI smoke
#   bindings/mex/Release/nxr_compute.mexw64        — MATLAB MEX (if MATLAB found)
#   test_*.exe                — 7 native test executables
#
# Prerequisites:
#   - CMake 3.20+ (Visual Studio bundles one on Windows)
#   - For the addon: `npm install` to get cmake-js + node-addon-api
#   - For MEX:        MATLAB R2018a or later, found by find_package(Matlab)
#   - For WASM:       use scripts/build-wasm.sh instead — separate toolchain
#
# Usage (from repo root):
#   bash scripts/build.sh [Release|Debug]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
CONFIG="${1:-Release}"

# Add VS-bundled CMake to PATH if not already there
CMAKE_BIN="/c/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin"
if [ -d "$CMAKE_BIN" ]; then
    export PATH="$CMAKE_BIN:$PATH"
fi

cd "$PROJECT_ROOT"

echo "Building nxr-compute (config: $CONFIG)..."

# Prefer cmake-js when available — it propagates the N-API include
# paths needed to build the addon. Fall back to plain cmake (skips
# the addon target via the if(DEFINED CMAKE_JS_INC) guard) when
# cmake-js is absent.
CMAKE_JS=""
if command -v cmake-js >/dev/null 2>&1; then
    CMAKE_JS="cmake-js"
elif [ -x "$PROJECT_ROOT/node_modules/.bin/cmake-js.cmd" ]; then
    CMAKE_JS="$PROJECT_ROOT/node_modules/.bin/cmake-js.cmd"
elif [ -x "$PROJECT_ROOT/node_modules/.bin/cmake-js" ]; then
    CMAKE_JS="$PROJECT_ROOT/node_modules/.bin/cmake-js"
fi

if [ -n "$CMAKE_JS" ]; then
    echo "  using cmake-js: $CMAKE_JS"
    "$CMAKE_JS" build --directory . --out build --config "$CONFIG"
else
    echo "  cmake-js not found — building without N-API addon."
    echo "  (Run 'npm install' first to enable the addon target.)"
    cmake -B build -S . -DCMAKE_BUILD_TYPE="$CONFIG"
    cmake --build build --config "$CONFIG"
fi

# Copy the addon to the project root for the smoke scripts to pick up.
ADDON_SRC=$(find "$PROJECT_ROOT/build" -name 'nxr_compute_addon.node' 2>/dev/null | head -1)
if [ -n "$ADDON_SRC" ]; then
    cp "$ADDON_SRC" "$PROJECT_ROOT/nxr_compute_addon.node"
    echo "✓ Copied nxr_compute_addon.node to project root"
fi

echo ""
echo "Outputs:"
find "$PROJECT_ROOT/build" \
    \( -name '*.node' \
    -o -name 'nxr_compute.exe' \
    -o -name 'nxr_compute.mexw64' \
    -o -name 'test_*.exe' \) 2>/dev/null \
    | sed 's|^|  |'
