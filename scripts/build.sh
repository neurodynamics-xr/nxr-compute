#!/usr/bin/env bash
# Build the native C++ addon (bioctree_addon.node)
#
# Prerequisites:
#   - Visual Studio 2026 (or later) with C++ workload
#   - node_modules installed (npm install)
#
# Usage:
#   bash scripts/build-native.sh [Release|Debug]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
NATIVE_DIR="$PROJECT_ROOT/native"
CONFIG="${1:-Release}"

# Add VS-bundled CMake to PATH
CMAKE_BIN="/c/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin"
if [ -d "$CMAKE_BIN" ]; then
    export PATH="$CMAKE_BIN:$PATH"
fi

echo "Building native addon (config: $CONFIG)..."
cd "$NATIVE_DIR"

"$PROJECT_ROOT/node_modules/.bin/cmake-js.cmd" build \
    --directory . \
    --out build_node \
    --config "$CONFIG"

# Copy the .node file to project root for easy require()
ADDON_PATH="$NATIVE_DIR/build_node/$CONFIG/bioctree_addon.node"
if [ -f "$ADDON_PATH" ]; then
    cp "$ADDON_PATH" "$PROJECT_ROOT/bioctree_addon.node"
    echo "✓ Copied bioctree_addon.node to project root"
    echo "  Size: $(du -h "$PROJECT_ROOT/bioctree_addon.node" | cut -f1)"
else
    echo "✗ bioctree_addon.node not found at $ADDON_PATH"
    exit 1
fi
