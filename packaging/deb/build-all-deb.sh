#!/bin/bash
set -e

# Master build script for all DEB packages
# Usage: ./build-all-deb.sh [VERSION]
# Configuration: Uses build-config.json in packaging/ directory

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load configuration
if [ -f "$PACKAGING_DIR/config-loader.sh" ]; then
    source "$PACKAGING_DIR/config-loader.sh"
    print_config
    VERSION="${1:-${VERSION:-${PACKAGING_VERSION:-1.0.0.beta}}}"
else
    echo "Warning: config-loader.sh not found, using defaults"
    VERSION="${1:-${VERSION:-1.0.0.beta}}"
fi

echo "=========================================="
echo "Building all DEB packages version $VERSION"
echo "=========================================="
echo ""

# Check for required tools
if ! command -v dpkg-deb &> /dev/null; then
    echo "Error: dpkg-deb not found. Please install dpkg-dev package."
    exit 1
fi

if ! command -v fakeroot &> /dev/null; then
    echo "Error: fakeroot not found. Please install fakeroot package."
    exit 1
fi

# Build NeuronDB
echo "Building NeuronDB DEB package..."
cd "$SCRIPT_DIR/neurondb"
export VERSION="$VERSION"
./build.sh
if [ $? -ne 0 ]; then
    echo "Error: NeuronDB DEB build failed"
    exit 1
fi
echo ""

# Build NeuronAgent
echo "Building NeuronAgent DEB package..."
cd "$SCRIPT_DIR/neuronagent"
export VERSION="$VERSION"
./build.sh
if [ $? -ne 0 ]; then
    echo "Error: NeuronAgent DEB build failed"
    exit 1
fi
echo ""

# Build NeuronMCP
echo "Building NeuronMCP DEB package..."
cd "$SCRIPT_DIR/neuronmcp"
export VERSION="$VERSION"
./build.sh
if [ $? -ne 0 ]; then
    echo "Error: NeuronMCP DEB build failed"
    exit 1
fi
echo ""

echo "=========================================="
echo "All DEB packages built successfully!"
echo "=========================================="
echo ""
echo "Packages:"
ls -lh "$SCRIPT_DIR"/*/*.deb 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Install with: sudo dpkg -i <package.deb>"

