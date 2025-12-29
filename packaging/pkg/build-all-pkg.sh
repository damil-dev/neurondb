#!/bin/bash
set -e

# Master build script for all macOS .pkg packages
# Usage: ./build-all-pkg.sh [VERSION]
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
echo "Building all macOS packages version $VERSION"
echo "=========================================="
echo ""

# Check for required tools
if ! command -v pkgbuild &> /dev/null; then
    echo "Error: pkgbuild not found. This script requires macOS and Xcode Command Line Tools."
    exit 1
fi

# Build NeuronDB
echo "Building NeuronDB macOS package..."
cd "$SCRIPT_DIR/neurondb"
export VERSION="$VERSION"
./build.sh
if [ $? -ne 0 ]; then
    echo "Error: NeuronDB macOS build failed"
    exit 1
fi
echo ""

# Build NeuronAgent
echo "Building NeuronAgent macOS package..."
cd "$SCRIPT_DIR/neuronagent"
export VERSION="$VERSION"
./build.sh
if [ $? -ne 0 ]; then
    echo "Error: NeuronAgent macOS build failed"
    exit 1
fi
echo ""

# Build NeuronMCP
echo "Building NeuronMCP macOS package..."
cd "$SCRIPT_DIR/neuronmcp"
export VERSION="$VERSION"
./build.sh
if [ $? -ne 0 ]; then
    echo "Error: NeuronMCP macOS build failed"
    exit 1
fi
echo ""

# Build NeuronDesktop
echo "Building NeuronDesktop macOS package..."
cd "$SCRIPT_DIR/neurondesktop"
export VERSION="$VERSION"
./build.sh
if [ $? -ne 0 ]; then
    echo "Error: NeuronDesktop macOS build failed"
    exit 1
fi
echo ""

echo "=========================================="
echo "All macOS packages built successfully!"
echo "=========================================="
echo ""
echo "Packages:"
ARCH=$(uname -m)
ls -lh "$SCRIPT_DIR"/*/*-${VERSION}-${ARCH}*.pkg 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Install with: sudo installer -pkg <package.pkg> -target /"

