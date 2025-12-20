#!/bin/bash
set -e

# Master build script for all RPM packages
# Usage: ./build-all-rpm.sh [VERSION]
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
echo "Building all RPM packages version $VERSION"
echo "=========================================="
echo ""

# Check for required tools
if ! command -v rpmbuild &> /dev/null; then
    echo "Error: rpmbuild not found. Please install rpm-build package."
    exit 1
fi

# Build NeuronDB
echo "Building NeuronDB RPM package..."
cd "$SCRIPT_DIR/neurondb"
export VERSION="$VERSION"
./build.sh
if [ $? -ne 0 ]; then
    echo "Error: NeuronDB RPM build failed"
    exit 1
fi
echo ""

# Build NeuronAgent
echo "Building NeuronAgent RPM package..."
cd "$SCRIPT_DIR/neuronagent"
export VERSION="$VERSION"
./build.sh
if [ $? -ne 0 ]; then
    echo "Error: NeuronAgent RPM build failed"
    exit 1
fi
echo ""

# Build NeuronMCP
echo "Building NeuronMCP RPM package..."
cd "$SCRIPT_DIR/neuronmcp"
export VERSION="$VERSION"
./build.sh
if [ $? -ne 0 ]; then
    echo "Error: NeuronMCP RPM build failed"
    exit 1
fi
echo ""

echo "=========================================="
echo "All RPM packages built successfully!"
echo "=========================================="
echo ""
echo "Packages:"
ls -lh "$SCRIPT_DIR"/*/*.rpm 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Install with: sudo rpm -ivh <package.rpm>"

