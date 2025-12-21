#!/bin/bash
set -e

# Master build script for all DEB packages
# Usage: ./build-all-deb.sh [VERSION]
# Configuration: Uses build-config.json in packaging/ directory

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PACKAGING_DIR/.." && pwd)"
REPO_DEB_DIR="$REPO_ROOT/repo/deb/pool/main"

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

# Function to place DEB package in repository structure
place_deb_in_repo() {
    local package_file="$1"
    local component="$2"
    
    if [ ! -f "$package_file" ]; then
        echo "Warning: Package file not found: $package_file"
        return 1
    fi
    
    # Extract package name from filename (e.g., neurondb_1.0.0.beta_amd64.deb -> neurondb)
    local package_name=$(basename "$package_file" | sed -E 's/^([^_]+)_.*/\1/')
    
    # Use component name if provided, otherwise use package name
    local repo_component="${component:-$package_name}"
    
    # Create destination directory: repo/deb/pool/main/n/<component>/
    local dest_dir="$REPO_DEB_DIR/n/$repo_component"
    mkdir -p "$dest_dir"
    
    # Copy package to repo
    cp "$package_file" "$dest_dir/"
    echo "Placed package in repository: $dest_dir/$(basename "$package_file")"
}

# Build NeuronDB
echo "Building NeuronDB DEB package..."
cd "$SCRIPT_DIR/neurondb"
export VERSION="$VERSION"
./build.sh
if [ $? -ne 0 ]; then
    echo "Error: NeuronDB DEB build failed"
    exit 1
fi
# Place package in repo structure
DEB_FILE=$(ls -1 "$SCRIPT_DIR/neurondb"/*.deb 2>/dev/null | head -1)
if [ -n "$DEB_FILE" ]; then
    place_deb_in_repo "$DEB_FILE" "neurondb"
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
# Place package in repo structure
DEB_FILE=$(ls -1 "$SCRIPT_DIR/neuronagent"/*.deb 2>/dev/null | head -1)
if [ -n "$DEB_FILE" ]; then
    place_deb_in_repo "$DEB_FILE" "neuronagent"
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
# Place package in repo structure
DEB_FILE=$(ls -1 "$SCRIPT_DIR/neuronmcp"/*.deb 2>/dev/null | head -1)
if [ -n "$DEB_FILE" ]; then
    place_deb_in_repo "$DEB_FILE" "neuronmcp"
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

