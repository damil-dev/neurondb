#!/bin/bash
set -e

# Master build script for all RPM packages
# Usage: ./build-all-rpm.sh [VERSION]
# Configuration: Uses build-config.json in packaging/ directory

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PACKAGING_DIR/.." && pwd)"
REPO_RPM_DIR="$REPO_ROOT/repo/rpm"

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

# Function to place RPM package in repository structure
place_rpm_in_repo() {
    local package_file="$1"
    
    if [ ! -f "$package_file" ]; then
        echo "Warning: Package file not found: $package_file"
        return 1
    fi
    
    # Extract EL version from package name (e.g., neurondb-1.0.0.beta-1.el9.x86_64.rpm -> el9)
    local el_version=$(basename "$package_file" | grep -oE '\.el[0-9]+\.' | sed 's/\.el\([0-9]\+\)\./\1/' | head -1)
    
    if [ -z "$el_version" ]; then
        echo "Warning: Could not extract EL version from package name: $package_file"
        echo "Skipping repo placement"
        return 1
    fi
    
    # Create destination directory: repo/rpm/el<version>/x86_64/
    local dest_dir="$REPO_RPM_DIR/el${el_version}/x86_64"
    mkdir -p "$dest_dir"
    
    # Copy package to repo
    cp "$package_file" "$dest_dir/"
    echo "Placed package in repository: $dest_dir/$(basename "$package_file")"
}

# Build NeuronDB
echo "Building NeuronDB RPM package..."
cd "$SCRIPT_DIR/neurondb"
export VERSION="$VERSION"
./build.sh
if [ $? -ne 0 ]; then
    echo "Error: NeuronDB RPM build failed"
    exit 1
fi
# Place package(s) in repo structure
for RPM_FILE in "$SCRIPT_DIR/neurondb"/*.rpm; do
    if [ -f "$RPM_FILE" ]; then
        place_rpm_in_repo "$RPM_FILE"
    fi
done
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
# Place package(s) in repo structure
for RPM_FILE in "$SCRIPT_DIR/neuronagent"/*.rpm; do
    if [ -f "$RPM_FILE" ]; then
        place_rpm_in_repo "$RPM_FILE"
    fi
done
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
# Place package(s) in repo structure
for RPM_FILE in "$SCRIPT_DIR/neuronmcp"/*.rpm; do
    if [ -f "$RPM_FILE" ]; then
        place_rpm_in_repo "$RPM_FILE"
    fi
done
echo ""

# Build NeuronDesktop
echo "Building NeuronDesktop RPM package..."
cd "$SCRIPT_DIR/neurondesktop"
export VERSION="$VERSION"
./build.sh
if [ $? -ne 0 ]; then
    echo "Error: NeuronDesktop RPM build failed"
    exit 1
fi
# Place package(s) in repo structure
for RPM_FILE in "$SCRIPT_DIR/neurondesktop"/*.rpm; do
    if [ -f "$RPM_FILE" ]; then
        place_rpm_in_repo "$RPM_FILE"
    fi
done
echo ""

echo "=========================================="
echo "All RPM packages built successfully!"
echo "=========================================="
echo ""
echo "Packages:"
ls -lh "$SCRIPT_DIR"/*/*.rpm 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Install with: sudo rpm -ivh <package.rpm>"

