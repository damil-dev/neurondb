#!/bin/bash
set -e

# Build script for NeuronDB RPM package
# Usage: ./build.sh
# Configuration: Uses build-config.json in packaging/ directory

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGING_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Load configuration
if [ -f "$PACKAGING_DIR/config-loader.sh" ]; then
    source "$PACKAGING_DIR/config-loader.sh"
    print_config
else
    echo "Warning: config-loader.sh not found, using defaults"
    VERSION="${VERSION:-1.0.0.beta}"
    GPU_BACKENDS="${GPU_BACKENDS:-none}"
fi

VERSION="${VERSION:-${PACKAGING_VERSION:-1.0.0.beta}}"
ARCH="${ARCH:-${PACKAGING_ARCH:-amd64}}"

echo "Building NeuronDB RPM package version $VERSION"

# Check for rpmbuild
if ! command -v rpmbuild &> /dev/null; then
    echo "Error: rpmbuild not found. Please install rpm-build package."
    exit 1
fi

# Create source tarball
cd "$REPO_ROOT"
TARBALL_NAME="neurondb-${VERSION}.tar.gz"
TEMP_DIR=$(mktemp -d)

# Copy NeuronDB source to temp directory
cp -r NeuronDB "$TEMP_DIR/neurondb-${VERSION}"

# Create tarball
cd "$TEMP_DIR"
tar czf "$REPO_ROOT/packaging/rpm/neurondb/$TARBALL_NAME" "neurondb-${VERSION}"
rm -rf "$TEMP_DIR"

# Setup RPM build directories
RPMBUILD_DIR="$HOME/rpmbuild"
mkdir -p "$RPMBUILD_DIR"/{BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS}

# Copy source and spec
cp "$REPO_ROOT/packaging/rpm/neurondb/$TARBALL_NAME" "$RPMBUILD_DIR/SOURCES/"
cp "$REPO_ROOT/packaging/rpm/neurondb/neurondb.spec" "$RPMBUILD_DIR/SPECS/"

# Update version in spec file
sed -i "s/Version: .*/Version: $VERSION/" "$RPMBUILD_DIR/SPECS/neurondb.spec"

# Add GPU dependencies if GPU backend is enabled
if [ -f "$PACKAGING_DIR/config-loader.sh" ]; then
    GPU_DEPS=$(get_gpu_deps_rpm)
    if [ -n "$GPU_DEPS" ]; then
        # Add GPU dependencies to the Requires line
        # This appends the GPU deps after the existing requirements
        sed -i "s/^Requires: \(.*\)$/Requires: \1, $GPU_DEPS/" "$RPMBUILD_DIR/SPECS/neurondb.spec"
        echo "Added GPU dependencies: $GPU_DEPS"
    fi
fi

# Extract source for build
cd "$RPMBUILD_DIR/BUILD"
tar xzf "$RPMBUILD_DIR/SOURCES/$TARBALL_NAME"
cd "neurondb-${VERSION}"

# Build RPM
echo "Building RPM..."
rpmbuild -ba "$RPMBUILD_DIR/SPECS/neurondb.spec"

# Copy built RPM
cp "$RPMBUILD_DIR/RPMS"/*/neurondb-${VERSION}-*.rpm "$REPO_ROOT/packaging/rpm/neurondb/" 2>/dev/null || true

echo "RPM package built: $(ls -1 $REPO_ROOT/packaging/rpm/neurondb/neurondb-${VERSION}-*.rpm 2>/dev/null | head -1)"

