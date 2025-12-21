#!/bin/bash
set -e

# Build script for NeuronMCP RPM package
# Usage: ./build.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VERSION="${VERSION:-1.0.0.beta}"

echo "Building NeuronMCP RPM package version $VERSION"

# Check for rpmbuild
if ! command -v rpmbuild &> /dev/null; then
    echo "Error: rpmbuild not found. Please install rpm-build package."
    exit 1
fi

# Check for Go
if ! command -v go &> /dev/null; then
    echo "Error: Go not found. Please install Go 1.23 or later."
    exit 1
fi

# Create source tarball
cd "$REPO_ROOT"
TARBALL_NAME="neuronmcp-${VERSION}.tar.gz"
TEMP_DIR=$(mktemp -d)

# Copy NeuronMCP source to temp directory
cp -r NeuronMCP "$TEMP_DIR/neuronmcp-${VERSION}"

# Create tarball
cd "$TEMP_DIR"
tar czf "$REPO_ROOT/packaging/rpm/neuronmcp/$TARBALL_NAME" "neuronmcp-${VERSION}"
rm -rf "$TEMP_DIR"

# Setup RPM build directories
RPMBUILD_DIR="$HOME/rpmbuild"
mkdir -p "$RPMBUILD_DIR"/{BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS}

# Copy source and spec
cp "$REPO_ROOT/packaging/rpm/neuronmcp/$TARBALL_NAME" "$RPMBUILD_DIR/SOURCES/"
cp "$REPO_ROOT/packaging/rpm/neuronmcp/neuronmcp.spec" "$RPMBUILD_DIR/SPECS/"

# Update version in spec file
sed -i "s/Version: .*/Version: $VERSION/" "$RPMBUILD_DIR/SPECS/neuronmcp.spec"

# Build RPM
echo "Building RPM..."
rpmbuild -ba "$RPMBUILD_DIR/SPECS/neuronmcp.spec"

# Copy built RPM
cp "$RPMBUILD_DIR/RPMS"/*/neuronmcp-${VERSION}-*.rpm "$REPO_ROOT/packaging/rpm/neuronmcp/" 2>/dev/null || true

echo "RPM package built: $(ls -1 $REPO_ROOT/packaging/rpm/neuronmcp/neuronmcp-${VERSION}-*.rpm 2>/dev/null | head -1)"



