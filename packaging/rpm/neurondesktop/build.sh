#!/bin/bash
set -e

# Build script for NeuronDesktop RPM package
# Usage: ./build.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VERSION="${VERSION:-1.0.0.beta}"

echo "Building NeuronDesktop RPM package version $VERSION"

# Check for required tools
if ! command -v rpmbuild &> /dev/null; then
    echo "Error: rpmbuild not found. Please install rpm-build package."
    exit 1
fi

# Check for Go
if ! command -v go &> /dev/null; then
    echo "Error: Go not found. Please install Go 1.23 or later."
    exit 1
fi

# Check for Node.js and npm
if ! command -v node &> /dev/null; then
    echo "Error: Node.js not found. Please install Node.js 18+ and npm."
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "Error: npm not found. Please install npm."
    exit 1
fi

# Create source tarball
cd "$REPO_ROOT"
TARBALL="neurondesktop-${VERSION}.tar.gz"
tar czf "$TARBALL" NeuronDesktop/ --exclude='.git' --exclude='node_modules' --exclude='.next'

# Move tarball to SOURCES
mkdir -p ~/rpmbuild/SOURCES
mv "$TARBALL" ~/rpmbuild/SOURCES/

# Copy spec file
mkdir -p ~/rpmbuild/SPECS
cp "$SCRIPT_DIR/neurondesktop.spec" ~/rpmbuild/SPECS/

# Update version in spec file
sed -i "s/^%define version .*/%define version $VERSION/" ~/rpmbuild/SPECS/neurondesktop.spec

# Build RPM
echo "Building RPM package..."
rpmbuild -ba ~/rpmbuild/SPECS/neurondesktop.spec

# Find and copy built package
RPM_FILE=$(find ~/rpmbuild/RPMS -name "neurondesktop-${VERSION}-*.rpm" | head -1)
if [ -n "$RPM_FILE" ]; then
    cp "$RPM_FILE" "$SCRIPT_DIR/"
    echo "Package built: $(basename "$RPM_FILE")"
else
    echo "Error: RPM package not found"
    exit 1
fi


