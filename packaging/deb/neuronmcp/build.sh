#!/bin/bash
set -e

# Build script for NeuronMCP DEB package
# Usage: ./build.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PACKAGE_DIR="$SCRIPT_DIR/package"
VERSION="${VERSION:-1.0.0.beta}"

echo "Building NeuronMCP DEB package version $VERSION"

# Check for Go
if ! command -v go &> /dev/null; then
    echo "Error: Go not found. Please install Go 1.23 or later."
    exit 1
fi

# Clean previous build
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/DEBIAN"
mkdir -p "$PACKAGE_DIR/usr/bin"
mkdir -p "$PACKAGE_DIR/etc/neuronmcp"

# Build NeuronMCP binary
cd "$REPO_ROOT/NeuronMCP"
echo "Building NeuronMCP binary..."

# Download dependencies
go mod download

# Build static binary
CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo \
    -ldflags="-w -s" \
    -o "$PACKAGE_DIR/usr/bin/neurondb-mcp" \
    ./cmd/neurondb-mcp

# Copy configuration example if it exists
if [ -f "mcp-config.json.example" ]; then
    cp mcp-config.json.example "$PACKAGE_DIR/etc/neuronmcp/mcp-config.json.example"
elif [ -f "mcp-config.example.json" ]; then
    cp mcp-config.example.json "$PACKAGE_DIR/etc/neuronmcp/mcp-config.json.example"
fi

# Copy DEBIAN control files
cp "$SCRIPT_DIR/DEBIAN/control" "$PACKAGE_DIR/DEBIAN/"
if [ -f "$SCRIPT_DIR/DEBIAN/postinst" ]; then
    cp "$SCRIPT_DIR/DEBIAN/postinst" "$PACKAGE_DIR/DEBIAN/"
    chmod +x "$PACKAGE_DIR/DEBIAN/postinst"
fi
if [ -f "$SCRIPT_DIR/DEBIAN/prerm" ]; then
    cp "$SCRIPT_DIR/DEBIAN/prerm" "$PACKAGE_DIR/DEBIAN/"
    chmod +x "$PACKAGE_DIR/DEBIAN/prerm"
fi

chmod +x "$PACKAGE_DIR/usr/bin/neurondb-mcp"

# Update version in control file
sed -i "s/Version: .*/Version: $VERSION/" "$PACKAGE_DIR/DEBIAN/control"

# Build package
echo "Building DEB package..."
cd "$SCRIPT_DIR"
fakeroot dpkg-deb --build package neuronmcp_${VERSION}_amd64.deb

echo "Package built: neuronmcp_${VERSION}_amd64.deb"
