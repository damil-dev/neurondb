#!/bin/bash
set -e

# Build script for NeuronMCP macOS .pkg package
# Usage: ./build.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGING_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Load configuration
if [ -f "$PACKAGING_DIR/config-loader.sh" ]; then
    source "$PACKAGING_DIR/config-loader.sh"
    print_config
fi

PACKAGE_DIR="$SCRIPT_DIR/package"
VERSION="${VERSION:-${PACKAGING_VERSION:-1.0.0.beta}}"
ARCH="${ARCH:-${PACKAGING_ARCH:-arm64}}"

# Detect macOS architecture if not specified
if [ -z "$ARCH" ] || [ "$ARCH" = "amd64" ]; then
    ARCH=$(uname -m)
fi

echo "Building NeuronMCP macOS package version $VERSION for $ARCH"

# Check for required tools
if ! command -v pkgbuild &> /dev/null; then
    echo "Error: pkgbuild not found. This script requires macOS and Xcode Command Line Tools."
    exit 1
fi

# Check for Go
if ! command -v go &> /dev/null; then
    echo "Error: Go not found. Please install Go 1.23 or later."
    exit 1
fi

# Clean previous build
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/usr/local/bin"
mkdir -p "$PACKAGE_DIR/etc/neuronmcp"
mkdir -p "$PACKAGE_DIR/Library/LaunchDaemons"
mkdir -p "$PACKAGE_DIR/var/lib/neuronmcp"
mkdir -p "$PACKAGE_DIR/var/log/neuronmcp"
mkdir -p "$PACKAGE_DIR/scripts"

# Build NeuronMCP binary
cd "$REPO_ROOT/NeuronMCP"
echo "Building NeuronMCP binary..."

# Download dependencies
go mod download

# Build binary for macOS
CGO_ENABLED=0 GOOS=darwin GOARCH="$ARCH" go build -a -installsuffix cgo \
    -ldflags="-w -s" \
    -o "$PACKAGE_DIR/usr/local/bin/neurondb-mcp" \
    ./cmd/neurondb-mcp

# Copy configuration example if it exists
if [ -f "mcp-config.json.example" ]; then
    cp mcp-config.json.example "$PACKAGE_DIR/etc/neuronmcp/mcp-config.json.example"
elif [ -f "mcp-config.example.json" ]; then
    cp mcp-config.example.json "$PACKAGE_DIR/etc/neuronmcp/mcp-config.json.example"
fi

# Copy launchd plist file
if [ -f "$SCRIPT_DIR/com.neurondb.neuronmcp.plist" ]; then
    cp "$SCRIPT_DIR/com.neurondb.neuronmcp.plist" "$PACKAGE_DIR/Library/LaunchDaemons/"
fi

# Copy postinstall script
if [ -f "$SCRIPT_DIR/scripts/postinstall" ]; then
    cp "$SCRIPT_DIR/scripts/postinstall" "$PACKAGE_DIR/scripts/"
    chmod +x "$PACKAGE_DIR/scripts/postinstall"
fi

# Make binary executable
chmod +x "$PACKAGE_DIR/usr/local/bin/neurondb-mcp"

# Build component package
echo "Building macOS package..."
COMPONENT_PKG="$SCRIPT_DIR/neuronmcp-${VERSION}-${ARCH}.pkg"

pkgbuild \
    --root "$PACKAGE_DIR" \
    --identifier com.neurondb.neuronmcp \
    --version "$VERSION" \
    --install-location "/" \
    --scripts "$PACKAGE_DIR/scripts" \
    "$COMPONENT_PKG"

echo "Package built: $COMPONENT_PKG"
echo ""
echo "Install with: sudo installer -pkg $COMPONENT_PKG -target /"

