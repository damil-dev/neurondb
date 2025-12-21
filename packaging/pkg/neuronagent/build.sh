#!/bin/bash
set -e

# Build script for NeuronAgent macOS .pkg package
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

echo "Building NeuronAgent macOS package version $VERSION for $ARCH"

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
mkdir -p "$PACKAGE_DIR/etc/neuronagent"
mkdir -p "$PACKAGE_DIR/usr/local/share/neuronagent/migrations"
mkdir -p "$PACKAGE_DIR/scripts"

# Build NeuronAgent binary
cd "$REPO_ROOT/NeuronAgent"
echo "Building NeuronAgent binary..."

# Download dependencies
go mod download

# Build binary for macOS
CGO_ENABLED=0 GOOS=darwin GOARCH="$ARCH" go build -a -installsuffix cgo \
    -ldflags="-w -s" \
    -o "$PACKAGE_DIR/usr/local/bin/neuronagent" \
    ./cmd/agent-server

# Copy configuration example
if [ -f "config.yaml" ]; then
    cp config.yaml "$PACKAGE_DIR/etc/neuronagent/config.yaml.example"
elif [ -f "configs/config.yaml.example" ]; then
    cp configs/config.yaml.example "$PACKAGE_DIR/etc/neuronagent/config.yaml.example"
fi

# Copy migrations
if [ -d "migrations" ]; then
    cp -r migrations/* "$PACKAGE_DIR/usr/local/share/neuronagent/migrations/"
fi

# Copy postinstall script
if [ -f "$SCRIPT_DIR/scripts/postinstall" ]; then
    cp "$SCRIPT_DIR/scripts/postinstall" "$PACKAGE_DIR/scripts/"
    chmod +x "$PACKAGE_DIR/scripts/postinstall"
fi

# Make binary executable
chmod +x "$PACKAGE_DIR/usr/local/bin/neuronagent"

# Build component package
echo "Building macOS package..."
COMPONENT_PKG="$SCRIPT_DIR/neuronagent-${VERSION}-${ARCH}.pkg"

pkgbuild \
    --root "$PACKAGE_DIR" \
    --identifier com.neurondb.neuronagent \
    --version "$VERSION" \
    --install-location "/" \
    --scripts "$PACKAGE_DIR/scripts" \
    "$COMPONENT_PKG"

echo "Package built: $COMPONENT_PKG"
echo ""
echo "Install with: sudo installer -pkg $COMPONENT_PKG -target /"

