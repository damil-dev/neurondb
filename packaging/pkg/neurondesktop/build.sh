#!/bin/bash
set -e

# Build script for NeuronDesktop macOS .pkg package
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

echo "Building NeuronDesktop macOS package version $VERSION for $ARCH"

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

# Check for Node.js and npm
if ! command -v node &> /dev/null; then
    echo "Error: Node.js not found. Please install Node.js 18+ and npm."
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "Error: npm not found. Please install npm."
    exit 1
fi

# Clean previous build
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/usr/local/bin"
mkdir -p "$PACKAGE_DIR/etc/neurondesktop"
mkdir -p "$PACKAGE_DIR/var/www/neurondesktop"
mkdir -p "$PACKAGE_DIR/Library/LaunchDaemons"
mkdir -p "$PACKAGE_DIR/var/lib/neurondesktop"
mkdir -p "$PACKAGE_DIR/var/log/neurondesktop"
mkdir -p "$PACKAGE_DIR/scripts"

# Build NeuronDesktop API binary
cd "$REPO_ROOT/NeuronDesktop/api"
echo "Building NeuronDesktop API binary..."

# Download dependencies
go mod download

# Build binary for macOS
CGO_ENABLED=0 GOOS=darwin GOARCH="$ARCH" go build -a -installsuffix cgo \
    -ldflags="-w -s" \
    -o "$PACKAGE_DIR/usr/local/bin/neurondesktop-api" \
    ./cmd/server

# Build NeuronDesktop frontend
cd "$REPO_ROOT/NeuronDesktop/frontend"
echo "Building NeuronDesktop frontend..."

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    npm install
fi

# Build production frontend
npm run build

# Copy frontend build to package
if [ -d ".next" ]; then
    cp -r .next "$PACKAGE_DIR/var/www/neurondesktop/"
fi
if [ -d "public" ]; then
    cp -r public "$PACKAGE_DIR/var/www/neurondesktop/" || true
fi

# Copy migrations if they exist
if [ -d "$REPO_ROOT/NeuronDesktop/api/migrations" ]; then
    mkdir -p "$PACKAGE_DIR/usr/local/share/neurondesktop/migrations"
    cp -r "$REPO_ROOT/NeuronDesktop/api/migrations"/* "$PACKAGE_DIR/usr/local/share/neurondesktop/migrations/"
fi

# Copy launchd plist file
if [ -f "$SCRIPT_DIR/com.neurondb.neurondesktop.plist" ]; then
    cp "$SCRIPT_DIR/com.neurondb.neurondesktop.plist" "$PACKAGE_DIR/Library/LaunchDaemons/"
fi

# Copy postinstall script
if [ -f "$SCRIPT_DIR/scripts/postinstall" ]; then
    cp "$SCRIPT_DIR/scripts/postinstall" "$PACKAGE_DIR/scripts/"
    chmod +x "$PACKAGE_DIR/scripts/postinstall"
fi

# Make binary executable
chmod +x "$PACKAGE_DIR/usr/local/bin/neurondesktop-api"

# Build component package
echo "Building macOS package..."
COMPONENT_PKG="$SCRIPT_DIR/neurondesktop-${VERSION}-${ARCH}.pkg"

pkgbuild \
    --root "$PACKAGE_DIR" \
    --identifier com.neurondb.neurondesktop \
    --version "$VERSION" \
    --install-location "/" \
    --scripts "$PACKAGE_DIR/scripts" \
    "$COMPONENT_PKG"

echo "Package built: $COMPONENT_PKG"
echo ""
echo "Install with: sudo installer -pkg $COMPONENT_PKG -target /"


