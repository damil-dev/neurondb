#!/bin/bash
set -e

# Build script for NeuronDesktop DEB package
# Usage: ./build.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PACKAGE_DIR="$SCRIPT_DIR/package"
VERSION="${VERSION:-1.0.0.beta}"

echo "Building NeuronDesktop DEB package version $VERSION"

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
mkdir -p "$PACKAGE_DIR/DEBIAN"
mkdir -p "$PACKAGE_DIR/usr/bin"
mkdir -p "$PACKAGE_DIR/etc/neurondesktop"
mkdir -p "$PACKAGE_DIR/var/www/neurondesktop"
mkdir -p "$PACKAGE_DIR/etc/systemd/system"

# Build NeuronDesktop API binary
cd "$REPO_ROOT/NeuronDesktop/api"
echo "Building NeuronDesktop API binary..."

# Download dependencies
go mod download

# Build static binary
CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo \
    -ldflags="-w -s" \
    -o "$PACKAGE_DIR/usr/bin/neurondesktop-api" \
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
    mkdir -p "$PACKAGE_DIR/usr/share/neurondesktop/migrations"
    cp -r "$REPO_ROOT/NeuronDesktop/api/migrations"/* "$PACKAGE_DIR/usr/share/neurondesktop/migrations/"
fi

# Copy systemd service file
cp "$SCRIPT_DIR/neurondesktop.service" "$PACKAGE_DIR/etc/systemd/system/"

# Copy DEBIAN control files
cp "$SCRIPT_DIR/DEBIAN/control" "$PACKAGE_DIR/DEBIAN/"
cp "$SCRIPT_DIR/DEBIAN/postinst" "$PACKAGE_DIR/DEBIAN/"
cp "$SCRIPT_DIR/DEBIAN/prerm" "$PACKAGE_DIR/DEBIAN/"

# Make scripts executable
chmod +x "$PACKAGE_DIR/DEBIAN/postinst"
chmod +x "$PACKAGE_DIR/DEBIAN/prerm"
chmod +x "$PACKAGE_DIR/usr/bin/neurondesktop-api"

# Update version in control file
sed -i "s/Version: .*/Version: $VERSION/" "$PACKAGE_DIR/DEBIAN/control"

# Build package
echo "Building DEB package..."
cd "$SCRIPT_DIR"
fakeroot dpkg-deb --build package neurondesktop_${VERSION}_amd64.deb

echo "Package built: neurondesktop_${VERSION}_amd64.deb"


