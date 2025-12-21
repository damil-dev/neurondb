#!/bin/bash
set -e

# Build script for NeuronAgent DEB package
# Usage: ./build.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PACKAGE_DIR="$SCRIPT_DIR/package"
VERSION="${VERSION:-1.0.0.beta}"

echo "Building NeuronAgent DEB package version $VERSION"

# Check for Go
if ! command -v go &> /dev/null; then
    echo "Error: Go not found. Please install Go 1.23 or later."
    exit 1
fi

# Clean previous build
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/DEBIAN"
mkdir -p "$PACKAGE_DIR/usr/bin"
mkdir -p "$PACKAGE_DIR/etc/neuronagent"
mkdir -p "$PACKAGE_DIR/usr/share/neuronagent/migrations"
mkdir -p "$PACKAGE_DIR/etc/systemd/system"

# Build NeuronAgent binary
cd "$REPO_ROOT/NeuronAgent"
echo "Building NeuronAgent binary..."

# Download dependencies
go mod download

# Build static binary
CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo \
    -ldflags="-w -s" \
    -o "$PACKAGE_DIR/usr/bin/neuronagent" \
    ./cmd/agent-server

# Copy configuration example
if [ -f "config.yaml" ]; then
    cp config.yaml "$PACKAGE_DIR/etc/neuronagent/config.yaml.example"
elif [ -f "configs/config.yaml.example" ]; then
    cp configs/config.yaml.example "$PACKAGE_DIR/etc/neuronagent/config.yaml.example"
fi

# Copy migrations
if [ -d "migrations" ]; then
    cp -r migrations/* "$PACKAGE_DIR/usr/share/neuronagent/migrations/"
fi

# Copy systemd service file
cp "$SCRIPT_DIR/neuronagent.service" "$PACKAGE_DIR/etc/systemd/system/"

# Copy DEBIAN control files
cp "$SCRIPT_DIR/DEBIAN/control" "$PACKAGE_DIR/DEBIAN/"
cp "$SCRIPT_DIR/DEBIAN/postinst" "$PACKAGE_DIR/DEBIAN/"
cp "$SCRIPT_DIR/DEBIAN/prerm" "$PACKAGE_DIR/DEBIAN/"

# Make scripts executable
chmod +x "$PACKAGE_DIR/DEBIAN/postinst"
chmod +x "$PACKAGE_DIR/DEBIAN/prerm"
chmod +x "$PACKAGE_DIR/usr/bin/neuronagent"

# Update version in control file
sed -i "s/Version: .*/Version: $VERSION/" "$PACKAGE_DIR/DEBIAN/control"

# Build package
echo "Building DEB package..."
cd "$SCRIPT_DIR"
fakeroot dpkg-deb --build package neuronagent_${VERSION}_amd64.deb

echo "Package built: neuronagent_${VERSION}_amd64.deb"



