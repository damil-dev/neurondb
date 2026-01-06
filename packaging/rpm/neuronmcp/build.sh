#!/bin/bash
#
# Build RPM package for NeuronMCP
#
# Usage:
#   VERSION=1.0.0 ./build.sh
#
# Output:
#   neuronmcp-{VERSION}-1.x86_64.rpm

set -euo pipefail

# Version from environment or default
VERSION=${VERSION:-1.0.0.beta}
RELEASE=${RELEASE:-1}
ARCH=${ARCH:-x86_64}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR=$(mktemp -d)
trap "rm -rf $BUILD_DIR" EXIT

echo "Building NeuronMCP RPM package version $VERSION"

# Check prerequisites
command -v rpmbuild >/dev/null 2>&1 || { echo "Error: rpmbuild not found. Install rpm-build." >&2; exit 1; }
command -v go >/dev/null 2>&1 || { echo "Error: Go not found. Install Go 1.21+." >&2; exit 1; }

# Create RPM build directories
RPM_DIR="$BUILD_DIR/rpm"
mkdir -p "$RPM_DIR"/{BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS}

# Build NeuronMCP binary
cd "$REPO_ROOT/NeuronMCP"
INSTALL_DIR="$RPM_DIR/BUILDROOT/neuronmcp-${VERSION}-${RELEASE}.${ARCH}"
mkdir -p "$INSTALL_DIR"/usr/bin
mkdir -p "$INSTALL_DIR"/usr/share/neuronmcp/sql
mkdir -p "$INSTALL_DIR"/etc/neuronmcp

go build -o "$INSTALL_DIR/usr/bin/neurondb-mcp" ./cmd/neuronmcp
cp -r sql/* "$INSTALL_DIR/usr/share/neuronmcp/sql/" 2>/dev/null || true
[ -f mcp-config.json.example ] && cp mcp-config.json.example "$INSTALL_DIR/etc/neuronmcp/mcp-config.json.example"

# Create spec file (simplified - full spec would go here)
echo "RPM spec file creation for NeuronMCP (placeholder)"
# Full RPM spec would be similar to DEB control file structure

echo "Package structure created in $INSTALL_DIR"
echo "Note: Full RPM build requires proper spec file - this is a placeholder"


