#!/bin/bash
#
# Build RPM package for NeuronAgent
#
# Usage:
#   VERSION=1.0.0 ./build.sh
#
# Output:
#   neuronagent-{VERSION}-1.x86_64.rpm

set -euo pipefail

# Version from environment or default
VERSION=${VERSION:-2.0.0}
RELEASE=${RELEASE:-1}
ARCH=${ARCH:-x86_64}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR=$(mktemp -d)
trap "rm -rf $BUILD_DIR" EXIT

echo "Building NeuronAgent RPM package version $VERSION"

# Check prerequisites
command -v rpmbuild >/dev/null 2>&1 || { echo "Error: rpmbuild not found. Install rpm-build." >&2; exit 1; }
command -v go >/dev/null 2>&1 || { echo "Error: Go not found. Install Go 1.21+." >&2; exit 1; }

# Create RPM build directories
RPM_DIR="$BUILD_DIR/rpm"
mkdir -p "$RPM_DIR"/{BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS}

# Build NeuronAgent binary
cd "$REPO_ROOT/NeuronAgent"
INSTALL_DIR="$RPM_DIR/BUILDROOT/neuronagent-${VERSION}-${RELEASE}.${ARCH}"
mkdir -p "$INSTALL_DIR"/usr/bin
mkdir -p "$INSTALL_DIR"/usr/share/neuronagent/migrations
mkdir -p "$INSTALL_DIR"/etc/neuronagent

go build -o "$INSTALL_DIR/usr/bin/neuronagent" ./cmd/neuronagent
cp -r migrations/* "$INSTALL_DIR/usr/share/neuronagent/migrations/" 2>/dev/null || true
[ -f config.yaml.example ] && cp config.yaml.example "$INSTALL_DIR/etc/neuronagent/config.yaml.example"

# Create spec file (simplified - full spec would go here)
echo "RPM spec file creation for NeuronAgent (placeholder)"
# Full RPM spec would be similar to DEB control file structure

echo "Package structure created in $INSTALL_DIR"
echo "Note: Full RPM build requires proper spec file - this is a placeholder"

