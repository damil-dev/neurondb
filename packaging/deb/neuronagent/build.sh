#!/bin/bash
#
# Build DEB package for NeuronAgent
#
# Usage:
#   VERSION=1.0.0 ./build.sh
#
# Output:
#   neuronagent_{VERSION}_amd64.deb

set -euo pipefail

# Version from environment or default
VERSION=${VERSION:-2.0.0}
ARCH=${ARCH:-amd64}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR=$(mktemp -d)
trap "rm -rf $BUILD_DIR" EXIT

echo "Building NeuronAgent DEB package version $VERSION"

# Check prerequisites
command -v dpkg-deb >/dev/null 2>&1 || { echo "Error: dpkg-deb not found. Install dpkg-dev." >&2; exit 1; }
command -v fakeroot >/dev/null 2>&1 || { echo "Error: fakeroot not found. Install fakeroot." >&2; exit 1; }
command -v go >/dev/null 2>&1 || { echo "Error: Go not found. Install Go 1.21+." >&2; exit 1; }

# Create package structure
PKG_DIR="$BUILD_DIR/neuronagent_${VERSION}"
mkdir -p "$PKG_DIR/DEBIAN"
mkdir -p "$PKG_DIR/usr/bin"
mkdir -p "$PKG_DIR/usr/share/neuronagent/migrations"
mkdir -p "$PKG_DIR/etc/neuronagent"
mkdir -p "$PKG_DIR/usr/share/doc/neuronagent"

# Build NeuronAgent binary
cd "$REPO_ROOT/NeuronAgent"
go build -o "$PKG_DIR/usr/bin/neuronagent" ./cmd/agent-server

# Copy migrations
cp -r migrations/* "$PKG_DIR/usr/share/neuronagent/migrations/" 2>/dev/null || true

# Copy example config
if [ -f config.yaml.example ]; then
    cp config.yaml.example "$PKG_DIR/etc/neuronagent/config.yaml.example"
fi

# Create control file
cat > "$PKG_DIR/DEBIAN/control" <<EOF
Package: neuronagent
Version: ${VERSION}
Architecture: ${ARCH}
Maintainer: neurondb <admin@neurondb.com>
Description: AI agent runtime system providing REST API and WebSocket endpoints
 NeuronAgent provides:
  - REST/WebSocket API for agent management
  - Multi-agent collaboration
  - Workflow engine and HITL support
  - Tool execution and memory management
  - Evaluation framework
Depends: libc6 (>= 2.17), postgresql-client
Section: net
Priority: optional
Homepage: https://github.com/neurondb/neurondb
EOF

# Create postinst script
cat > "$PKG_DIR/DEBIAN/postinst" <<'EOF'
#!/bin/bash
set -e
# Create systemd service file if systemd is available
if command -v systemctl >/dev/null 2>&1; then
    echo "To enable NeuronAgent service:"
    echo "  sudo systemctl enable neuronagent"
    echo "  sudo systemctl start neuronagent"
fi
EOF
chmod +x "$PKG_DIR/DEBIAN/postinst"

# Create copyright file
cat > "$PKG_DIR/usr/share/doc/neuronagent/copyright" <<EOF
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: NeuronAgent
Source: https://github.com/neurondb/neurondb

Files: *
Copyright: 2024 NeuronDB
License: Proprietary
EOF

# Build DEB package
DEB_FILE="neuronagent_${VERSION}_${ARCH}.deb"
fakeroot dpkg-deb --build "$PKG_DIR" "$REPO_ROOT/packaging/deb/neuronagent/$DEB_FILE"

echo "Package built: $DEB_FILE"
ls -lh "$REPO_ROOT/packaging/deb/neuronagent/$DEB_FILE"

