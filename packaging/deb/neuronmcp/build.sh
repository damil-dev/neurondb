#!/bin/bash
#
# Build DEB package for NeuronMCP
#
# Usage:
#   VERSION=1.0.0 ./build.sh
#
# Output:
#   neuronmcp_{VERSION}_amd64.deb

set -euo pipefail

# Version from environment or default
VERSION=${VERSION:-2.0.0}
ARCH=${ARCH:-amd64}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR=$(mktemp -d)
trap "rm -rf $BUILD_DIR" EXIT

echo "Building NeuronMCP DEB package version $VERSION"

# Check prerequisites
command -v dpkg-deb >/dev/null 2>&1 || { echo "Error: dpkg-deb not found. Install dpkg-dev." >&2; exit 1; }
command -v fakeroot >/dev/null 2>&1 || { echo "Error: fakeroot not found. Install fakeroot." >&2; exit 1; }
command -v go >/dev/null 2>&1 || { echo "Error: Go not found. Install Go 1.21+." >&2; exit 1; }

# Create package structure
PKG_DIR="$BUILD_DIR/neuronmcp_${VERSION}"
mkdir -p "$PKG_DIR/DEBIAN"
mkdir -p "$PKG_DIR/usr/bin"
mkdir -p "$PKG_DIR/usr/share/neuronmcp/sql"
mkdir -p "$PKG_DIR/etc/neuronmcp"
mkdir -p "$PKG_DIR/usr/share/doc/neuronmcp"

# Build NeuronMCP binary
cd "$REPO_ROOT/NeuronMCP"
go build -o "$PKG_DIR/usr/bin/neurondb-mcp" ./cmd/neurondb-mcp

# Copy SQL files
cp -r sql/* "$PKG_DIR/usr/share/neuronmcp/sql/" 2>/dev/null || true

# Copy example config
if [ -f mcp-config.json.example ]; then
    cp mcp-config.json.example "$PKG_DIR/etc/neuronmcp/mcp-config.json.example"
fi

# Create control file
cat > "$PKG_DIR/DEBIAN/control" <<EOF
Package: neuronmcp
Version: ${VERSION}
Architecture: ${ARCH}
Maintainer: neurondb <admin@neurondb.com>
Description: Model Context Protocol server for NeuronDB PostgreSQL extension
 NeuronMCP provides:
  - MCP protocol server (JSON-RPC 2.0)
  - 100+ tools for database operations
  - Resource access for NeuronDB
  - Integration with Claude Desktop and other MCP clients
Depends: libc6 (>= 2.17), postgresql-client
Section: net
Priority: optional
Homepage: https://github.com/neurondb/neurondb
EOF

# Create postinst script
cat > "$PKG_DIR/DEBIAN/postinst" <<'EOF'
#!/bin/bash
set -e
echo "NeuronMCP installed. Configure MCP clients to use /usr/bin/neurondb-mcp"
EOF
chmod +x "$PKG_DIR/DEBIAN/postinst"

# Create copyright file
cat > "$PKG_DIR/usr/share/doc/neuronmcp/copyright" <<EOF
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: NeuronMCP
Source: https://github.com/neurondb/neurondb

Files: *
Copyright: 2024 NeuronDB
License: Proprietary
EOF

# Build DEB package
DEB_FILE="neuronmcp_${VERSION}_${ARCH}.deb"
fakeroot dpkg-deb --build "$PKG_DIR" "$REPO_ROOT/packaging/deb/neuronmcp/$DEB_FILE"

echo "Package built: $DEB_FILE"
ls -lh "$REPO_ROOT/packaging/deb/neuronmcp/$DEB_FILE"

