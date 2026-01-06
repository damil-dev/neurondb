#!/bin/bash
#
# Build DEB package for NeuronDesktop
#
# Usage:
#   VERSION=1.0.0 ./build.sh
#
# Output:
#   neurondesktop_{VERSION}_amd64.deb

set -euo pipefail

# Version from environment or default
VERSION=${VERSION:-1.0.0}
# Strip 'v' prefix if present (Debian packages require versions to start with digit)
VERSION=${VERSION#v}
ARCH=${ARCH:-amd64}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR=$(mktemp -d)
trap "rm -rf $BUILD_DIR" EXIT

echo "Building NeuronDesktop DEB package version $VERSION"

# Check prerequisites
command -v dpkg-deb >/dev/null 2>&1 || { echo "Error: dpkg-deb not found. Install dpkg-dev." >&2; exit 1; }
command -v fakeroot >/dev/null 2>&1 || { echo "Error: fakeroot not found. Install fakeroot." >&2; exit 1; }
command -v go >/dev/null 2>&1 || { echo "Error: Go not found. Install Go 1.21+." >&2; exit 1; }

# Create package structure
PKG_DIR="$BUILD_DIR/neurondesktop_${VERSION}"
mkdir -p "$PKG_DIR/DEBIAN"
mkdir -p "$PKG_DIR/usr/bin"
mkdir -p "$PKG_DIR/usr/share/neurondesktop/migrations"
mkdir -p "$PKG_DIR/etc/neurondesktop"
mkdir -p "$PKG_DIR/usr/share/doc/neurondesktop"

# Build NeuronDesktop API binary
cd "$REPO_ROOT/NeuronDesktop/api"
go build -o "$PKG_DIR/usr/bin/neurondesktop" ./cmd/server

# Copy migrations
cp -r migrations/* "$PKG_DIR/usr/share/neurondesktop/migrations/" 2>/dev/null || true

# Copy SQL setup files
if [ -f neurondesktop.sql ]; then
    cp neurondesktop.sql "$PKG_DIR/usr/share/neurondesktop/"
fi

# Create control file
cat > "$PKG_DIR/DEBIAN/control" <<EOF
Package: neurondesktop
Version: ${VERSION}
Architecture: ${ARCH}
Maintainer: neurondb <admin@neurondb.com>
Description: Web-based desktop interface for NeuronDB
 NeuronDesktop provides:
  - Web UI for database management
  - SQL console with AI assistance
  - Agent management interface
  - REST API backend
Depends: libc6 (>= 2.17), postgresql-client
Section: web
Priority: optional
Homepage: https://github.com/neurondb/neurondb
EOF

# Create postinst script
cat > "$PKG_DIR/DEBIAN/postinst" <<'EOF'
#!/bin/bash
set -e
echo "NeuronDesktop API installed at /usr/bin/neurondesktop"
echo "Note: Frontend must be deployed separately (Next.js app)"
EOF
chmod +x "$PKG_DIR/DEBIAN/postinst"

# Create copyright file
cat > "$PKG_DIR/usr/share/doc/neurondesktop/copyright" <<EOF
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: NeuronDesktop
Source: https://github.com/neurondb/neurondb
Files: *
Copyright: 2024 NeuronDB
License: Proprietary
EOF

# Build DEB package
DEB_FILE="neurondesktop_${VERSION}_${ARCH}.deb"
fakeroot dpkg-deb --build "$PKG_DIR" "$REPO_ROOT/packaging/deb/neurondesktop/$DEB_FILE"

echo "Package built: $DEB_FILE"
ls -lh "$REPO_ROOT/packaging/deb/neurondesktop/$DEB_FILE"

