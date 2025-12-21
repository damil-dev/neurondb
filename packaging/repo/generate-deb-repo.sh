#!/bin/bash
set -e

# Generate DEB repository metadata and sign with GPG
# Usage: ./generate-deb-repo.sh [GPG_KEY_ID]
# If GPG_KEY_ID is not provided, signing will be skipped

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PACKAGING_DIR/.." && pwd)"

REPO_DEB_DIR="$REPO_ROOT/repo/deb"
POOL_DIR="$REPO_DEB_DIR/pool/main"
DISTS_DIR="$REPO_DEB_DIR/dists/stable"
BINARY_DIR="$DISTS_DIR/main/binary-amd64"

GPG_KEY_ID="${1:-}"

echo "=========================================="
echo "Generating DEB repository metadata"
echo "=========================================="
echo ""

# Check for required tools
if ! command -v dpkg-scanpackages &> /dev/null; then
    echo "Error: dpkg-scanpackages not found. Please install dpkg-dev package."
    exit 1
fi

if ! command -v apt-ftparchive &> /dev/null; then
    echo "Error: apt-ftparchive not found. Please install apt-utils package."
    exit 1
fi

# Check if pool directory exists and has packages
if [ ! -d "$POOL_DIR" ]; then
    echo "Error: Pool directory not found: $POOL_DIR"
    echo "Please build packages first using packaging/deb/build-all-deb.sh"
    exit 1
fi

PACKAGE_COUNT=$(find "$POOL_DIR" -name "*.deb" 2>/dev/null | wc -l)
if [ "$PACKAGE_COUNT" -eq 0 ]; then
    echo "Warning: No .deb packages found in $POOL_DIR"
    echo "Please build packages first using packaging/deb/build-all-deb.sh"
    exit 1
fi

echo "Found $PACKAGE_COUNT package(s) in pool"
echo ""

# Create dists directory structure
mkdir -p "$BINARY_DIR"

# Generate Packages file
echo "Generating Packages file..."
cd "$POOL_DIR"
dpkg-scanpackages . /dev/null 2>/dev/null | gzip -9c > "$BINARY_DIR/Packages.gz"
echo "Created: $BINARY_DIR/Packages.gz"

# Generate uncompressed Packages file (some tools prefer it)
dpkg-scanpackages . /dev/null 2>/dev/null > "$BINARY_DIR/Packages"
echo "Created: $BINARY_DIR/Packages"

# Generate Release file using apt-ftparchive
echo "Generating Release file..."
cd "$REPO_DEB_DIR"

# Create apt-ftparchive configuration if it doesn't exist
APT_CONF=$(mktemp)
cat > "$APT_CONF" <<EOF
APT::FTPArchive::Release::Origin "NeuronDB";
APT::FTPArchive::Release::Label "NeuronDB Repository";
APT::FTPArchive::Release::Suite "stable";
APT::FTPArchive::Release::Codename "stable";
APT::FTPArchive::Release::Architectures "amd64";
APT::FTPArchive::Release::Components "main";
APT::FTPArchive::Release::Description "NeuronDB Package Repository";
EOF

# Generate Release file
apt-ftparchive -c "$APT_CONF" release "$DISTS_DIR" > "$DISTS_DIR/Release"
rm -f "$APT_CONF"
echo "Created: $DISTS_DIR/Release"

# Sign Release file if GPG key is provided
if [ -n "$GPG_KEY_ID" ]; then
    echo "Signing Release file with GPG key: $GPG_KEY_ID"
    
    # Check if GPG key exists
    if ! gpg --list-secret-keys --keyid-format LONG "$GPG_KEY_ID" &>/dev/null; then
        echo "Warning: GPG key '$GPG_KEY_ID' not found in keyring"
        echo "Skipping GPG signing. Repository will be unsigned."
    else
        # Create detached signature (Release.gpg)
        gpg --default-key "$GPG_KEY_ID" --armor --detach-sign --output "$DISTS_DIR/Release.gpg" "$DISTS_DIR/Release"
        echo "Created: $DISTS_DIR/Release.gpg"
        
        # Create inline signature (InRelease)
        gpg --default-key "$GPG_KEY_ID" --clearsign --output "$DISTS_DIR/InRelease" "$DISTS_DIR/Release"
        echo "Created: $DISTS_DIR/InRelease"
    fi
else
    echo "No GPG key provided. Skipping signing."
    echo "Repository will be unsigned. Users will need to disable GPG checking."
fi

echo ""
echo "=========================================="
echo "DEB repository metadata generated successfully!"
echo "=========================================="
echo ""
echo "Repository structure:"
echo "  Packages: $BINARY_DIR/Packages.gz"
echo "  Release: $DISTS_DIR/Release"
if [ -n "$GPG_KEY_ID" ]; then
    echo "  Release.gpg: $DISTS_DIR/Release.gpg"
    echo "  InRelease: $DISTS_DIR/InRelease"
fi
echo ""

