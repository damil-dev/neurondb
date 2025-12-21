#!/bin/bash
set -e

# Master script to build packages and generate repository
# Usage: ./publish.sh [VERSION] [GPG_KEY_ID]
#   VERSION: Package version (default: from build-config.json)
#   GPG_KEY_ID: GPG key ID for signing (optional)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PACKAGING_DIR/.." && pwd)"

VERSION="${1:-}"
GPG_KEY_ID="${2:-}"

echo "=========================================="
echo "NeuronDB Package Repository Publisher"
echo "=========================================="
echo ""

# Load configuration if available
if [ -f "$PACKAGING_DIR/config-loader.sh" ]; then
    source "$PACKAGING_DIR/config-loader.sh"
    if [ -z "$VERSION" ]; then
        VERSION="${PACKAGING_VERSION:-1.0.0.beta}"
    fi
else
    if [ -z "$VERSION" ]; then
        VERSION="1.0.0.beta"
    fi
fi

echo "Version: $VERSION"
if [ -n "$GPG_KEY_ID" ]; then
    echo "GPG Key: $GPG_KEY_ID"
else
    echo "GPG Key: Not provided (packages will be unsigned)"
fi
echo ""

# Ask for confirmation
read -p "Continue with package build and repository generation? (y/n) [y]: " CONFIRM
CONFIRM=${CONFIRM:-y}
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "=========================================="
echo "Step 1: Building DEB packages"
echo "=========================================="
echo ""

cd "$PACKAGING_DIR/deb"
./build-all-deb.sh "$VERSION"

if [ $? -ne 0 ]; then
    echo "Error: DEB package build failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Building RPM packages"
echo "=========================================="
echo ""

cd "$PACKAGING_DIR/rpm"
./build-all-rpm.sh "$VERSION"

if [ $? -ne 0 ]; then
    echo "Error: RPM package build failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 3: Generating DEB repository metadata"
echo "=========================================="
echo ""

cd "$SCRIPT_DIR"
if [ -n "$GPG_KEY_ID" ]; then
    ./generate-deb-repo.sh "$GPG_KEY_ID"
else
    ./generate-deb-repo.sh
fi

if [ $? -ne 0 ]; then
    echo "Error: DEB repository generation failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 4: Generating RPM repository metadata"
echo "=========================================="
echo ""

cd "$SCRIPT_DIR"
if [ -n "$GPG_KEY_ID" ]; then
    ./generate-rpm-repo.sh "$GPG_KEY_ID"
else
    ./generate-rpm-repo.sh
fi

if [ $? -ne 0 ]; then
    echo "Error: RPM repository generation failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Package Repository Published Successfully!"
echo "=========================================="
echo ""
echo "Repository structure is ready in: $REPO_ROOT/repo/"
echo ""
echo "Next steps:"
echo "1. Commit and push the repo/ directory to GitHub main branch"
echo "2. Enable GitHub Pages in repository settings"
echo "3. Configure GitHub Pages to serve from '/repo' folder on main branch"
echo ""
echo "Repository URL will be:"
echo "  https://USERNAME.github.io/neurondb/repo/"
echo ""

