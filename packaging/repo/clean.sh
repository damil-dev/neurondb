#!/bin/bash
set -e

# Clean repository directory while preserving structure
# Usage: ./clean.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PACKAGING_DIR/.." && pwd)"

REPO_DEB_DIR="$REPO_ROOT/repo/deb"
REPO_RPM_DIR="$REPO_ROOT/repo/rpm"

echo "Cleaning repository directory..."

# Remove packages but keep directory structure
if [ -d "$REPO_DEB_DIR/pool" ]; then
    find "$REPO_DEB_DIR/pool" -name "*.deb" -type f -delete
    echo "Removed DEB packages from pool/"
fi

if [ -d "$REPO_DEB_DIR/dists" ]; then
    rm -rf "$REPO_DEB_DIR/dists/stable/main/binary-amd64/Packages"*
    rm -f "$REPO_DEB_DIR/dists/stable/Release"*
    rm -f "$REPO_DEB_DIR/dists/stable/InRelease"
    echo "Removed DEB repository metadata"
fi

if [ -d "$REPO_RPM_DIR" ]; then
    find "$REPO_RPM_DIR" -name "*.rpm" -type f -delete
    find "$REPO_RPM_DIR" -type d -name "repodata" -exec rm -rf {} + 2>/dev/null || true
    echo "Removed RPM packages and repodata/"
fi

echo "Repository cleaned successfully!"
echo "Directory structure preserved."

