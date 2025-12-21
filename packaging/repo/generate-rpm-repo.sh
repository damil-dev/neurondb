#!/bin/bash
set -e

# Generate RPM repository metadata using createrepo
# Usage: ./generate-rpm-repo.sh [GPG_KEY_ID]
# If GPG_KEY_ID is not provided, signing will be skipped

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PACKAGING_DIR/.." && pwd)"

REPO_RPM_DIR="$REPO_ROOT/repo/rpm"
GPG_KEY_ID="${1:-}"

echo "=========================================="
echo "Generating RPM repository metadata"
echo "=========================================="
echo ""

# Check for createrepo or createrepo_c
CREATEREPO_CMD=""
if command -v createrepo_c &> /dev/null; then
    CREATEREPO_CMD="createrepo_c"
elif command -v createrepo &> /dev/null; then
    CREATEREPO_CMD="createrepo"
else
    echo "Error: createrepo or createrepo_c not found."
    echo "Please install createrepo package (RHEL/CentOS/Rocky: dnf install createrepo_c)"
    exit 1
fi

echo "Using: $CREATEREPO_CMD"
echo ""

# Check if rpm directory exists
if [ ! -d "$REPO_RPM_DIR" ]; then
    echo "Error: RPM directory not found: $REPO_RPM_DIR"
    echo "Please build packages first using packaging/rpm/build-all-rpm.sh"
    exit 1
fi

# Find all el*/x86_64 directories
REPO_DIRS=$(find "$REPO_RPM_DIR" -type d -path "*/el*/x86_64" 2>/dev/null | sort)

if [ -z "$REPO_DIRS" ]; then
    echo "Warning: No el*/x86_64 directories found in $REPO_RPM_DIR"
    echo "Please build packages first using packaging/rpm/build-all-rpm.sh"
    exit 1
fi

# Process each repository directory
for REPO_DIR in $REPO_DIRS; do
    EL_VERSION=$(echo "$REPO_DIR" | grep -oE 'el[0-9]+' | head -1)
    
    if [ -z "$EL_VERSION" ]; then
        echo "Warning: Could not extract EL version from path: $REPO_DIR"
        continue
    fi
    
    # Count RPM files
    RPM_COUNT=$(find "$REPO_DIR" -maxdepth 1 -name "*.rpm" 2>/dev/null | wc -l)
    
    if [ "$RPM_COUNT" -eq 0 ]; then
        echo "Skipping $EL_VERSION (no RPM files found)"
        continue
    fi
    
    echo "Processing $EL_VERSION ($RPM_COUNT package(s))..."
    echo "  Directory: $REPO_DIR"
    
    # Remove existing repodata if it exists
    if [ -d "$REPO_DIR/repodata" ]; then
        rm -rf "$REPO_DIR/repodata"
    fi
    
    # Generate repodata
    cd "$REPO_DIR"
    
    if [ -n "$GPG_KEY_ID" ]; then
        # Check if GPG key exists
        if ! gpg --list-secret-keys --keyid-format LONG "$GPG_KEY_ID" &>/dev/null; then
            echo "  Warning: GPG key '$GPG_KEY_ID' not found. Creating unsigned repository."
            $CREATEREPO_CMD --update .
        else
            # Create signed repository
            echo "  Signing with GPG key: $GPG_KEY_ID"
            $CREATEREPO_CMD --update --gpg-sign "$GPG_KEY_ID" .
        fi
    else
        # Create unsigned repository
        echo "  Creating unsigned repository (no GPG key provided)"
        $CREATEREPO_CMD --update .
    fi
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Repository metadata generated successfully"
    else
        echo "  ✗ Failed to generate repository metadata"
        exit 1
    fi
    
    echo ""
done

echo "=========================================="
echo "RPM repository metadata generated successfully!"
echo "=========================================="
echo ""
echo "Repository directories:"
for REPO_DIR in $REPO_DIRS; do
    EL_VERSION=$(echo "$REPO_DIR" | grep -oE 'el[0-9]+' | head -1)
    RPM_COUNT=$(find "$REPO_DIR" -maxdepth 1 -name "*.rpm" 2>/dev/null | wc -l)
    if [ "$RPM_COUNT" -gt 0 ]; then
        echo "  $EL_VERSION: $REPO_DIR ($RPM_COUNT package(s))"
    fi
done
echo ""

