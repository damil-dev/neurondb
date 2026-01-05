#!/bin/bash
# Release script for NeuronDB ecosystem
# Usage: ./scripts/release.sh <version> [--dry-run]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VERSION="${1:-}"
DRY_RUN="${2:-}"

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version> [--dry-run]"
    echo "Example: $0 1.0.0"
    exit 1
fi

# Validate version format (semver)
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?$ ]]; then
    echo "Error: Invalid version format. Use semver (e.g., 1.0.0)"
    exit 1
fi

echo "Releasing NeuronDB ecosystem version $VERSION"
if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "DRY RUN MODE - No changes will be made"
fi

# Update version in all components
update_version() {
    local component=$1
    local file=$2
    
    if [ -f "$file" ]; then
        if [ "$DRY_RUN" != "--dry-run" ]; then
            sed -i.bak "s/version.*=.*\".*\"/version = \"$VERSION\"/g" "$file" || true
            sed -i.bak "s/VERSION.*=.*\".*\"/VERSION=\"$VERSION\"/g" "$file" || true
            sed -i.bak "s/\"version\":.*\".*\"/\"version\": \"$VERSION\"/g" "$file" || true
            rm -f "$file.bak"
        else
            echo "Would update: $file"
        fi
    fi
}

# Update version files
echo "Updating version files..."
update_version "root" "$REPO_ROOT/env.example"
update_version "root" "$REPO_ROOT/package.json" 2>/dev/null || true

# Update component versions
update_version "NeuronDB" "$REPO_ROOT/NeuronDB/neurondb.control"
update_version "NeuronAgent" "$REPO_ROOT/NeuronAgent/go.mod"
update_version "NeuronDesktop" "$REPO_ROOT/NeuronDesktop/frontend/package.json"
update_version "NeuronMCP" "$REPO_ROOT/NeuronMCP/go.mod"

# Generate SBOMs
generate_sbom() {
    local component=$1
    local image=$2
    
    echo "Generating SBOM for $component..."
    
    if command -v syft >/dev/null 2>&1; then
        if [ "$DRY_RUN" != "--dry-run" ]; then
            syft "$image" -o spdx-json > "$REPO_ROOT/releases/$VERSION/${component}.sbom.json"
        else
            echo "Would generate SBOM: releases/$VERSION/${component}.sbom.json"
        fi
    else
        echo "Warning: syft not found, skipping SBOM generation"
    fi
}

# Create release directory
if [ "$DRY_RUN" != "--dry-run" ]; then
    mkdir -p "$REPO_ROOT/releases/$VERSION"
fi

# Build and tag images
build_images() {
    echo "Building Docker images..."
    
    if [ "$DRY_RUN" = "--dry-run" ]; then
        echo "Would build images with tag: $VERSION"
        return
    fi
    
    # Build NeuronDB
    docker build -t "ghcr.io/neurondb/neurondb-postgres:$VERSION" \
        -t "ghcr.io/neurondb/neurondb-postgres:latest" \
        -f "$REPO_ROOT/NeuronDB/docker/Dockerfile" \
        "$REPO_ROOT/NeuronDB"
    
    # Build NeuronAgent
    docker build -t "ghcr.io/neurondb/neuronagent:$VERSION" \
        -t "ghcr.io/neurondb/neuronagent:latest" \
        -f "$REPO_ROOT/NeuronAgent/docker/Dockerfile" \
        "$REPO_ROOT/NeuronAgent"
    
    # Build NeuronMCP
    docker build -t "ghcr.io/neurondb/neuronmcp:$VERSION" \
        -t "ghcr.io/neurondb/neuronmcp:latest" \
        -f "$REPO_ROOT/NeuronMCP/docker/Dockerfile" \
        "$REPO_ROOT/NeuronMCP"
    
    # Build NeuronDesktop
    docker build -t "ghcr.io/neurondb/neurondesktop-api:$VERSION" \
        -t "ghcr.io/neurondb/neurondesktop-api:latest" \
        -f "$REPO_ROOT/NeuronDesktop/api/Dockerfile" \
        "$REPO_ROOT/NeuronDesktop/api"
    
    docker build -t "ghcr.io/neurondb/neurondesktop-frontend:$VERSION" \
        -t "ghcr.io/neurondb/neurondesktop-frontend:latest" \
        -f "$REPO_ROOT/NeuronDesktop/frontend/Dockerfile" \
        "$REPO_ROOT/NeuronDesktop/frontend"
}

# Sign images (if cosign is available)
sign_images() {
    if command -v cosign >/dev/null 2>&1; then
        echo "Signing Docker images..."
        
        if [ "$DRY_RUN" = "--dry-run" ]; then
            echo "Would sign images with cosign"
            return
        fi
        
        # Sign images (requires COSIGN_PASSWORD and key)
        if [ -n "$COSIGN_PASSWORD" ]; then
            cosign sign --key "$COSIGN_KEY" "ghcr.io/neurondb/neurondb-postgres:$VERSION"
            cosign sign --key "$COSIGN_KEY" "ghcr.io/neurondb/neuronagent:$VERSION"
            cosign sign --key "$COSIGN_KEY" "ghcr.io/neurondb/neuronmcp:$VERSION"
            cosign sign --key "$COSIGN_KEY" "ghcr.io/neurondb/neurondesktop-api:$VERSION"
            cosign sign --key "$COSIGN_KEY" "ghcr.io/neurondb/neurondesktop-frontend:$VERSION"
        else
            echo "Warning: COSIGN_PASSWORD not set, skipping image signing"
        fi
    else
        echo "Warning: cosign not found, skipping image signing"
    fi
}

# Create release manifest
create_manifest() {
    cat > "$REPO_ROOT/releases/$VERSION/manifest.json" <<EOF
{
  "version": "$VERSION",
  "released_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "components": {
    "neurondb": {
      "version": "$VERSION",
      "image": "ghcr.io/neurondb/neurondb-postgres:$VERSION"
    },
    "neuronagent": {
      "version": "$VERSION",
      "image": "ghcr.io/neurondb/neuronagent:$VERSION"
    },
    "neuronmcp": {
      "version": "$VERSION",
      "image": "ghcr.io/neurondb/neuronmcp:$VERSION"
    },
    "neurondesktop-api": {
      "version": "$VERSION",
      "image": "ghcr.io/neurondb/neurondesktop-api:$VERSION"
    },
    "neurondesktop-frontend": {
      "version": "$VERSION",
      "image": "ghcr.io/neurondb/neurondesktop-frontend:$VERSION"
    }
  },
  "compatibility": {
    "postgresql": ["16", "17", "18"],
    "go": ["1.21", "1.22", "1.23", "1.24"],
    "node": ["18", "20", "22"]
  }
}
EOF
}

# Main release process
if [ "$DRY_RUN" != "--dry-run" ]; then
    build_images
    generate_sbom "neurondb" "ghcr.io/neurondb/neurondb-postgres:$VERSION"
    generate_sbom "neuronagent" "ghcr.io/neurondb/neuronagent:$VERSION"
    generate_sbom "neuronmcp" "ghcr.io/neurondb/neuronmcp:$VERSION"
    sign_images
    create_manifest
else
    echo "DRY RUN - Would execute:"
    echo "  - Update version files"
    echo "  - Build Docker images"
    echo "  - Generate SBOMs"
    echo "  - Sign images"
    echo "  - Create release manifest"
fi

echo "Release process complete for version $VERSION"

