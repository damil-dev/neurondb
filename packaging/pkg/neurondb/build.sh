#!/bin/bash
set -e

# Build script for NeuronDB macOS .pkg package
# Usage: ./build.sh [PG_CONFIG_PATH]
# Configuration: Uses build-config.json in packaging/ directory

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGING_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load configuration
if [ -f "$PACKAGING_DIR/config-loader.sh" ]; then
    source "$PACKAGING_DIR/config-loader.sh"
    print_config
else
    echo "Warning: config-loader.sh not found, using defaults"
    VERSION="${VERSION:-1.0.0.beta}"
    GPU_BACKENDS="${GPU_BACKENDS:-none}"
    PG_VERSIONS=("16" "17" "18")
fi

# Try to find NeuronDB source
if [ -d "/tmp/NeuronDB" ]; then
    REPO_ROOT="/tmp"
    NEURONDB_SRC="/tmp/NeuronDB"
else
    REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
    NEURONDB_SRC="$REPO_ROOT/NeuronDB"
fi

PACKAGE_DIR="$SCRIPT_DIR/package"
VERSION="${VERSION:-${PACKAGING_VERSION:-1.0.0.beta}}"
ARCH="${ARCH:-${PACKAGING_ARCH:-arm64}}"

# Detect macOS architecture if not specified
if [ -z "$ARCH" ] || [ "$ARCH" = "amd64" ]; then
    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ]; then
        ARCH="x86_64"
    elif [ "$ARCH" = "arm64" ]; then
        ARCH="arm64"
    fi
fi

echo "Building NeuronDB macOS package version $VERSION for $ARCH"

# Check for required tools
if ! command -v pkgbuild &> /dev/null; then
    echo "Error: pkgbuild not found. This script requires macOS and Xcode Command Line Tools."
    exit 1
fi

# Clean previous build
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

# Use configured PostgreSQL versions or detect
if [ -n "${PACKAGING_PG_VERSIONS:-}" ]; then
    PG_VERSIONS="${PACKAGING_PG_VERSIONS}"
    echo "Using PostgreSQL versions from config: $PG_VERSIONS"
else
    # Detect PostgreSQL versions from Homebrew
    PG_VERSIONS=()
    for pg_ver in 16 17 18; do
        PG_PREFIX=""
        if [ "$ARCH" = "arm64" ]; then
            PG_PREFIX="/opt/homebrew/opt/postgresql@${pg_ver}"
        else
            PG_PREFIX="/usr/local/opt/postgresql@${pg_ver}"
        fi
        
        if [ -f "${PG_PREFIX}/bin/pg_config" ]; then
            PG_VERSIONS+=("$pg_ver")
        fi
    done
    
    if [ ${#PG_VERSIONS[@]} -eq 0 ]; then
        echo "Error: No PostgreSQL versions found"
        echo "Please install PostgreSQL 16, 17, or 18 via Homebrew"
        echo "Example: brew install postgresql@17"
        exit 1
    fi
    
    echo "Detected PostgreSQL versions: ${PG_VERSIONS[*]}"
fi

# Build for each PostgreSQL version
cd "$NEURONDB_SRC"

for PG_VERSION in "${PG_VERSIONS[@]}"; do
    # Determine PostgreSQL prefix based on architecture
    if [ "$ARCH" = "arm64" ]; then
        PG_PREFIX="/opt/homebrew/opt/postgresql@${PG_VERSION}"
    else
        PG_PREFIX="/usr/local/opt/postgresql@${PG_VERSION}"
    fi
    
    PG_CONFIG="${PG_PREFIX}/bin/pg_config"
    
    if [ ! -f "$PG_CONFIG" ]; then
        echo "Warning: pg_config not found for PostgreSQL $PG_VERSION, skipping"
        continue
    fi
    
    echo "Building for PostgreSQL $PG_VERSION..."
    
    # Build environment variables from config
    BUILD_ENV=""
    
    # GPU backends
    if [ -n "${PACKAGING_GPU_BACKENDS:-}" ] && [ "${PACKAGING_GPU_BACKENDS}" != "none" ]; then
        BUILD_ENV="GPU_BACKENDS=${PACKAGING_GPU_BACKENDS}"
    else
        BUILD_ENV="GPU_BACKENDS=none"
    fi
    
    # Metal (macOS-specific)
    if [ "${PACKAGING_METAL_ENABLED:-false}" = "true" ] || [ "${PACKAGING_GPU_BACKENDS:-}" = "metal" ]; then
        BUILD_ENV="$BUILD_ENV METAL_ENABLED=1"
    fi
    
    # Compiler flags
    if [ -n "${PACKAGING_CXXFLAGS:-}" ]; then
        BUILD_ENV="$BUILD_ENV CXXFLAGS=\"${PACKAGING_CXXFLAGS}\""
    fi
    if [ -n "${PACKAGING_CFLAGS:-}" ]; then
        BUILD_ENV="$BUILD_ENV CFLAGS=\"${PACKAGING_CFLAGS}\""
    fi
    if [ -n "${PACKAGING_LDFLAGS:-}" ]; then
        BUILD_ENV="$BUILD_ENV LDFLAGS=\"${PACKAGING_LDFLAGS}\""
    fi
    
    # Clean previous build
    eval "$BUILD_ENV make clean" 2>/dev/null || true
    
    # Build extension with configured options
    echo "  GPU Backends: ${PACKAGING_GPU_BACKENDS:-none}"
    echo "  Compute Mode: ${PACKAGING_COMPUTE_MODE:-cpu}"
    if [ "${PACKAGING_METAL_ENABLED:-false}" = "true" ]; then
        echo "  Metal: Enabled"
    fi
    
    eval "$BUILD_ENV make PG_CONFIG=\"$PG_CONFIG\""
    
    # Get installation paths
    PG_LIBDIR=$($PG_CONFIG --pkglibdir)
    PG_SHAREDIR=$($PG_CONFIG --sharedir)
    
    # Create package directory structure
    mkdir -p "$PACKAGE_DIR$PG_LIBDIR"
    mkdir -p "$PACKAGE_DIR$PG_SHAREDIR/extension"
    
    # Copy shared library (macOS uses .dylib)
    if [ -f "neurondb.dylib" ]; then
        cp neurondb.dylib "$PACKAGE_DIR$PG_LIBDIR/"
    elif [ -f "neurondb.so" ]; then
        cp neurondb.so "$PACKAGE_DIR$PG_LIBDIR/neurondb.dylib"
    else
        echo "Error: neurondb.dylib not found after build"
        exit 1
    fi
    
    # Copy SQL files and control file
    cp neurondb--1.0.sql "$PACKAGE_DIR$PG_SHAREDIR/extension/" 2>/dev/null || true
    cp neurondb.control "$PACKAGE_DIR$PG_SHAREDIR/extension/" 2>/dev/null || true
    
    # Clean for next version
    make clean 2>/dev/null || true
done

# Copy scripts
mkdir -p "$PACKAGE_DIR/scripts"
cp "$SCRIPT_DIR/scripts/postinstall" "$PACKAGE_DIR/scripts/" 2>/dev/null || true

# Make scripts executable
chmod +x "$PACKAGE_DIR/scripts/"* 2>/dev/null || true

# Build component package
echo "Building macOS package..."
COMPONENT_PKG="$SCRIPT_DIR/neurondb-${VERSION}-${ARCH}.pkg"

pkgbuild \
    --root "$PACKAGE_DIR" \
    --identifier com.neurondb.neurondb \
    --version "$VERSION" \
    --install-location "/" \
    --scripts "$PACKAGE_DIR/scripts" \
    "$COMPONENT_PKG"

echo "Component package built: $COMPONENT_PKG"

# Create distribution package (optional, for nicer installer)
DIST_XML="$SCRIPT_DIR/distribution.xml"
cat > "$DIST_XML" << EOF
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="1">
    <title>NeuronDB PostgreSQL Extension ${VERSION}</title>
    <organization>com.neurondb</organization>
    <domains enable_localSystem="true"/>
    <options customize="never" require-scripts="false" rootVolumeOnly="true"/>
    <pkg-ref id="com.neurondb.neurondb"/>
    <choices-outline>
        <line choice="default">
            <line choice="com.neurondb.neurondb"/>
        </line>
    </choices-outline>
    <choice id="default"/>
    <choice id="com.neurondb.neurondb" visible="false">
        <pkg-ref id="com.neurondb.neurondb"/>
    </choice>
    <pkg-ref id="com.neurondb.neurondb" version="${VERSION}" onConclusion="none">neurondb-${VERSION}-${ARCH}.pkg</pkg-ref>
</installer-gui-script>
EOF

DIST_PKG="$SCRIPT_DIR/neurondb-${VERSION}-${ARCH}-installer.pkg"
if command -v productbuild &> /dev/null; then
    productbuild \
        --distribution "$DIST_XML" \
        --package-path "$SCRIPT_DIR" \
        --resources "$SCRIPT_DIR/Resources" \
        "$DIST_PKG"
    
    echo "Distribution package built: $DIST_PKG"
    echo ""
    echo "Install with: sudo installer -pkg $DIST_PKG -target /"
else
    echo "Note: productbuild not found, skipping distribution package"
    echo "Component package can be installed with: sudo installer -pkg $COMPONENT_PKG -target /"
fi

