#!/bin/bash
set -e

# Build script for NeuronDB DEB package
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

# Try to find NeuronDB source - check if we're in Docker (use /tmp) or local build
if [ -d "/tmp/NeuronDB" ]; then
    REPO_ROOT="/tmp"
    NEURONDB_SRC="/tmp/NeuronDB"
else
    REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
    NEURONDB_SRC="$REPO_ROOT/NeuronDB"
fi
PACKAGE_DIR="$SCRIPT_DIR/package"
VERSION="${VERSION:-${PACKAGING_VERSION:-1.0.0.beta}}"
ARCH="${ARCH:-${PACKAGING_ARCH:-amd64}}"

echo "Building NeuronDB DEB package version $VERSION"

# Clean previous build
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/DEBIAN"

# Use configured PostgreSQL versions or detect
if [ -n "${PACKAGING_PG_VERSIONS:-}" ]; then
    # Use versions from config
    PG_VERSIONS="${PACKAGING_PG_VERSIONS}"
    echo "Using PostgreSQL versions from config: $PG_VERSIONS"
else
    # Detect PostgreSQL versions
    PG_VERSIONS=$(ls -1 /usr/lib/postgresql/ 2>/dev/null | grep -E '^[0-9]+$' | sort -n || true)
    
    if [ -z "$PG_VERSIONS" ]; then
        echo "Error: No PostgreSQL versions found in /usr/lib/postgresql/"
        echo "Please install PostgreSQL 16, 17, or 18 development packages"
        echo "Or configure versions in build-config.json"
        exit 1
    fi
    
    echo "Detected PostgreSQL versions: $(echo $PG_VERSIONS | tr '\n' ' ')"
fi

# Build for each PostgreSQL version
cd "$NEURONDB_SRC"

for PG_VERSION in $PG_VERSIONS; do
    PG_CONFIG="/usr/lib/postgresql/$PG_VERSION/bin/pg_config"
    
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
    
    # CUDA
    if [ "${PACKAGING_CUDA_ENABLED:-false}" = "true" ] && [ -n "${PACKAGING_CUDA_PATH:-}" ]; then
        BUILD_ENV="$BUILD_ENV CUDA_PATH=${PACKAGING_CUDA_PATH}"
    fi
    
    # ROCm
    if [ "${PACKAGING_ROCM_ENABLED:-false}" = "true" ] && [ -n "${PACKAGING_ROCM_PATH:-}" ]; then
        BUILD_ENV="$BUILD_ENV ROCM_PATH=${PACKAGING_ROCM_PATH}"
    fi
    
    # Metal
    if [ "${PACKAGING_METAL_ENABLED:-false}" = "true" ]; then
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
    if [ "${PACKAGING_CUDA_ENABLED:-false}" = "true" ]; then
        echo "  CUDA: Enabled (${PACKAGING_CUDA_PATH:-auto})"
    fi
    if [ "${PACKAGING_ROCM_ENABLED:-false}" = "true" ]; then
        echo "  ROCm: Enabled (${PACKAGING_ROCM_PATH:-auto})"
    fi
    if [ "${PACKAGING_METAL_ENABLED:-false}" = "true" ]; then
        echo "  Metal: Enabled"
    fi
    
    # Add ML library linking flags if libraries are installed
    # Check for XGBoost
    if find /usr/local/lib -name "libxgboost*.so*" -o -name "libxgboost*.a" 2>/dev/null | grep -q .; then
        BUILD_ENV="$BUILD_ENV SHLIB_LINK=\"\$(SHLIB_LINK) -L/usr/local/lib -lxgboost -Wl,-rpath,/usr/local/lib\""
    fi
    # Check for LightGBM (can be lib_lightgbm or liblightgbm)
    if find /usr/local/lib -name "*lightgbm*.so*" -o -name "*lightgbm*.a" 2>/dev/null | grep -q .; then
        # Try to find the actual library name
        LIGHTGBM_LIB=$(find /usr/local/lib -name "*lightgbm*.so*" 2>/dev/null | head -1 | xargs basename -s .so 2>/dev/null | sed 's/\.so.*//' || echo "lightgbm")
        BUILD_ENV="$BUILD_ENV SHLIB_LINK=\"\$(SHLIB_LINK) -L/usr/local/lib -l${LIGHTGBM_LIB#lib} -Wl,-rpath,/usr/local/lib\""
    fi
    # Check for CatBoost
    if find /usr/local/lib -name "*catboost*.so*" -o -name "*catboost*.a" 2>/dev/null | grep -q .; then
        BUILD_ENV="$BUILD_ENV SHLIB_LINK=\"\$(SHLIB_LINK) -L/usr/local/lib -lcatboostmodel -Wl,-rpath,/usr/local/lib\""
    fi
    eval "$BUILD_ENV make PG_CONFIG=\"$PG_CONFIG\""
    
    # Install to package directory
    PG_LIBDIR=$($PG_CONFIG --pkglibdir)
    PG_SHAREDIR=$($PG_CONFIG --sharedir)
    
    mkdir -p "$PACKAGE_DIR$PG_LIBDIR"
    mkdir -p "$PACKAGE_DIR$PG_SHAREDIR/extension"
    
    # Copy shared library
    if [ -f "neurondb.so" ]; then
        cp neurondb.so "$PACKAGE_DIR$PG_LIBDIR/"
    elif [ -f "neurondb.dylib" ]; then
        cp neurondb.dylib "$PACKAGE_DIR$PG_LIBDIR/neurondb.so"
    else
        echo "Error: neurondb.so not found after build"
        exit 1
    fi
    
    # Copy SQL files and control file
    cp neurondb--1.0.sql "$PACKAGE_DIR$PG_SHAREDIR/extension/" 2>/dev/null || true
    cp neurondb.control "$PACKAGE_DIR$PG_SHAREDIR/extension/" 2>/dev/null || true
    
    # Clean for next version
    make clean 2>/dev/null || true
done

# Copy DEBIAN control files
cp "$SCRIPT_DIR/DEBIAN/control" "$PACKAGE_DIR/DEBIAN/"
cp "$SCRIPT_DIR/DEBIAN/postinst" "$PACKAGE_DIR/DEBIAN/"
cp "$SCRIPT_DIR/DEBIAN/prerm" "$PACKAGE_DIR/DEBIAN/"

# Make scripts executable
chmod +x "$PACKAGE_DIR/DEBIAN/postinst"
chmod +x "$PACKAGE_DIR/DEBIAN/prerm"

# Update version in control file
sed -i "s/Version: .*/Version: $VERSION/" "$PACKAGE_DIR/DEBIAN/control"

# Add GPU dependencies if GPU backend is enabled
if [ -f "$PACKAGING_DIR/config-loader.sh" ]; then
    GPU_DEPS=$(get_gpu_deps_deb)
    if [ -n "$GPU_DEPS" ]; then
        # Add GPU dependencies to the Depends line
        # This appends the GPU deps after the existing dependencies
        sed -i "s/^Depends: \(.*\)$/Depends: \1, $GPU_DEPS/" "$PACKAGE_DIR/DEBIAN/control"
        echo "Added GPU dependencies: $GPU_DEPS"
    fi
fi

# Build package
echo "Building DEB package..."
cd "$SCRIPT_DIR"
fakeroot dpkg-deb --build package neurondb_${VERSION}_${ARCH}.deb

echo "Package built: neurondb_${VERSION}_${ARCH}.deb"

