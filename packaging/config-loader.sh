#!/bin/bash
# Configuration loader for packaging scripts
# Loads build-config.json and provides helper functions

CONFIG_FILE="${PACKAGING_CONFIG:-$(cd "$(dirname "$0")" && pwd)/build-config.json}"

# Default values
DEFAULT_VERSION="1.0.0.beta"
DEFAULT_ARCH="amd64"
DEFAULT_GPU_BACKENDS="none"
DEFAULT_COMPUTE_MODE="cpu"
DEFAULT_PG_VERSIONS=("16" "17" "18")

# Load config if it exists
if [ -f "$CONFIG_FILE" ]; then
    # Use jq if available, otherwise use basic parsing
    if command -v jq &> /dev/null; then
        VERSION=$(jq -r '.version // "'"$DEFAULT_VERSION"'"' "$CONFIG_FILE")
        ARCH=$(jq -r '.architecture // "'"$DEFAULT_ARCH"'"' "$CONFIG_FILE")
        GPU_BACKENDS=$(jq -r '.neurondb.gpu_backends // "'"$DEFAULT_GPU_BACKENDS"'"' "$CONFIG_FILE")
        COMPUTE_MODE=$(jq -r '.neurondb.compute_mode // "'"$DEFAULT_COMPUTE_MODE"'"' "$CONFIG_FILE")
        
        # Read PostgreSQL versions array
        PG_VERSIONS_JSON=$(jq -r '.neurondb.postgresql_versions // ["16","17","18"] | .[]' "$CONFIG_FILE")
        PG_VERSIONS=()
        while IFS= read -r version; do
            PG_VERSIONS+=("$version")
        done <<< "$PG_VERSIONS_JSON"
        
        # CUDA settings
        CUDA_ENABLED=$(jq -r '.neurondb.cuda.enabled // false' "$CONFIG_FILE")
        CUDA_PATH=$(jq -r '.neurondb.cuda.path // ""' "$CONFIG_FILE")
        CUDA_VERSION=$(jq -r '.neurondb.cuda.version // ""' "$CONFIG_FILE")
        
        # ROCm settings
        ROCM_ENABLED=$(jq -r '.neurondb.rocm.enabled // false' "$CONFIG_FILE")
        ROCM_PATH=$(jq -r '.neurondb.rocm.path // ""' "$CONFIG_FILE")
        ROCM_VERSION=$(jq -r '.neurondb.rocm.version // ""' "$CONFIG_FILE")
        
        # Metal settings
        METAL_ENABLED=$(jq -r '.neurondb.metal.enabled // false' "$CONFIG_FILE")
        
        # Build args
        CXXFLAGS=$(jq -r '.neurondb.build_args.CXXFLAGS // ""' "$CONFIG_FILE")
        CFLAGS=$(jq -r '.neurondb.build_args.CFLAGS // ""' "$CONFIG_FILE")
        LDFLAGS=$(jq -r '.neurondb.build_args.LDFLAGS // ""' "$CONFIG_FILE")
    else
        # Fallback: use defaults if jq not available
        echo "Warning: jq not found, using default configuration values"
        VERSION="$DEFAULT_VERSION"
        ARCH="$DEFAULT_ARCH"
        GPU_BACKENDS="$DEFAULT_GPU_BACKENDS"
        COMPUTE_MODE="$DEFAULT_COMPUTE_MODE"
        PG_VERSIONS=("${DEFAULT_PG_VERSIONS[@]}")
        CUDA_ENABLED=false
        CUDA_PATH=""
        CUDA_VERSION=""
        ROCM_ENABLED=false
        ROCM_PATH=""
        ROCM_VERSION=""
        METAL_ENABLED=false
        CXXFLAGS=""
        CFLAGS=""
        LDFLAGS=""
    fi
else
    # Use defaults if config file doesn't exist
    VERSION="$DEFAULT_VERSION"
    ARCH="$DEFAULT_ARCH"
    GPU_BACKENDS="$DEFAULT_GPU_BACKENDS"
    COMPUTE_MODE="$DEFAULT_COMPUTE_MODE"
    PG_VERSIONS=("${DEFAULT_PG_VERSIONS[@]}")
    CUDA_ENABLED=false
    CUDA_PATH=""
    CUDA_VERSION=""
    ROCM_ENABLED=false
    ROCM_PATH=""
    ROCM_VERSION=""
    METAL_ENABLED=false
    CXXFLAGS=""
    CFLAGS=""
    LDFLAGS=""
fi

# Export variables for use in build scripts
export PACKAGING_VERSION="$VERSION"
export PACKAGING_ARCH="$ARCH"
export PACKAGING_GPU_BACKENDS="$GPU_BACKENDS"
export PACKAGING_COMPUTE_MODE="$COMPUTE_MODE"
export PACKAGING_PG_VERSIONS="${PG_VERSIONS[*]}"
export PACKAGING_CUDA_ENABLED="$CUDA_ENABLED"
export PACKAGING_CUDA_PATH="$CUDA_PATH"
export PACKAGING_CUDA_VERSION="$CUDA_VERSION"
export PACKAGING_ROCM_ENABLED="$ROCM_ENABLED"
export PACKAGING_ROCM_PATH="$ROCM_PATH"
export PACKAGING_ROCM_VERSION="$ROCM_VERSION"
export PACKAGING_METAL_ENABLED="$METAL_ENABLED"
export PACKAGING_CXXFLAGS="$CXXFLAGS"
export PACKAGING_CFLAGS="$CFLAGS"
export PACKAGING_LDFLAGS="$LDFLAGS"

# Function to get build environment for NeuronDB
get_neurondb_build_env() {
    local env_vars=""
    
    # GPU backends
    if [ -n "$PACKAGING_GPU_BACKENDS" ] && [ "$PACKAGING_GPU_BACKENDS" != "none" ]; then
        env_vars="GPU_BACKENDS=$PACKAGING_GPU_BACKENDS"
    else
        env_vars="GPU_BACKENDS=none"
    fi
    
    # CUDA
    if [ "$PACKAGING_CUDA_ENABLED" = "true" ] && [ -n "$PACKAGING_CUDA_PATH" ]; then
        env_vars="$env_vars CUDA_PATH=$PACKAGING_CUDA_PATH"
    fi
    
    # ROCm
    if [ "$PACKAGING_ROCM_ENABLED" = "true" ] && [ -n "$PACKAGING_ROCM_PATH" ]; then
        env_vars="$env_vars ROCM_PATH=$PACKAGING_ROCM_PATH"
    fi
    
    # Metal
    if [ "$PACKAGING_METAL_ENABLED" = "true" ]; then
        env_vars="$env_vars METAL_ENABLED=1"
    fi
    
    # Compiler flags
    if [ -n "$PACKAGING_CXXFLAGS" ]; then
        env_vars="$env_vars CXXFLAGS=\"$PACKAGING_CXXFLAGS\""
    fi
    if [ -n "$PACKAGING_CFLAGS" ]; then
        env_vars="$env_vars CFLAGS=\"$PACKAGING_CFLAGS\""
    fi
    if [ -n "$PACKAGING_LDFLAGS" ]; then
        env_vars="$env_vars LDFLAGS=\"$PACKAGING_LDFLAGS\""
    fi
    
    echo "$env_vars"
}

# Function to print current configuration
print_config() {
    echo "Build Configuration:"
    echo "  Version: $PACKAGING_VERSION"
    echo "  Architecture: $PACKAGING_ARCH"
    echo "  GPU Backends: $PACKAGING_GPU_BACKENDS"
    echo "  Compute Mode: $PACKAGING_COMPUTE_MODE"
    echo "  PostgreSQL Versions: ${PACKAGING_PG_VERSIONS}"
    echo "  CUDA Enabled: $PACKAGING_CUDA_ENABLED"
    if [ "$PACKAGING_CUDA_ENABLED" = "true" ]; then
        echo "    CUDA Path: $PACKAGING_CUDA_PATH"
        echo "    CUDA Version: $PACKAGING_CUDA_VERSION"
    fi
    echo "  ROCm Enabled: $PACKAGING_ROCM_ENABLED"
    if [ "$PACKAGING_ROCM_ENABLED" = "true" ]; then
        echo "    ROCm Path: $PACKAGING_ROCM_PATH"
        echo "    ROCm Version: $PACKAGING_ROCM_VERSION"
    fi
    echo "  Metal Enabled: $PACKAGING_METAL_ENABLED"
    echo ""
}

