#!/usr/bin/env bash

# NeuronMCP Run Script
# Installs dependencies from requirements.txt and runs the MCP server
# Compatible with macOS, Rocky Linux, Ubuntu, and other Linux distributions

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to script directory
cd "$SCRIPT_DIR"

# Colors for output (only if terminal supports it)
if [[ -t 1 ]] && [[ "${TERM:-}" != "dumb" ]]; then
    GREEN='\033[0;32m'
    BLUE='\033[0;34m'
    YELLOW='\033[1;33m'
    RED='\033[0;31m'
    NC='\033[0m'
else
    GREEN=''
    BLUE=''
    YELLOW=''
    RED=''
    NC=''
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}NeuronMCP Server Startup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to install Python dependencies
install_python_deps() {
    if [ ! -f "requirements.txt" ]; then
        echo -e "${YELLOW}Info: requirements.txt not found, skipping Python dependencies${NC}"
        return 0
    fi

    echo -e "${GREEN}✓ Found requirements.txt${NC}"
    echo -e "${BLUE}Installing Python dependencies...${NC}"
    
    # Try different pip installation methods
    local pip_cmd=""
    if command -v pip3 &> /dev/null; then
        pip_cmd="pip3"
    elif python3 -m pip --version &> /dev/null 2>&1; then
        pip_cmd="python3 -m pip"
    else
        echo -e "${YELLOW}Warning: pip not found, skipping Python dependencies${NC}"
        return 0
    fi

    # Try installation (without --user first, then with --user if needed)
    if $pip_cmd install -r requirements.txt --quiet --disable-pip-version-check 2>/dev/null; then
        echo -e "${GREEN}✓ Python dependencies installed${NC}"
    elif $pip_cmd install --user -r requirements.txt --quiet --disable-pip-version-check 2>/dev/null; then
        echo -e "${GREEN}✓ Python dependencies installed (user install)${NC}"
    else
        echo -e "${YELLOW}Warning: Python dependencies installation had issues (continuing anyway)${NC}"
    fi
}

# Install Python dependencies if Python is available
if command -v python3 &> /dev/null; then
    install_python_deps
else
    echo -e "${YELLOW}Info: python3 not found, skipping Python dependencies${NC}"
fi

# Check if Go is available (needed for building)
if ! command -v go &> /dev/null; then
    echo -e "${YELLOW}Warning: go is not installed, cannot build from source${NC}"
fi

# Try to find the binary
BINARY_PATH=""
if [ -f "${SCRIPT_DIR}/bin/neurondb-mcp" ]; then
    BINARY_PATH="${SCRIPT_DIR}/bin/neurondb-mcp"
elif [ -f "${SCRIPT_DIR}/bin/neuronmcp" ]; then
    BINARY_PATH="${SCRIPT_DIR}/bin/neuronmcp"
elif [ -f "${SCRIPT_DIR}/neurondb-mcp" ]; then
    BINARY_PATH="${SCRIPT_DIR}/neurondb-mcp"
elif [ -f "${SCRIPT_DIR}/neuronmcp" ]; then
    BINARY_PATH="${SCRIPT_DIR}/neuronmcp"
fi

# If binary doesn't exist, try to build it
if [ -z "$BINARY_PATH" ] || [ ! -f "$BINARY_PATH" ]; then
    if command -v go &> /dev/null && [ -f "Makefile" ]; then
        echo -e "${BLUE}Binary not found, building from source...${NC}"
        if make build 2>/dev/null; then
            if [ -f "${SCRIPT_DIR}/bin/neurondb-mcp" ]; then
                BINARY_PATH="${SCRIPT_DIR}/bin/neurondb-mcp"
            elif [ -f "${SCRIPT_DIR}/bin/neuronmcp" ]; then
                BINARY_PATH="${SCRIPT_DIR}/bin/neuronmcp"
            fi
        else
            echo -e "${YELLOW}Warning: Build failed, will try to use existing binary${NC}"
        fi
    fi
fi

# Check if binary exists
if [ -z "$BINARY_PATH" ] || [ ! -f "$BINARY_PATH" ]; then
    echo -e "${RED}Error: MCP server binary not found${NC}" >&2
    echo -e "${YELLOW}Please build the binary first with: make build${NC}" >&2
    echo -e "${YELLOW}Or ensure the binary is in one of these locations:${NC}" >&2
    echo -e "${YELLOW}  - bin/neurondb-mcp${NC}" >&2
    echo -e "${YELLOW}  - bin/neuronmcp${NC}" >&2
    echo -e "${YELLOW}  - neurondb-mcp${NC}" >&2
    echo -e "${YELLOW}  - neuronmcp${NC}" >&2
    exit 1
fi

# Make binary executable if it isn't
if [ ! -x "$BINARY_PATH" ]; then
    chmod +x "$BINARY_PATH"
fi

# Set default environment variables if not already set
export NEURONDB_HOST="${NEURONDB_HOST:-localhost}"
export NEURONDB_PORT="${NEURONDB_PORT:-5432}"
export NEURONDB_DATABASE="${NEURONDB_DATABASE:-neurondb}"
export NEURONDB_USER="${NEURONDB_USER:-pgedge}"
export NEURONDB_PASSWORD="${NEURONDB_PASSWORD:-}"

# Display configuration
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Starting NeuronMCP Server${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Binary: ${BINARY_PATH}"
echo -e "Database: ${NEURONDB_DATABASE}@${NEURONDB_HOST}:${NEURONDB_PORT}"
echo -e "User: ${NEURONDB_USER}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Run the MCP server
exec "$BINARY_PATH"

