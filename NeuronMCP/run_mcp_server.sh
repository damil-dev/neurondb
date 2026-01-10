#!/bin/bash

# NeuronMCP Server Startup Script
# Runs the MCP server with configured environment variables

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY_PATH="${SCRIPT_DIR}/bin/neurondb-mcp"

# Check if binary exists
if [ ! -f "$BINARY_PATH" ]; then
    echo "Error: MCP server binary not found at $BINARY_PATH" >&2
    echo "Please build the binary first with: make build" >&2
    exit 1
fi

# Set environment variables
export NEURONDB_HOST="localhost"
export NEURONDB_PORT="5432"
export NEURONDB_DATABASE="neurondb"
export NEURONDB_USER="pge"
export NEURONDB_PASSWORD="test"

# Display configuration
echo "=========================================="
echo "NeuronMCP Server Starting"
echo "=========================================="
echo "Command: $BINARY_PATH"
echo "Database: $NEURONDB_DATABASE@$NEURONDB_HOST:$NEURONDB_PORT"
echo "User: $NEURONDB_USER"
echo "=========================================="
echo ""

# Change to script directory
cd "$SCRIPT_DIR"

# Run the MCP server
exec "$BINARY_PATH"









