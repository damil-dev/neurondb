#!/bin/bash

# NeuronMCP Server Startup Script
# Runs the MCP server with configured environment variables

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Try neurondb-mcp first, fall back to neuronmcp
if [ -f "${SCRIPT_DIR}/bin/neurondb-mcp" ]; then
    BINARY_PATH="${SCRIPT_DIR}/bin/neurondb-mcp"
elif [ -f "${SCRIPT_DIR}/bin/neuronmcp" ]; then
    BINARY_PATH="${SCRIPT_DIR}/bin/neuronmcp"
elif [ -f "${SCRIPT_DIR}/neurondb-mcp" ]; then
    BINARY_PATH="${SCRIPT_DIR}/neurondb-mcp"
else
    BINARY_PATH="${SCRIPT_DIR}/bin/neurondb-mcp"
fi

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
export NEURONDB_USER="pgedge"
export NEURONDB_PASSWORD=""

# Display configuration
echo "=========================================="
echo "NeuronMCP Server Starting"
echo "=========================================="
echo "Command: $BINARY_PATH"
echo "Database: $NEURONDB_DATABASE@$NEURONDB_HOST:$NEURONDB_PORT"
echo "User: $NEURONDB_USER"
echo "=========================================="
echo ""
echo "Press Ctrl+C to shutdown gracefully"
echo ""

# Change to script directory
cd "$SCRIPT_DIR"

# Run the MCP server in the foreground using exec
# This replaces the shell process, maintaining stdio connection
# The binary itself handles SIGINT and SIGTERM signals for graceful shutdown
# Using exec is necessary because:
# 1. MCP servers communicate via stdio (stdin/stdout for JSON-RPC)
# 2. Running in background (&) disconnects stdin, causing immediate EOF and exit
# 3. The binary has its own signal handlers for graceful shutdown
exec "$BINARY_PATH"








