#!/bin/sh
# Docker entrypoint script for NeuronMCP
# Performs pre-start validation and initialization

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Check if binary exists
if [ ! -f "/app/neurondb-mcp" ]; then
    log_error "Binary /app/neurondb-mcp not found!"
    exit 1
fi

if [ ! -x "/app/neurondb-mcp" ]; then
    log_error "Binary /app/neurondb-mcp is not executable!"
    exit 1
fi

log_info "Binary found and executable"

# Validate environment variables
if [ -z "${NEURONDB_HOST}" ] && [ -z "${NEURONDB_CONNECTION_STRING}" ]; then
    log_warn "NEURONDB_HOST not set, will use default or config file"
fi

if [ -z "${NEURONDB_DATABASE}" ] && [ -z "${NEURONDB_CONNECTION_STRING}" ]; then
    log_warn "NEURONDB_DATABASE not set, will use default or config file"
fi

# Optional: Test database connectivity (requires psql or similar)
# Uncomment if you want to verify database connection before starting
# if command -v psql >/dev/null 2>&1; then
#     log_info "Testing database connectivity..."
#     if [ -n "${NEURONDB_CONNECTION_STRING}" ]; then
#         CONN_STR="${NEURONDB_CONNECTION_STRING}"
#     else
#         CONN_STR="postgresql://${NEURONDB_USER:-neurondb}:${NEURONDB_PASSWORD:-neurondb}@${NEURONDB_HOST:-localhost}:${NEURONDB_PORT:-5432}/${NEURONDB_DATABASE:-neurondb}"
#     fi
#     
#     if psql "${CONN_STR}" -c "SELECT 1;" >/dev/null 2>&1; then
#         log_info "Database connection successful"
#     else
#         log_warn "Database connection test failed (continuing anyway)"
#     fi
# fi

# Validate config file if specified
if [ -n "${NEURONDB_MCP_CONFIG}" ] && [ -f "${NEURONDB_MCP_CONFIG}" ]; then
    log_info "Config file found: ${NEURONDB_MCP_CONFIG}"
    # Basic JSON validation (requires python3)
    if command -v python3 >/dev/null 2>&1; then
        if python3 -m json.tool "${NEURONDB_MCP_CONFIG}" >/dev/null 2>&1; then
            log_info "Config file is valid JSON"
        else
            log_error "Config file is not valid JSON: ${NEURONDB_MCP_CONFIG}"
            exit 1
        fi
    fi
elif [ -n "${NEURONDB_MCP_CONFIG}" ]; then
    log_warn "Config file specified but not found: ${NEURONDB_MCP_CONFIG}"
fi

# Log startup information
log_info "Starting NeuronMCP server..."
log_info "  Host: ${NEURONDB_HOST:-localhost}"
log_info "  Port: ${NEURONDB_PORT:-5433}"
log_info "  Database: ${NEURONDB_DATABASE:-neurondb}"
log_info "  User: ${NEURONDB_USER:-neurondb}"
log_info "  Log Level: ${NEURONDB_LOG_LEVEL:-info}"
log_info "  Log Format: ${NEURONDB_LOG_FORMAT:-text}"

# Execute the binary with all arguments
exec /app/neurondb-mcp "$@"

