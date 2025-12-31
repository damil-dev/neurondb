#!/bin/bash
set -euo pipefail

# Custom entrypoint wrapper for NeuronDB
# This ensures the NeuronDB extension is created after PostgreSQL starts

# Ensure ML libraries are in library path
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-/usr/local/lib}

# Function to ensure NeuronDB extension exists
ensure_neurondb_extension() {
    local pguser="${POSTGRES_USER:-postgres}"
    local db_name="${POSTGRES_DB:-$pguser}"
    
    # Wait for PostgreSQL to be ready
    echo "[neurondb-entrypoint] Waiting for PostgreSQL to be ready..."
    local max_attempts=30
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if pg_isready -U "$pguser" >/dev/null 2>&1; then
            break
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -eq $max_attempts ]; then
        echo "[neurondb-entrypoint] ⚠ PostgreSQL did not become ready in time"
        return 1
    fi
    
    # Additional wait to ensure PostgreSQL is fully initialized
    sleep 2
    
    # Ensure NeuronDB extension exists
    echo "[neurondb-entrypoint] Ensuring NeuronDB extension is available..."
    if psql -U "$pguser" -d "$db_name" -tAc "SELECT 1 FROM pg_extension WHERE extname = 'neurondb';" 2>/dev/null | grep -q 1; then
        echo "[neurondb-entrypoint] ✓ NeuronDB extension already exists"
        return 0
    fi
    
    # Check if schema exists but extension doesn't (leftover from failed init)
    if psql -U "$pguser" -d "$db_name" -tAc "SELECT 1 FROM pg_namespace WHERE nspname = 'neurondb';" 2>/dev/null | grep -q 1; then
        echo "[neurondb-entrypoint] Found existing neurondb schema without extension, cleaning up..."
        psql -U "$pguser" -d "$db_name" -c "DROP SCHEMA IF EXISTS neurondb CASCADE;" 2>/dev/null || true
    fi
    
    # Try to create the extension
    if psql -U "$pguser" -d "$db_name" -c "CREATE EXTENSION IF NOT EXISTS neurondb;" 2>/dev/null; then
        echo "[neurondb-entrypoint] ✓ NeuronDB extension created successfully"
        return 0
    else
        echo "[neurondb-entrypoint] ⚠ Could not create NeuronDB extension"
        echo "[neurondb-entrypoint]   This may be normal if shared_preload_libraries needs a restart"
        return 1
    fi
}

# Check if this is the first argument and if it's 'postgres'.
# If so, we need to start PostgreSQL and then ensure the extension.
#
# When running under docker-compose, the container may start with no explicit
# CMD args; guard against set -u.
if [ "${1:-}" = 'postgres' ]; then
    # Start PostgreSQL in the background
    /usr/local/bin/docker-entrypoint.sh "$@" &
    POSTGRES_PID=$!
    
    # Set up signal handlers to forward signals to PostgreSQL
    trap "kill -TERM $POSTGRES_PID" SIGTERM SIGINT
    
    # Ensure extension exists after PostgreSQL starts (with timeout)
    # Run this in background so we don't block signal handling
    (
        sleep 3  # Give PostgreSQL time to start
        ensure_neurondb_extension || true
    ) &
    
    # Wait for PostgreSQL process and forward its exit code
    wait $POSTGRES_PID
    exit $?
else
    # For other commands, just pass through to original entrypoint
    exec /usr/local/bin/docker-entrypoint.sh "$@"
fi

