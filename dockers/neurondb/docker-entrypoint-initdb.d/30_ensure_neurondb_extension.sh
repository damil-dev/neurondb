#!/bin/bash
set -euo pipefail

# This script ensures the NeuronDB extension is created after PostgreSQL restarts
# It runs after all initdb scripts complete and PostgreSQL is running

echo "[init] Ensuring NeuronDB extension is available..."

# Wait for PostgreSQL to be fully ready
until pg_isready -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-postgres}" >/dev/null 2>&1; do
    sleep 0.5
done

# Small additional wait to ensure PostgreSQL is fully initialized
sleep 1

# Try to create the extension (it may already exist or may fail if shared_preload_libraries isn't loaded yet)
# We use psql with the database from environment or default
DB_NAME="${POSTGRES_DB:-postgres}"
PGUSER="${POSTGRES_USER:-postgres}"

# Check if extension exists
if psql -U "$PGUSER" -d "$DB_NAME" -tAc "SELECT 1 FROM pg_extension WHERE extname = 'neurondb';" 2>/dev/null | grep -q 1; then
    echo "[init] NeuronDB extension already exists"
    exit 0
fi

# Try to create the extension
if psql -U "$PGUSER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS neurondb;" 2>/dev/null; then
    echo "[init] ✓ NeuronDB extension created successfully"
else
    echo "[init] ⚠ Could not create NeuronDB extension (shared_preload_libraries may need restart)"
    echo "[init]   Extension will be created automatically after container restart"
fi





