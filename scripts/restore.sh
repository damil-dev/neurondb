#!/bin/bash
# Restore script for NeuronDB ecosystem
# Usage: ./scripts/restore.sh <backup-file>

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <backup-file>"
    exit 1
fi

BACKUP_FILE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load environment variables
if [ -f "$REPO_ROOT/.env" ]; then
    source "$REPO_ROOT/.env"
fi

# Default values
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5433}"
DB_NAME="${DB_NAME:-neurondb}"
DB_USER="${DB_USER:-neurondb}"
DB_PASSWORD="${DB_PASSWORD:-neurondb}"

# Extract backup
TEMP_DIR=$(mktemp -d)
echo "Extracting backup..."
tar -xzf "$BACKUP_FILE" -C "$TEMP_DIR"

BACKUP_DIR=$(find "$TEMP_DIR" -type d -name "neurondb_backup_*" | head -1)

if [ -z "$BACKUP_DIR" ]; then
    echo "Error: Could not find backup directory in archive"
    exit 1
fi

echo "Restore directory: $BACKUP_DIR"

# Read manifest
if [ -f "$BACKUP_DIR/manifest.json" ]; then
    echo "Backup manifest:"
    cat "$BACKUP_DIR/manifest.json"
    echo ""
fi

# Confirm restore
read -p "This will overwrite existing data. Continue? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Restore cancelled"
    exit 0
fi

# Restore PostgreSQL database
echo "Restoring PostgreSQL database..."
PGPASSWORD="$DB_PASSWORD" pg_restore \
    -h "$DB_HOST" \
    -p "$DB_PORT" \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    --clean \
    --if-exists \
    "$BACKUP_DIR/neurondb.dump"

# Restore NeuronDesktop database
if [ -f "$BACKUP_DIR/neurondesk.dump" ]; then
    echo "Restoring NeuronDesktop database..."
    PGPASSWORD="$DB_PASSWORD" pg_restore \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "${NEURONDESK_DB_NAME:-neurondesk}" \
        --clean \
        --if-exists \
        "$BACKUP_DIR/neurondesk.dump"
fi

# Restore configuration files (optional)
if [ -d "$BACKUP_DIR/config" ]; then
    read -p "Restore configuration files? (yes/no): " restore_config
    if [ "$restore_config" = "yes" ]; then
        echo "Restoring configuration files..."
        cp -r "$BACKUP_DIR/config/"* "$REPO_ROOT/" || true
    fi
fi

# Cleanup
rm -rf "$TEMP_DIR"

echo "Restore complete!"

