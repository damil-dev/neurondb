#!/bin/bash
# Backup script for NeuronDB ecosystem
# Usage: ./scripts/backup.sh [backup-dir]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BACKUP_DIR="${1:-$REPO_ROOT/backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/neurondb_backup_$TIMESTAMP"

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

echo "Starting NeuronDB backup..."
echo "Backup directory: $BACKUP_PATH"

# Create backup directory
mkdir -p "$BACKUP_PATH"

# Backup PostgreSQL database
echo "Backing up PostgreSQL database..."
PGPASSWORD="$DB_PASSWORD" pg_dump \
    -h "$DB_HOST" \
    -p "$DB_PORT" \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    -F c \
    -f "$BACKUP_PATH/neurondb.dump"

# Backup NeuronDesktop database
if [ -n "$NEURONDESK_DB_NAME" ]; then
    echo "Backing up NeuronDesktop database..."
    PGPASSWORD="$DB_PASSWORD" pg_dump \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "${NEURONDESK_DB_NAME:-neurondesk}" \
        -F c \
        -f "$BACKUP_PATH/neurondesk.dump"
fi

# Backup configuration files
echo "Backing up configuration files..."
mkdir -p "$BACKUP_PATH/config"
cp -r "$REPO_ROOT/NeuronDesktop/api/migrations" "$BACKUP_PATH/config/" 2>/dev/null || true
cp -r "$REPO_ROOT/NeuronAgent/sql" "$BACKUP_PATH/config/" 2>/dev/null || true
cp "$REPO_ROOT/docker-compose.yml" "$BACKUP_PATH/config/" 2>/dev/null || true
cp "$REPO_ROOT/.env" "$BACKUP_PATH/config/" 2>/dev/null || true

# Create backup manifest
cat > "$BACKUP_PATH/manifest.json" <<EOF
{
  "timestamp": "$TIMESTAMP",
  "version": "1.0.0",
  "databases": {
    "neurondb": "$DB_NAME",
    "neurondesk": "${NEURONDESK_DB_NAME:-neurondesk}"
  },
  "files": [
    "neurondb.dump",
    "neurondesk.dump",
    "config/"
  ]
}
EOF

# Compress backup
echo "Compressing backup..."
tar -czf "$BACKUP_PATH.tar.gz" -C "$BACKUP_DIR" "neurondb_backup_$TIMESTAMP"
rm -rf "$BACKUP_PATH"

echo "Backup complete: $BACKUP_PATH.tar.gz"

# Optional: Upload to S3
if [ -n "$BACKUP_S3_BUCKET" ]; then
    echo "Uploading to S3..."
    aws s3 cp "$BACKUP_PATH.tar.gz" "s3://$BACKUP_S3_BUCKET/backups/"
    echo "Uploaded to s3://$BACKUP_S3_BUCKET/backups/neurondb_backup_$TIMESTAMP.tar.gz"
fi

