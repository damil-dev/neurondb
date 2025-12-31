#!/bin/bash
#
# Unified database setup for the NeuronDB ecosystem (no Docker required).
#
# This script exists because older docs/scripts referenced:
#   ./scripts/setup_neurondb_ecosystem.sh
# and it should not be a dead link.
#
# What it does (against an existing PostgreSQL instance):
# - creates the target database (if missing)
# - creates the `neurondb` extension (if possible)
# - installs NeuronMCP schema
# - installs NeuronAgent schema + runs migrations
#
# Configuration via env vars (defaults match older docs):
#   DB_HOST=localhost
#   DB_PORT=5432
#   DB_NAME=neurondb
#   DB_USER=postgres
#   DB_PASSWORD=   (optional; uses PGPASSWORD if set)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-neurondb}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-}"

if ! command -v psql >/dev/null 2>&1; then
  echo "ERROR: psql not found. Install PostgreSQL client tools first." >&2
  exit 1
fi

export PGHOST="${DB_HOST}"
export PGPORT="${DB_PORT}"
export PGUSER="${DB_USER}"

if [ -n "${DB_PASSWORD}" ]; then
  export PGPASSWORD="${DB_PASSWORD}"
fi

echo "== NeuronDB ecosystem database setup =="
echo "Target: ${DB_HOST}:${DB_PORT}/${DB_NAME} (user: ${DB_USER})"
echo ""

echo "-> Ensuring database exists: ${DB_NAME}"
DB_EXISTS="$(psql -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname = '${DB_NAME}'" || true)"
if [ "${DB_EXISTS}" != "1" ]; then
  createdb "${DB_NAME}"
  echo "   created database ${DB_NAME}"
else
  echo "   database already exists"
fi

echo "-> Creating extension: neurondb"
set +e
psql -d "${DB_NAME}" -c "CREATE EXTENSION IF NOT EXISTS neurondb;" >/dev/null 2>&1
EXT_RC=$?
set -e
if [ ${EXT_RC} -ne 0 ]; then
  echo "   WARN: could not create extension 'neurondb'."
  echo "         If you're using Docker, run: docker compose up -d"
  echo "         If you're doing local build, run: make install-neurondb (or see NeuronDB/INSTALL.md)"
else
  echo "   extension ensured"
fi

echo "-> Setting up NeuronMCP schema"
if [ ! -f "${REPO_ROOT}/NeuronMCP/scripts/setup_neurondb_mcp.sh" ]; then
  echo "ERROR: missing NeuronMCP setup script: NeuronMCP/scripts/setup_neurondb_mcp.sh" >&2
  exit 1
fi
export NEURONDB_HOST="${DB_HOST}"
export NEURONDB_PORT="${DB_PORT}"
export NEURONDB_DATABASE="${DB_NAME}"
export NEURONDB_USER="${DB_USER}"
export NEURONDB_PASSWORD="${DB_PASSWORD}"
bash "${REPO_ROOT}/NeuronMCP/scripts/setup_neurondb_mcp.sh"
echo "   NeuronMCP schema done"

echo "-> Setting up NeuronAgent schema"
if [ ! -f "${REPO_ROOT}/NeuronAgent/scripts/setup_neurondb_agent.sh" ]; then
  echo "ERROR: missing NeuronAgent setup script: NeuronAgent/scripts/setup_neurondb_agent.sh" >&2
  exit 1
fi
bash "${REPO_ROOT}/NeuronAgent/scripts/setup_neurondb_agent.sh"
echo "   NeuronAgent schema done"

echo "-> Running NeuronAgent migrations"
if [ ! -f "${REPO_ROOT}/NeuronAgent/scripts/run_migrations.sh" ]; then
  echo "ERROR: missing NeuronAgent migration runner: NeuronAgent/scripts/run_migrations.sh" >&2
  exit 1
fi
bash "${REPO_ROOT}/NeuronAgent/scripts/run_migrations.sh"
echo "   NeuronAgent migrations done"

echo ""
echo "OK: ecosystem DB setup completed."
echo "Next: ./scripts/integration-test.sh --tier 0 (or run docker services if using Docker)"


