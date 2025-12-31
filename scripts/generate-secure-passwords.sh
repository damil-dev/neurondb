#!/bin/bash
# Generate secure passwords for NeuronDB deployment
# Usage: ./scripts/generate-secure-passwords.sh > .env.secure
# Then review and copy values to your .env file

set -e

echo "# Secure passwords generated on $(date)"
echo "# Copy these values to your .env file"
echo ""
echo "# PostgreSQL / NeuronDB"
echo "POSTGRES_USER=neurondb"
echo "POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -d '\n')"
echo "POSTGRES_DB=neurondb"
echo ""
echo "# NeuronAgent (must match POSTGRES_PASSWORD)"
echo "DB_PASSWORD=\${POSTGRES_PASSWORD}"
echo ""
echo "# NeuronMCP (must match POSTGRES_PASSWORD)"
echo "NEURONDB_PASSWORD=\${POSTGRES_PASSWORD}"
echo ""
echo "# Generate NeuronAgent API key (if needed)"
echo "# Use: openssl rand -hex 32"

