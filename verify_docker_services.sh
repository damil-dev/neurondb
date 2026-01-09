#!/bin/bash
# Quick verification script for Docker services
# Usage: ./verify_docker_services.sh [neurondb|neuronagent|neuronmcp|neurondesktop|all]

set -e

PROFILE="${DOCKER_PROFILE:-cpu}"
SERVICE="${1:-all}"

cd "$(dirname "$0")"

echo "=== NeuronDB Docker Service Verification ==="
echo "Profile: $PROFILE"
echo "Service: $SERVICE"
echo ""

case "$SERVICE" in
  neurondb)
    echo "1. Checking NeuronDB container..."
    docker compose --profile "$PROFILE" ps neurondb
    echo ""
    echo "2. Checking NeuronDB health..."
    docker compose --profile "$PROFILE" exec neurondb pg_isready -U neurondb -d neurondb 2>&1 || echo "  Container not ready yet"
    echo ""
    echo "3. Testing NeuronDB extension..."
    docker compose --profile "$PROFILE" exec neurondb psql -U neurondb -d neurondb -c "SELECT neurondb.version();" 2>&1 || echo "  Extension not loaded yet"
    ;;
  
  neuronagent)
    echo "1. Checking NeuronAgent container..."
    docker compose --profile "$PROFILE" ps neuronagent
    echo ""
    echo "2. Checking NeuronAgent health endpoint..."
    curl -fsS http://localhost:8080/health 2>&1 || echo "  Service not ready yet"
    ;;
  
  neuronmcp)
    echo "1. Checking NeuronMCP container..."
    docker compose --profile "$PROFILE" ps neuronmcp
    echo ""
    echo "2. Checking NeuronMCP binary..."
    docker compose --profile "$PROFILE" exec neuronmcp test -f /app/neuronmcp 2>&1 && echo "  ✓ Binary exists" || echo "  ✗ Binary not found"
    ;;
  
  neurondesktop|desktop)
    echo "1. Checking NeuronDesktop containers..."
    docker compose --profile "$PROFILE" ps neurondesk-api neurondesk-frontend
    echo ""
    echo "2. Checking NeuronDesktop API health..."
    curl -fsS http://localhost:8081/health 2>&1 || echo "  API not ready yet"
    echo ""
    echo "3. Checking NeuronDesktop frontend..."
    curl -fsS http://localhost:3000 2>&1 | head -1 || echo "  Frontend not ready yet"
    ;;
  
  all)
    echo "Verifying all services..."
    echo ""
    $0 neurondb
    echo ""
    $0 neuronagent
    echo ""
    $0 neuronmcp
    echo ""
    $0 neurondesktop
    ;;
  
  *)
    echo "Usage: $0 [neurondb|neuronagent|neuronmcp|neurondesktop|all]"
    exit 1
    ;;
esac

echo ""
echo "=== Verification Complete ==="


