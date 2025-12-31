#!/bin/bash
# NeuronMCP Docker Management Script
# Usage: ./run_neuronmcp_docker.sh [build|clean|run]
#   build - Build the NeuronMCP Docker image
#   clean - Stop and remove containers
#   run   - Start NeuronMCP container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Detect docker-compose command
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
    # Check if it supports profiles
    if docker-compose --profile test config &> /dev/null 2>&1; then
        USE_PROFILES=true
    else
        USE_PROFILES=false
    fi
else
    DOCKER_COMPOSE_CMD="docker compose"
    USE_PROFILES=true
fi

# Default connection settings
NEURONDB_HOST="${NEURONDB_HOST:-neurondb}"
NEURONDB_PORT="${NEURONDB_PORT:-5432}"
NEURONDB_DATABASE="${NEURONDB_DATABASE:-neurondb}"
NEURONDB_USER="${NEURONDB_USER:-neurondb}"
NEURONDB_PASSWORD="${NEURONDB_PASSWORD:-neurondb}"

case "${1:-run}" in
    build)
        echo "Building NeuronMCP Docker image..."
        $DOCKER_COMPOSE_CMD build neuronmcp
        echo "Build completed!"
        ;;
    clean)
        echo "Cleaning up NeuronMCP containers..."
        $DOCKER_COMPOSE_CMD stop neuronmcp 2>/dev/null || true
        $DOCKER_COMPOSE_CMD rm -f neuronmcp 2>/dev/null || true
        echo "Cleanup completed!"
        ;;
    run)
        echo "Starting NeuronMCP container..."
        echo "Connecting to NeuronDB at ${NEURONDB_HOST}:${NEURONDB_PORT}"
        
        # Export environment variables for docker-compose
        export NEURONDB_HOST NEURONDB_PORT NEURONDB_DATABASE NEURONDB_USER NEURONDB_PASSWORD
        
        # Use --no-deps to avoid docker-compose trying to recreate dependent services
        # Services should be started in order: NeuronDB first, then NeuronAgent, then NeuronMCP
        if [ "$USE_PROFILES" = true ]; then
            $DOCKER_COMPOSE_CMD --profile cpu up -d --no-deps neuronmcp
        else
            $DOCKER_COMPOSE_CMD up -d --no-deps neuronmcp
        fi
        echo "NeuronMCP is starting..."
        echo "Check status: docker ps | grep neurondb-mcp"
        echo "View logs: docker logs neurondb-mcp -f"
        ;;
    *)
        echo "Usage: $0 [build|clean|run]"
        echo ""
        echo "Commands:"
        echo "  build - Build the NeuronMCP Docker image"
        echo "  clean - Stop and remove containers"
        echo "  run   - Start NeuronMCP container (default)"
        echo ""
        echo "Environment variables:"
        echo "  NEURONDB_HOST     - NeuronDB host (default: neurondb)"
        echo "  NEURONDB_PORT     - NeuronDB port (default: 5432)"
        echo "  NEURONDB_DATABASE - Database name (default: neurondb)"
        echo "  NEURONDB_USER     - Database user (default: neurondb)"
        echo "  NEURONDB_PASSWORD - Database password (default: neurondb)"
        exit 1
        ;;
esac

