#!/bin/bash
# NeuronAgent Docker Management Script
# Usage: ./run_neuronagent_docker.sh [build|clean|run]
#   build - Build the NeuronAgent Docker image
#   clean - Stop and remove containers
#   run   - Start NeuronAgent container

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
DB_HOST="${DB_HOST:-neurondb}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-neurondb}"
DB_USER="${DB_USER:-neurondb}"
DB_PASSWORD="${DB_PASSWORD:-neurondb}"
SERVER_PORT="${SERVER_PORT:-8080}"

case "${1:-run}" in
    build)
        echo "Building NeuronAgent Docker image..."
        if [ "$USE_PROFILES" = true ]; then
            $DOCKER_COMPOSE_CMD --profile cpu build neuronagent
        else
            $DOCKER_COMPOSE_CMD build neuronagent
        fi
        echo "Build completed!"
        ;;
    clean)
        echo "Cleaning up NeuronAgent containers..."
        if [ "$USE_PROFILES" = true ]; then
            $DOCKER_COMPOSE_CMD --profile cpu stop neuronagent 2>/dev/null || true
            $DOCKER_COMPOSE_CMD --profile cpu rm -f neuronagent 2>/dev/null || true
        else
            $DOCKER_COMPOSE_CMD stop neuronagent 2>/dev/null || true
            $DOCKER_COMPOSE_CMD rm -f neuronagent 2>/dev/null || true
        fi
        echo "Cleanup completed!"
        ;;
    run)
        echo "Starting NeuronAgent container..."
        echo "Connecting to NeuronDB at ${DB_HOST}:${DB_PORT}"
        
        # Export environment variables for docker-compose
        export DB_HOST DB_PORT DB_NAME DB_USER DB_PASSWORD SERVER_PORT
        
        if [ "$USE_PROFILES" = true ]; then
            $DOCKER_COMPOSE_CMD --profile cpu up -d --no-deps neuronagent
        else
            $DOCKER_COMPOSE_CMD up -d --no-deps neuronagent
        fi
        echo "NeuronAgent is starting..."
        echo "API endpoint: http://localhost:${SERVER_PORT}"
        echo "Health check: curl http://localhost:${SERVER_PORT}/health"
        echo "Check status: docker ps | grep neuronagent"
        echo "View logs: docker logs neuronagent -f"
        ;;
    *)
        echo "Usage: $0 [build|clean|run]"
        echo ""
        echo "Commands:"
        echo "  build - Build the NeuronAgent Docker image"
        echo "  clean - Stop and remove containers"
        echo "  run   - Start NeuronAgent container (default)"
        echo ""
        echo "Environment variables:"
        echo "  DB_HOST     - NeuronDB host (default: neurondb)"
        echo "  DB_PORT     - NeuronDB port (default: 5432)"
        echo "  DB_NAME     - Database name (default: neurondb)"
        echo "  DB_USER     - Database user (default: neurondb)"
        echo "  DB_PASSWORD - Database password (default: neurondb)"
        echo "  SERVER_PORT - API server port (default: 8080)"
        exit 1
        ;;
esac

