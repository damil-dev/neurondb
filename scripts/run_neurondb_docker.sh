#!/bin/bash
# NeuronDB Docker Management Script
# Usage: ./run_neurondb_docker.sh [build|clean|run] [cpu|cuda]
#   build - Build the NeuronDB Docker image
#   clean - Stop and remove containers/volumes
#   run   - Start NeuronDB container
#   cpu   - Use CPU variant (default)
#   cuda  - Use CUDA GPU variant

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

# Default PostgreSQL version and variant
PG_MAJOR="${PG_MAJOR:-18}"
VARIANT="${2:-cpu}"

# Determine service name and profile based on variant
case "${VARIANT}" in
    cuda)
        SERVICE_NAME="neurondb-cuda"
        PROFILE="cuda"
        CONTAINER_NAME="neurondb-cuda"
        PORT="${POSTGRES_CUDA_PORT:-5434}"
        ;;
    cpu|*)
        SERVICE_NAME="neurondb"
        PROFILE="cpu"
        CONTAINER_NAME="neurondb-cpu"
        PORT="${POSTGRES_PORT:-5433}"
        VARIANT="cpu"
        ;;
esac

case "${1:-run}" in
    build)
        echo "Building NeuronDB Docker image (PostgreSQL ${PG_MAJOR}, ${VARIANT} variant)..."
        export PG_MAJOR="${PG_MAJOR}"
        if [ "$VARIANT" = "cuda" ]; then
            export CUDA_VERSION="${CUDA_VERSION:-12.4.1}"
            export ONNX_VERSION="${ONNX_VERSION:-1.17.0}"
            if [ "$USE_PROFILES" = true ]; then
                $DOCKER_COMPOSE_CMD --profile ${PROFILE} build \
                    --build-arg PG_MAJOR="${PG_MAJOR}" \
                    --build-arg CUDA_VERSION="${CUDA_VERSION}" \
                    --build-arg ONNX_VERSION="${ONNX_VERSION}" \
                    ${SERVICE_NAME}
            else
                $DOCKER_COMPOSE_CMD build \
                    --build-arg PG_MAJOR="${PG_MAJOR}" \
                    --build-arg CUDA_VERSION="${CUDA_VERSION}" \
                    --build-arg ONNX_VERSION="${ONNX_VERSION}" \
                    ${SERVICE_NAME}
            fi
        else
            if [ "$USE_PROFILES" = true ]; then
                $DOCKER_COMPOSE_CMD --profile ${PROFILE} build --build-arg PG_MAJOR="${PG_MAJOR}" ${SERVICE_NAME}
            else
                $DOCKER_COMPOSE_CMD build --build-arg PG_MAJOR="${PG_MAJOR}" ${SERVICE_NAME}
            fi
        fi
        echo "Build completed!"
        ;;
    clean)
        echo "Cleaning up NeuronDB containers and volumes (${VARIANT} variant)..."
        if [ "$USE_PROFILES" = true ]; then
            $DOCKER_COMPOSE_CMD --profile ${PROFILE} down -v
        else
            $DOCKER_COMPOSE_CMD down -v
        fi
        echo "Cleanup completed!"
        ;;
    run)
        echo "Starting NeuronDB container (PostgreSQL ${PG_MAJOR}, ${VARIANT} variant)..."
        export PG_MAJOR="${PG_MAJOR}"
        if [ "$VARIANT" = "cuda" ]; then
            export CUDA_VERSION="${CUDA_VERSION:-12.4.1}"
            export ONNX_VERSION="${ONNX_VERSION:-1.17.0}"
            if [ "$USE_PROFILES" = true ]; then
                $DOCKER_COMPOSE_CMD --profile ${PROFILE} up -d ${SERVICE_NAME}
            else
                $DOCKER_COMPOSE_CMD up -d ${SERVICE_NAME}
            fi
        else
            if [ "$USE_PROFILES" = true ]; then
                $DOCKER_COMPOSE_CMD --profile ${PROFILE} up -d ${SERVICE_NAME}
            else
                $DOCKER_COMPOSE_CMD up -d ${SERVICE_NAME}
            fi
        fi
        echo "NeuronDB is starting..."
        echo "Waiting for container to be healthy..."
        sleep 5
        echo "Connection: postgresql://neurondb:neurondb@localhost:${PORT}/neurondb"
        echo "Check status: docker ps | grep ${CONTAINER_NAME}"
        echo "Check health: docker inspect ${CONTAINER_NAME} | grep -A 10 Health"
        if [ "$VARIANT" = "cuda" ]; then
            echo "Check GPU: docker exec ${CONTAINER_NAME} nvidia-smi"
        fi
        ;;
    *)
        echo "Usage: $0 [build|clean|run] [cpu|cuda]"
        echo ""
        echo "Commands:"
        echo "  build - Build the NeuronDB Docker image"
        echo "  clean - Stop and remove containers/volumes"
        echo "  run   - Start NeuronDB container (default)"
        echo ""
        echo "Variants:"
        echo "  cpu   - CPU-only variant (default)"
        echo "  cuda  - CUDA GPU variant"
        echo ""
        echo "Environment variables:"
        echo "  PG_MAJOR - PostgreSQL version (default: 18)"
        echo "  CUDA_VERSION - CUDA version for cuda variant (default: 12.4.1)"
        echo "  ONNX_VERSION - ONNX Runtime version (default: 1.17.0)"
        echo ""
        echo "Examples:"
        echo "  $0 run cpu     # Run CPU variant"
        echo "  $0 run cuda    # Run CUDA variant"
        echo "  $0 build cuda  # Build CUDA variant"
        exit 1
        ;;
esac

