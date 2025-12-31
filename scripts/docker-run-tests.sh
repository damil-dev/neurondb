#!/bin/bash
# Run NeuronDB Tests with Docker PostgreSQL
# 
# This script runs NeuronDB test suite (run_test.py) against a Docker PostgreSQL container.
# PostgreSQL and NeuronDB must run in Docker.
#
# It handles Docker-specific configuration and connection parameters.
#
# Usage:
#   ./run_tests_docker.sh [options]
#
# Options:
#   --port PORT          Docker PostgreSQL port (default: auto-detect or 5433)
#   --compute MODE        Compute mode: cpu, gpu, auto (default: cpu)
#   --category CATEGORY   Test category: basic, advance, negative, all (default: basic)
#   --container NAME     Docker container name (default: auto-detect)
#   --user USER          Database user (default: neurondb)
#   --password PASS      Database password (default: neurondb)
#   --db NAME            Database name (default: neurondb)
#   --host HOST          Database host (default: localhost)
#   --verbose            Enable verbose output
#   --test TEST_NAME     Run specific test
#   --module MODULE      Run tests for specific module
#   --help               Show this help message
#
# Examples:
#   # Run basic tests with CPU mode on default port
#   ./run_tests_docker.sh --compute=cpu
#
#   # Run tests on specific port
#   ./run_tests_docker.sh --compute=cpu --port=5433
#
#   # Run all tests with verbose output
#   ./run_tests_docker.sh --compute=cpu --category=all --verbose

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DEFAULT_COMPUTE="cpu"
DEFAULT_CATEGORY="basic"
DEFAULT_USER="neurondb"
DEFAULT_PASSWORD="neurondb"
DEFAULT_DB="neurondb"
DEFAULT_HOST="localhost"
DEFAULT_PORT=""

# Container name mappings
CPU_CONTAINER="neurondb-cpu"
CUDA_CONTAINER="neurondb-cuda"
ROCM_CONTAINER="neurondb-rocm"
METAL_CONTAINER="neurondb-metal"

# Default ports for each container type
CPU_PORT=5433
CUDA_PORT=5434
ROCM_PORT=5435
METAL_PORT=5436

# Parse arguments
COMPUTE_MODE="$DEFAULT_COMPUTE"
CATEGORY="$DEFAULT_CATEGORY"
DB_USER="$DEFAULT_USER"
DB_PASSWORD="$DEFAULT_PASSWORD"
DB_NAME="$DEFAULT_DB"
DB_HOST="$DEFAULT_HOST"
DB_PORT="$DEFAULT_PORT"
CONTAINER_NAME=""
VERBOSE=""
TEST_NAME=""
MODULE=""

# Function to show help
show_help() {
    cat << EOF
Run NeuronDB Tests with Docker PostgreSQL

Usage: $0 [options]

Options:
  --port PORT          Docker PostgreSQL port (default: auto-detect or 5433)
  --compute MODE       Compute mode: cpu, gpu, auto (default: cpu)
  --category CATEGORY  Test category: basic, advance, negative, all (default: basic)
  --container NAME     Docker container name (default: auto-detect)
  --user USER          Database user (default: neurondb)
  --password PASS      Database password (default: neurondb)
  --db NAME            Database name (default: neurondb)
  --host HOST          Database host (default: localhost)
  --verbose            Enable verbose output
  --test TEST_NAME     Run specific test
  --module MODULE      Run tests for specific module
  --help               Show this help message

Examples:
  # Run basic tests with CPU mode (auto-detects Docker container and port)
  $0 --compute=cpu

  # Run tests on specific port
  $0 --compute=cpu --port=5433

  # Run all tests with verbose output
  $0 --compute=cpu --category=all --verbose

  # Run tests with specific Docker container
  $0 --compute=cpu --container=neurondb-cpu
EOF
}

# Function to parse option value (supports both --opt=value and --opt value)
parse_option_value() {
    local arg="$1"
    if [[ "$arg" == *"="* ]]; then
        # Format: --opt=value
        echo "${arg#*=}"
    else
        # Format: --opt value
        echo "$2"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port=*)
            DB_PORT="${1#*=}"
            shift
            ;;
        --port)
            DB_PORT=$(parse_option_value "$1" "$2")
            if [[ "$DB_PORT" != "$2" ]]; then
                shift
            else
                shift 2
            fi
            ;;
        --compute=*)
            COMPUTE_MODE="${1#*=}"
            shift
            ;;
        --compute)
            COMPUTE_MODE=$(parse_option_value "$1" "$2")
            if [[ "$COMPUTE_MODE" != "$2" ]]; then
                shift
            else
                shift 2
            fi
            ;;
        --category=*)
            CATEGORY="${1#*=}"
            shift
            ;;
        --category)
            CATEGORY=$(parse_option_value "$1" "$2")
            if [[ "$CATEGORY" != "$2" ]]; then
                shift
            else
                shift 2
            fi
            ;;
        --container=*)
            CONTAINER_NAME="${1#*=}"
            shift
            ;;
        --container)
            CONTAINER_NAME=$(parse_option_value "$1" "$2")
            if [[ "$CONTAINER_NAME" != "$2" ]]; then
                shift
            else
                shift 2
            fi
            ;;
        --user=*)
            DB_USER="${1#*=}"
            shift
            ;;
        --user)
            DB_USER=$(parse_option_value "$1" "$2")
            if [[ "$DB_USER" != "$2" ]]; then
                shift
            else
                shift 2
            fi
            ;;
        --password=*)
            DB_PASSWORD="${1#*=}"
            shift
            ;;
        --password)
            DB_PASSWORD=$(parse_option_value "$1" "$2")
            if [[ "$DB_PASSWORD" != "$2" ]]; then
                shift
            else
                shift 2
            fi
            ;;
        --db=*)
            DB_NAME="${1#*=}"
            shift
            ;;
        --db)
            DB_NAME=$(parse_option_value "$1" "$2")
            if [[ "$DB_NAME" != "$2" ]]; then
                shift
            else
                shift 2
            fi
            ;;
        --host=*)
            DB_HOST="${1#*=}"
            shift
            ;;
        --host)
            DB_HOST=$(parse_option_value "$1" "$2")
            if [[ "$DB_HOST" != "$2" ]]; then
                shift
            else
                shift 2
            fi
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --test=*)
            TEST_NAME="${1#*=}"
            shift
            ;;
        --test)
            TEST_NAME=$(parse_option_value "$1" "$2")
            if [[ "$TEST_NAME" != "$2" ]]; then
                shift
            else
                shift 2
            fi
            ;;
        --module=*)
            MODULE="${1#*=}"
            shift
            ;;
        --module)
            MODULE=$(parse_option_value "$1" "$2")
            if [[ "$MODULE" != "$2" ]]; then
                shift
            else
                shift 2
            fi
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Error:${NC} Unknown option: $1" >&2
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Global variable for docker command
DOCKER_CMD="docker"

# Function to detect Docker container and port
detect_container_and_port() {
    local compute="$1"
    
    # Check if docker command is available and accessible
    if ! command -v docker &> /dev/null; then
        echo -e "${YELLOW}Warning:${NC} docker command not found" >&2
        return 1
    fi
    
    # Try docker without sudo first, fall back to sudo if needed
    DOCKER_CMD="docker"
    if ! docker ps &>/dev/null 2>&1; then
        if command -v sudo &> /dev/null && sudo docker ps &>/dev/null 2>&1; then
            DOCKER_CMD="sudo docker"
            echo -e "${BLUE}Note:${NC} Using sudo for docker commands" >&2
        else
            echo -e "${YELLOW}Warning:${NC} Cannot access docker (permission denied). Continuing with default port..." >&2
            return 1
        fi
    fi
    
    # If container name is provided, use it
    if [ -n "$CONTAINER_NAME" ]; then
        if $DOCKER_CMD ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
            echo -e "${BLUE}Using specified container: ${CONTAINER_NAME}${NC}"
            # Try to get port from container
            local port_mapping=$($DOCKER_CMD port "$CONTAINER_NAME" 2>/dev/null | grep "5432/tcp" | head -1 | cut -d: -f2)
            if [ -n "$port_mapping" ]; then
                DB_PORT="$port_mapping"
                echo -e "${BLUE}Detected port from container: ${DB_PORT}${NC}"
            fi
            return 0
        else
            echo -e "${YELLOW}Warning:${NC} Container '${CONTAINER_NAME}' not found. Trying to auto-detect..." >&2
        fi
    fi
    
    # Auto-detect container based on compute mode
    local container=""
    case "$compute" in
        cpu)
            container="$CPU_CONTAINER"
            DB_PORT="${DB_PORT:-$CPU_PORT}"
            ;;
        gpu|cuda)
            container="$CUDA_CONTAINER"
            DB_PORT="${DB_PORT:-$CUDA_PORT}"
            ;;
        rocm)
            container="$ROCM_CONTAINER"
            DB_PORT="${DB_PORT:-$ROCM_PORT}"
            ;;
        metal)
            container="$METAL_CONTAINER"
            DB_PORT="${DB_PORT:-$METAL_PORT}"
            ;;
        *)
            container="$CPU_CONTAINER"
            DB_PORT="${DB_PORT:-$CPU_PORT}"
            ;;
    esac
    
    # Check if container is running
    if $DOCKER_CMD ps --format "{{.Names}}" | grep -q "^${container}$"; then
        CONTAINER_NAME="$container"
        echo -e "${GREEN}Detected running container: ${CONTAINER_NAME}${NC}"
        
        # Try to get actual port from container
        local port_mapping=$($DOCKER_CMD port "$CONTAINER_NAME" 2>/dev/null | grep "5432/tcp" | head -1 | cut -d: -f2)
        if [ -n "$port_mapping" ]; then
            DB_PORT="$port_mapping"
            echo -e "${GREEN}Detected port from container: ${DB_PORT}${NC}"
        else
            echo -e "${BLUE}Using default port for ${compute} mode: ${DB_PORT}${NC}"
        fi
        return 0
    else
        echo -e "${YELLOW}Warning:${NC} Container '${container}' not found. Using default port: ${DB_PORT}${NC}" >&2
        CONTAINER_NAME="$container"
        return 1
    fi
}

# Function to verify Docker connection
verify_connection() {
    local port="$1"
    local user="$2"
    local password="$3"
    local db="$4"
    local host="$5"
    local is_local="$6"
    
    echo -e "${BLUE}Verifying connection to PostgreSQL at ${host}:${port}...${NC}"
    
    if PGPASSWORD="$password" psql -h "$host" -p "$port" -U "$user" -d "$db" -c "SELECT version();" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Connection successful${NC}"
        return 0
    else
        echo -e "${RED}✗ Connection failed${NC}" >&2
        echo -e "${YELLOW}Please ensure:${NC}" >&2
        echo -e "  1. Docker container is running: ${YELLOW}docker ps | grep neurondb${NC}" >&2
        if [ -n "$CONTAINER_NAME" ]; then
            echo -e "  2. Port is correct: ${YELLOW}docker port ${CONTAINER_NAME}${NC}" >&2
        else
            echo -e "  2. Port is correct: ${YELLOW}docker ps --format 'table {{.Names}}\t{{.Ports}}'${NC}" >&2
        fi
        echo -e "  3. Credentials match docker-compose.yml settings" >&2
        echo -e "  4. Container is healthy: ${YELLOW}docker ps | grep ${CONTAINER_NAME:-neurondb}${NC}" >&2
        return 1
    fi
}

# Function to check if compute mode needs restart
check_compute_mode() {
    local port="$1"
    local user="$2"
    local password="$3"
    local db="$4"
    local host="$5"
    local compute="$6"
    
    # Map compute mode to enum value
    local mode_enum=0
    case "$compute" in
        cpu) mode_enum=0 ;;
        gpu) mode_enum=1 ;;
        auto) mode_enum=2 ;;
        *) mode_enum=0 ;;
    esac
    
    local current_mode=$(PGPASSWORD="$password" psql -h "$host" -p "$port" -U "$user" -d "$db" -t -A -c "SELECT current_setting('neurondb.compute_mode');" 2>/dev/null | tr -d ' ')
    
    if [ "$current_mode" = "$mode_enum" ]; then
        echo -e "${GREEN}✓ Compute mode already set to ${compute} (${mode_enum})${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ Compute mode is ${current_mode}, needs to be ${mode_enum} for ${compute} mode${NC}"
        return 1
    fi
}

# Main execution
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}NeuronDB Docker Test Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}PostgreSQL and NeuronDB must run in Docker${NC}"
echo ""

# Detect container and port
if ! detect_container_and_port "$COMPUTE_MODE"; then
    echo -e "${YELLOW}Warning:${NC} Could not detect running container. Continuing with default port..." >&2
    echo -e "${YELLOW}Please ensure Docker container is running:${NC} ${BLUE}docker ps | grep neurondb${NC}" >&2
fi

# Verify connection
if ! verify_connection "$DB_PORT" "$DB_USER" "$DB_PASSWORD" "$DB_NAME" "$DB_HOST" "false"; then
    exit 1
fi

# Check compute mode
if check_compute_mode "$DB_PORT" "$DB_USER" "$DB_PASSWORD" "$DB_NAME" "$DB_HOST" "$COMPUTE_MODE"; then
    echo -e "${GREEN}No restart needed${NC}"
else
    echo -e "${YELLOW}Note:${NC} The test script will attempt to set compute mode."
    echo -e "${YELLOW}If it fails to restart PostgreSQL (Docker limitation), you may need to:${NC}"
    echo -e "  1. Let the script set the mode, then manually restart: ${BLUE}docker restart ${CONTAINER_NAME}${NC}"
    echo -e "  2. Or pre-configure: ${BLUE}docker exec ${CONTAINER_NAME} psql -U ${DB_USER} -d ${DB_NAME} -c \"ALTER SYSTEM SET neurondb.compute_mode = $mode_enum;\" && docker restart ${CONTAINER_NAME}${NC}"
    echo ""
fi

# Build test command
TEST_DIR="$REPO_ROOT/NeuronDB/tests"
if [ ! -f "$TEST_DIR/run_test.py" ]; then
    echo -e "${RED}Error:${NC} Test script not found: $TEST_DIR/run_test.py" >&2
    exit 1
fi

echo -e "${BLUE}Running tests...${NC}"
echo ""

# Build command arguments
CMD_ARGS=(
    "--compute=$COMPUTE_MODE"
    "--port=$DB_PORT"
    "--host=$DB_HOST"
    "--user=$DB_USER"
    "--password=$DB_PASSWORD"
    "--db=$DB_NAME"
    "--category=$CATEGORY"
)

if [ -n "$VERBOSE" ]; then
    CMD_ARGS+=("$VERBOSE")
fi

if [ -n "$TEST_NAME" ]; then
    CMD_ARGS+=("--test=$TEST_NAME")
fi

if [ -n "$MODULE" ]; then
    CMD_ARGS+=("--module=$MODULE")
fi

# Run the test script
cd "$TEST_DIR"
python3 run_test.py "${CMD_ARGS[@]}"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Tests completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Tests completed with errors${NC}"
    echo -e "${RED}========================================${NC}"
    
    # Check if it's a restart issue
    if [ -n "$CONTAINER_NAME" ]; then
        echo ""
        echo -e "${YELLOW}If the error was related to PostgreSQL restart, try:${NC}"
        echo -e "  ${BLUE}docker restart ${CONTAINER_NAME}${NC}"
        echo -e "  Then re-run this script"
    fi
fi

exit $EXIT_CODE

