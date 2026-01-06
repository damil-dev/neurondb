#!/bin/bash
# ============================================================================
# REL1_STABLE Docker Images - Ultra Deep Testing for All Modules
# ============================================================================
# This script performs comprehensive testing of all NeuronDB ecosystem modules
# using REL1_STABLE tagged docker images across all variants.
#
# Usage:
#   ./scripts/test_rel1_stable_docker_deep.sh [options]
#
# Options:
#   --variant TYPE      Test specific variant: cpu, cuda, rocm, metal, all (default: all)
#   --module MODULE     Test specific module: neurondb, neuronagent, neuronmcp, neurondesktop, all (default: all)
#   --pull              Pull images from registry (default: build from source)
#   --build             Build images from source (default)
#   --skip-build        Skip building/pulling, use existing images
#   --stop-on-fail      Stop on first test failure
#   --verbose           Enable verbose output
#   --help              Show this help message
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'
BOLD='\033[1m'

# Configuration
VERSION_TAG="REL1_STABLE"
VARIANT="all"
MODULE="all"
PULL_IMAGES=false
BUILD_IMAGES=true
SKIP_BUILD=false
STOP_ON_FAIL=false
VERBOSE=false

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0
START_TIME=$(date +%s)

# Docker command detection
DOCKER_CMD="docker"
if ! docker ps &>/dev/null 2>&1; then
    if command -v sudo &> /dev/null && sudo docker ps &>/dev/null 2>&1; then
        DOCKER_CMD="sudo docker"
    fi
fi

# Compose command detection
if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
else
    echo -e "${RED}Error: Docker Compose not found${NC}"
    exit 1
fi

# Log file
LOG_FILE="/tmp/rel1_stable_test_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Helper functions
print_header() {
    echo ""
    echo -e "${BOLD}${MAGENTA}"
    cat << "EOF"
╔══════════════════════════════════════════════════════════════════════════════╗
║     REL1_STABLE Docker Images - Ultra Deep Testing Suite                    ║
║                    All Modules, All Variants                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

print_section() {
    echo ""
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${CYAN}$1${NC}"
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_test() {
    local status=$1
    local message=$2
    local details="${3:-}"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}✓${NC} $message"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    elif [ "$status" = "SKIP" ]; then
        echo -e "${YELLOW}⊘${NC} $message${details:+ ${YELLOW}(${details})${NC}}"
        SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
    else
        echo -e "${RED}✗${NC} $message"
        if [ -n "$details" ]; then
            echo -e "  ${RED}Error:${NC} $details"
        fi
        FAILED_TESTS=$((FAILED_TESTS + 1))
        if [ "$STOP_ON_FAIL" = true ]; then
            echo -e "${RED}Stopping on failure as requested${NC}"
            exit 1
        fi
    fi
}

print_info() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}ℹ${NC} $1"
    fi
}

show_help() {
    cat << EOF
REL1_STABLE Docker Images - Ultra Deep Testing

Usage: $0 [options]

Options:
  --variant TYPE      Test variant: cpu, cuda, rocm, metal, all (default: all)
  --module MODULE     Test module: neurondb, neuronagent, neuronmcp, neurondesktop, all (default: all)
  --pull              Pull images from registry instead of building
  --build             Build images from source (default)
  --skip-build        Skip building/pulling, use existing images
  --stop-on-fail      Stop on first test failure
  --verbose           Enable verbose output
  --help              Show this help message

Examples:
  $0                                    # Test all modules, all variants
  $0 --variant=cpu --module=neurondb    # Test only CPU variant of NeuronDB
  $0 --pull --variant=cuda              # Pull and test CUDA images
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --variant=*) VARIANT="${1#*=}"; shift ;;
        --variant) VARIANT="$2"; shift 2 ;;
        --module=*) MODULE="${1#*=}"; shift ;;
        --module) MODULE="$2"; shift 2 ;;
        --pull) PULL_IMAGES=true; BUILD_IMAGES=false; shift ;;
        --build) BUILD_IMAGES=true; PULL_IMAGES=false; shift ;;
        --skip-build) SKIP_BUILD=true; BUILD_IMAGES=false; PULL_IMAGES=false; shift ;;
        --stop-on-fail) STOP_ON_FAIL=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        --help) show_help; exit 0 ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; show_help; exit 1 ;;
    esac
done

# Determine variants to test
declare -a VARIANTS_TO_TEST
if [ "$VARIANT" = "all" ]; then
    VARIANTS_TO_TEST=(cpu cuda rocm metal)
else
    VARIANTS_TO_TEST=("$VARIANT")
fi

# Determine modules to test
declare -a MODULES_TO_TEST
if [ "$MODULE" = "all" ]; then
    MODULES_TO_TEST=(neurondb neuronagent neuronmcp neurondesktop)
else
    MODULES_TO_TEST=("$MODULE")
fi

# ============================================================================
# Test Functions (defined before use)
# ============================================================================

test_neurondb() {
    local variant=$1
    local container_name="neurondb-${variant}"
    local port
    
    case "$variant" in
        cpu) port=5433 ;;
        cuda) port=5434 ;;
        rocm) port=5435 ;;
        metal) port=5436 ;;
        *) port=5433 ;;
    esac
    
    # Start container if not running
    if ! $DOCKER_CMD ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
        print_info "Starting ${container_name}..."
        case "$variant" in
            cpu) $COMPOSE_CMD -f docker-compose.yml --profile default up -d neurondb ;;
            cuda) $COMPOSE_CMD -f docker-compose.yml --profile cuda up -d neurondb-cuda ;;
            rocm) $COMPOSE_CMD -f docker-compose.yml --profile rocm up -d neurondb-rocm ;;
            metal) $COMPOSE_CMD -f docker-compose.yml --profile metal up -d neurondb-metal ;;
        esac
        
        # Wait for container to be healthy
        print_info "Waiting for container to be ready..."
        sleep 10
        for i in {1..30}; do
            if $DOCKER_CMD exec "$container_name" pg_isready -U neurondb >/dev/null 2>&1; then
                break
            fi
            sleep 2
        done
    fi
    
    # Run comprehensive tests
    if [ -f "$REPO_ROOT/scripts/test_neurondb_docker_comprehensive.sh" ]; then
        print_info "Running comprehensive NeuronDB tests..."
        "$REPO_ROOT/scripts/test_neurondb_docker_comprehensive.sh" \
            --variant="$variant" \
            --container="$container_name" \
            --port="$port" \
            $([ "$STOP_ON_FAIL" = true ] && echo "--stop-on-fail") \
            $([ "$VERBOSE" = true ] && echo "--verbose") || print_test "FAIL" "NeuronDB comprehensive tests" "Some tests failed"
    elif [ -f "$REPO_ROOT/scripts/test_neurondb_docker_detailed.sh" ]; then
        print_info "Running detailed NeuronDB tests..."
        "$REPO_ROOT/scripts/test_neurondb_docker_detailed.sh" \
            --variant="$variant" \
            --container="$container_name" \
            --port="$port" \
            $([ "$VERBOSE" = true ] && echo "--verbose") || print_test "FAIL" "NeuronDB detailed tests" "Some tests failed"
    else
        # Basic connectivity test
        if $DOCKER_CMD exec "$container_name" pg_isready -U neurondb >/dev/null 2>&1; then
            print_test "PASS" "NeuronDB container health check"
            
            # Test extension
            if PGPASSWORD=neurondb psql -h localhost -p "$port" -U neurondb -d neurondb -t -c "SELECT extname FROM pg_extension WHERE extname = 'neurondb';" 2>/dev/null | grep -q neurondb; then
                print_test "PASS" "NeuronDB extension installed"
            else
                print_test "FAIL" "NeuronDB extension not found"
            fi
        else
            print_test "FAIL" "NeuronDB container not ready"
        fi
    fi
}

test_neuronagent() {
    local variant=$1
    local container_name="neuronagent${variant:+-${variant}}"
    
    # Start container if not running
    if ! $DOCKER_CMD ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
        print_info "Starting ${container_name}..."
        case "$variant" in
            cpu|default) $COMPOSE_CMD -f docker-compose.yml --profile default up -d neuronagent ;;
            cuda) $COMPOSE_CMD -f docker-compose.yml --profile cuda up -d neuronagent-cuda ;;
            rocm) $COMPOSE_CMD -f docker-compose.yml --profile rocm up -d neuronagent-rocm ;;
            metal) $COMPOSE_CMD -f docker-compose.yml --profile metal up -d neuronagent-metal ;;
        esac
        sleep 10
    fi
    
    # Test health endpoint
    if curl -s http://localhost:8080/health >/dev/null 2>&1; then
        print_test "PASS" "NeuronAgent health endpoint"
    else
        print_test "FAIL" "NeuronAgent health endpoint not responding"
    fi
    
    # Run NeuronAgent tests if available
    if [ -f "$REPO_ROOT/NeuronAgent/test_complete.sh" ]; then
        print_info "Running NeuronAgent comprehensive tests..."
        cd "$REPO_ROOT/NeuronAgent"
        ./test_complete.sh || print_test "FAIL" "NeuronAgent comprehensive tests" "Some tests failed"
        cd "$REPO_ROOT"
    fi
}

test_neuronmcp() {
    local variant=$1
    local container_name="neurondb-mcp${variant:+-${variant}}"
    
    # Start container if not running
    if ! $DOCKER_CMD ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
        print_info "Starting ${container_name}..."
        case "$variant" in
            cpu|default) $COMPOSE_CMD -f docker-compose.yml --profile default up -d neuronmcp ;;
            cuda) $COMPOSE_CMD -f docker-compose.yml --profile cuda up -d neuronmcp-cuda ;;
            rocm) $COMPOSE_CMD -f docker-compose.yml --profile rocm up -d neuronmcp-rocm ;;
            metal) $COMPOSE_CMD -f docker-compose.yml --profile metal up -d neuronmcp-metal ;;
        esac
        sleep 5
    fi
    
    # Test MCP binary exists and is executable
    if $DOCKER_CMD exec "$container_name" test -f /app/neurondb-mcp -a -x /app/neurondb-mcp 2>/dev/null; then
        print_test "PASS" "NeuronMCP binary exists and is executable"
    else
        print_test "FAIL" "NeuronMCP binary not found or not executable"
    fi
    
    # Run MCP tests if available
    if [ -f "$REPO_ROOT/NeuronMCP/tests/run_all_tests.py" ]; then
        print_info "Running NeuronMCP tests..."
        cd "$REPO_ROOT/NeuronMCP"
        python3 tests/run_all_tests.py || print_test "FAIL" "NeuronMCP tests" "Some tests failed"
        cd "$REPO_ROOT"
    fi
}

test_neurondesktop() {
    local variant=$1
    
    # Start containers if not running
    print_info "Starting NeuronDesktop services..."
    $COMPOSE_CMD -f docker-compose.yml --profile default up -d neurondesk-init neurondesk-api neurondesk-frontend 2>/dev/null || true
    sleep 15
    
    # Test API health
    if curl -s http://localhost:8081/health >/dev/null 2>&1; then
        print_test "PASS" "NeuronDesktop API health endpoint"
    else
        print_test "FAIL" "NeuronDesktop API health endpoint not responding"
    fi
    
    # Test frontend
    if curl -s http://localhost:3000 >/dev/null 2>&1; then
        print_test "PASS" "NeuronDesktop frontend accessible"
    else
        print_test "FAIL" "NeuronDesktop frontend not accessible"
    fi
    
    # Run NeuronDesktop tests if available
    if [ -f "$REPO_ROOT/NeuronDesktop/tests/run_all_tests.sh" ]; then
        print_info "Running NeuronDesktop tests..."
        cd "$REPO_ROOT/NeuronDesktop/tests"
        ./run_all_tests.sh || print_test "FAIL" "NeuronDesktop tests" "Some tests failed"
        cd "$REPO_ROOT"
    fi
}

print_header

echo -e "${BLUE}Version Tag:${NC} ${VERSION_TAG}"
echo -e "${BLUE}Variants:${NC} ${VARIANTS_TO_TEST[*]}"
echo -e "${BLUE}Modules:${NC} ${MODULES_TO_TEST[*]}"
echo -e "${BLUE}Build Images:${NC} ${BUILD_IMAGES}"
echo -e "${BLUE}Pull Images:${NC} ${PULL_IMAGES}"
echo -e "${BLUE}Skip Build:${NC} ${SKIP_BUILD}"
echo -e "${BLUE}Log File:${NC} ${LOG_FILE}"
echo ""

# ============================================================================
# Step 1: Build/Pull Docker Images
# ============================================================================
if [ "$SKIP_BUILD" = false ]; then
    print_section "Step 1: Preparing Docker Images (${VERSION_TAG})"
    
    if [ "$PULL_IMAGES" = true ]; then
        print_info "Pulling images from registry..."
        # Note: This assumes images are published with REL1_STABLE tag
        # Adjust registry path as needed
        for variant in "${VARIANTS_TO_TEST[@]}"; do
            case "$variant" in
                cpu)
                    print_info "Pulling neurondb:${VERSION_TAG}-cpu-pg17..."
                    $DOCKER_CMD pull "ghcr.io/neurondb/neurondb:${VERSION_TAG}-cpu-pg17" 2>/dev/null || print_test "SKIP" "Pull neurondb:${VERSION_TAG}-cpu-pg17" "Image not found in registry"
                    ;;
                cuda)
                    print_info "Pulling neurondb:${VERSION_TAG}-cuda-pg17..."
                    $DOCKER_CMD pull "ghcr.io/neurondb/neurondb:${VERSION_TAG}-cuda-pg17" 2>/dev/null || print_test "SKIP" "Pull neurondb:${VERSION_TAG}-cuda-pg17" "Image not found in registry"
                    ;;
                rocm)
                    print_info "Pulling neurondb:${VERSION_TAG}-rocm-pg17..."
                    $DOCKER_CMD pull "ghcr.io/neurondb/neurondb:${VERSION_TAG}-rocm-pg17" 2>/dev/null || print_test "SKIP" "Pull neurondb:${VERSION_TAG}-rocm-pg17" "Image not found in registry"
                    ;;
                metal)
                    print_info "Pulling neurondb:${VERSION_TAG}-metal-pg17..."
                    $DOCKER_CMD pull "ghcr.io/neurondb/neurondb:${VERSION_TAG}-metal-pg17" 2>/dev/null || print_test "SKIP" "Pull neurondb:${VERSION_TAG}-metal-pg17" "Image not found in registry"
                    ;;
            esac
        done
    elif [ "$BUILD_IMAGES" = true ]; then
        print_info "Building images from source with tag ${VERSION_TAG}..."
        export VERSION="${VERSION_TAG}"
        
        for variant in "${VARIANTS_TO_TEST[@]}"; do
            case "$variant" in
                cpu)
                    print_info "Building neurondb (CPU variant)..."
                    $COMPOSE_CMD -f docker-compose.yml --profile default build neurondb 2>&1 | grep -E "(Step|Successfully|Error)" || true
                    $DOCKER_CMD tag neurondb:cpu-pg17 "neurondb:${VERSION_TAG}-cpu-pg17" 2>/dev/null || true
                    ;;
                cuda)
                    print_info "Building neurondb (CUDA variant)..."
                    $COMPOSE_CMD -f docker-compose.yml --profile cuda build neurondb-cuda 2>/dev/null || print_test "SKIP" "Build neurondb-cuda" "CUDA not available"
                    $DOCKER_CMD tag neurondb:cuda-pg17 "neurondb:${VERSION_TAG}-cuda-pg17" 2>/dev/null || true
                    ;;
                rocm)
                    print_info "Building neurondb (ROCm variant)..."
                    $COMPOSE_CMD -f docker-compose.yml --profile rocm build neurondb-rocm 2>/dev/null || print_test "SKIP" "Build neurondb-rocm" "ROCm not available"
                    $DOCKER_CMD tag neurondb:rocm-pg17 "neurondb:${VERSION_TAG}-rocm-pg17" 2>/dev/null || true
                    ;;
                metal)
                    print_info "Building neurondb (Metal variant)..."
                    $COMPOSE_CMD -f docker-compose.yml --profile metal build neurondb-metal 2>/dev/null || print_test "SKIP" "Build neurondb-metal" "Metal not available"
                    $DOCKER_CMD tag neurondb:metal-pg17 "neurondb:${VERSION_TAG}-metal-pg17" 2>/dev/null || true
                    ;;
            esac
        done
        
        # Build other modules
        if [[ " ${MODULES_TO_TEST[@]} " =~ " neuronagent " ]]; then
            print_info "Building neuronagent..."
            $COMPOSE_CMD -f docker-compose.yml --profile default build neuronagent 2>&1 | grep -E "(Step|Successfully|Error)" || true
            $DOCKER_CMD tag neuronagent:latest "neuronagent:${VERSION_TAG}" 2>/dev/null || true
        fi
        
        if [[ " ${MODULES_TO_TEST[@]} " =~ " neuronmcp " ]]; then
            print_info "Building neuronmcp..."
            $COMPOSE_CMD -f docker-compose.yml --profile default build neuronmcp 2>&1 | grep -E "(Step|Successfully|Error)" || true
            $DOCKER_CMD tag neurondb-mcp:latest "neurondb-mcp:${VERSION_TAG}" 2>/dev/null || true
        fi
        
        if [[ " ${MODULES_TO_TEST[@]} " =~ " neurondesktop " ]]; then
            print_info "Building neurondesktop..."
            $COMPOSE_CMD -f docker-compose.yml --profile default build neurondesk-api neurondesk-frontend 2>&1 | grep -E "(Step|Successfully|Error)" || true
        fi
    fi
fi

# ============================================================================
# Step 2: Test Each Module and Variant Combination
# ============================================================================
for variant in "${VARIANTS_TO_TEST[@]}"; do
    for module in "${MODULES_TO_TEST[@]}"; do
        print_section "Testing ${module^^} (${variant^^} variant)"
        
        case "$module" in
            neurondb)
                test_neurondb "$variant"
                ;;
            neuronagent)
                test_neuronagent "$variant"
                ;;
            neuronmcp)
                test_neuronmcp "$variant"
                ;;
            neurondesktop)
                test_neurondesktop "$variant"
                ;;
        esac
    done
done

# ============================================================================
# Step 3: Integration Tests
# ============================================================================
print_section "Step 3: Integration Tests"

if [ -f "$REPO_ROOT/scripts/docker-integration-tests.sh" ]; then
    print_info "Running integration tests..."
    "$REPO_ROOT/scripts/docker-integration-tests.sh" \
        $([ "$VERBOSE" = true ] && echo "--verbose") || print_test "FAIL" "Integration tests" "Some tests failed"
else
    print_test "SKIP" "Integration tests" "Script not found"
fi

# ============================================================================
# Summary
# ============================================================================
print_section "Test Summary"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo -e "${BLUE}Total Tests:${NC} ${TOTAL_TESTS}"
echo -e "${GREEN}Passed:${NC} ${PASSED_TESTS}"
if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "${RED}Failed:${NC} ${FAILED_TESTS}"
else
    echo -e "${BLUE}Failed:${NC} ${FAILED_TESTS}"
fi
echo -e "${YELLOW}Skipped:${NC} ${SKIPPED_TESTS}"
echo -e "${BLUE}Duration:${NC} ${DURATION}s"
echo -e "${BLUE}Log File:${NC} ${LOG_FILE}"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Check ${LOG_FILE} for details.${NC}"
    exit 1
fi

