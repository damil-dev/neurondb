#!/bin/bash
# Comprehensive NeuronDB Docker Dependency Verification Script
# This script verifies that NeuronDB Docker setup is working and has all required dependencies

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

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
WARNINGS=0

# Function to print test result
print_test() {
    local test_name="$1"
    local result="$2"
    local details="$3"
    
    if [ "$result" = "PASS" ]; then
        echo -e "${GREEN}✓${NC} $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    elif [ "$result" = "WARN" ]; then
        echo -e "${YELLOW}⚠${NC} $test_name"
        if [ -n "$details" ]; then
            echo -e "  ${YELLOW}Warning:${NC} $details"
        fi
        WARNINGS=$((WARNINGS + 1))
    else
        echo -e "${RED}✗${NC} $test_name"
        if [ -n "$details" ]; then
            echo -e "  ${RED}Error:${NC} $details"
        fi
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# Function to check if command exists
check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

echo "=========================================="
echo "NeuronDB Docker Dependency Verification"
echo "=========================================="
echo ""

# 1. Check Docker installation
echo -e "${BLUE}1. Docker Environment Checks${NC}"
echo "----------------------------------------"

if check_command docker; then
    DOCKER_VERSION=$(docker --version 2>&1)
    print_test "Docker installed" "PASS" "$DOCKER_VERSION"
    
    if docker info &> /dev/null; then
        print_test "Docker daemon running" "PASS"
    else
        print_test "Docker daemon running" "FAIL" "Docker daemon is not running or not accessible"
    fi
else
    print_test "Docker installed" "FAIL" "Docker is not installed"
fi

if check_command docker-compose || docker compose version &> /dev/null; then
    if docker compose version &> /dev/null; then
        COMPOSE_VERSION=$(docker compose version 2>&1)
        print_test "Docker Compose installed" "PASS" "$COMPOSE_VERSION"
    else
        COMPOSE_VERSION=$(docker-compose --version 2>&1)
        print_test "Docker Compose installed" "PASS" "$COMPOSE_VERSION"
    fi
else
    print_test "Docker Compose installed" "WARN" "Docker Compose not found (docker compose may be available)"
fi

echo ""

# 2. Check Dockerfile existence and structure
echo -e "${BLUE}2. Dockerfile Verification${NC}"
echo "----------------------------------------"

DOCKERFILES=(
    "dockers/neurondb/Dockerfile.package"
    "dockers/neurondb/Dockerfile.package.cuda"
    "dockers/neurondb/Dockerfile"
)

for dockerfile in "${DOCKERFILES[@]}"; do
    if [ -f "$dockerfile" ]; then
        print_test "Dockerfile exists: $dockerfile" "PASS"
    else
        print_test "Dockerfile exists: $dockerfile" "FAIL" "File not found"
    fi
done

# Check repo-root docker-compose.yml
if [ -f "docker-compose.yml" ]; then
    print_test "docker-compose.yml exists" "PASS"
else
    print_test "docker-compose.yml exists" "FAIL" "File not found"
fi

echo ""

# 3. Verify Dockerfile dependencies
echo -e "${BLUE}3. Dockerfile Dependency Analysis${NC}"
echo "----------------------------------------"

if [ -f "dockers/neurondb/Dockerfile.package" ]; then
    echo "Analyzing dockers/neurondb/Dockerfile.package dependencies..."
    
    # Check for PostgreSQL
    if grep -q "postgres:" "dockers/neurondb/Dockerfile.package"; then
        PG_VERSION=$(grep -oP "postgres:\K[0-9]+" "dockers/neurondb/Dockerfile.package" | head -1 || echo "17")
        print_test "PostgreSQL base image" "PASS" "Version: ${PG_VERSION}"
    fi
    
    # Check for ONNX Runtime
    if grep -q "ONNX_VERSION" "dockers/neurondb/Dockerfile.package"; then
        ONNX_VERSION=$(grep -oP "ONNX_VERSION=\K[0-9.]+" "dockers/neurondb/Dockerfile.package" | head -1 || echo "1.17.0")
        print_test "ONNX Runtime dependency" "PASS" "Version: ${ONNX_VERSION}"
    fi
    
    # Check for ML libraries
    if grep -q "XGBoost" "dockers/neurondb/Dockerfile.package"; then
        print_test "XGBoost ML library" "PASS" "Included in Dockerfile"
    fi
    
    if grep -q "LightGBM" "dockers/neurondb/Dockerfile.package"; then
        print_test "LightGBM ML library" "PASS" "Included in Dockerfile"
    fi
    
    if grep -q "CatBoost" "dockers/neurondb/Dockerfile.package"; then
        print_test "CatBoost ML library" "PASS" "Included in Dockerfile"
    fi
    
    # Check for build dependencies
    BUILD_DEPS=("build-essential" "cmake" "git" "libcurl4-openssl-dev" "libssl-dev" "zlib1g-dev" "libomp-dev" "libeigen3-dev")
    for dep in "${BUILD_DEPS[@]}"; do
        if grep -q "$dep" "dockers/neurondb/Dockerfile.package"; then
            print_test "Build dependency: $dep" "PASS"
        else
            print_test "Build dependency: $dep" "WARN" "Not explicitly listed (may be in base image)"
        fi
    done
fi

echo ""

# 4. Check if containers exist and their status
echo -e "${BLUE}4. Container Status Check${NC}"
echo "----------------------------------------"

CONTAINERS=("neurondb-cpu" "neurondb-cuda")
for container in "${CONTAINERS[@]}"; do
    if docker ps -a --format "{{.Names}}" | grep -q "^${container}$"; then
        STATUS=$(docker inspect --format='{{.State.Status}}' "$container" 2>/dev/null || echo "unknown")
        if [ "$STATUS" = "running" ]; then
            print_test "Container: $container" "PASS" "Status: $STATUS"
        else
            print_test "Container: $container" "WARN" "Status: $STATUS (not running)"
        fi
    else
        print_test "Container: $container" "WARN" "Container does not exist"
    fi
done

echo ""

# 5. Verify Docker images
echo -e "${BLUE}5. Docker Image Verification${NC}"
echo "----------------------------------------"

IMAGES=("neurondb:cpu-pg17" "neurondb:cuda-pg17")
for image in "${IMAGES[@]}"; do
    if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${image}$"; then
        IMAGE_SIZE=$(docker images --format "{{.Size}}" "$image" 2>/dev/null | head -1)
        print_test "Docker image: $image" "PASS" "Size: $IMAGE_SIZE"
    else
        print_test "Docker image: $image" "WARN" "Image not built yet"
    fi
done

echo ""

# 6. Test container functionality (if running)
echo -e "${BLUE}6. Container Functionality Test${NC}"
echo "----------------------------------------"

if docker ps --format "{{.Names}}" | grep -q "^neurondb-cpu$"; then
    echo "Testing neurondb-cpu container..."
    
    # Test PostgreSQL connection
    if docker exec neurondb-cpu pg_isready -U neurondb &> /dev/null; then
        print_test "PostgreSQL ready in container" "PASS"
    else
        print_test "PostgreSQL ready in container" "FAIL" "PostgreSQL not responding"
    fi
    
    # Test extension installation
    if docker exec neurondb-cpu psql -U neurondb -d neurondb -tAc "SELECT 1 FROM pg_extension WHERE extname = 'neurondb';" 2>/dev/null | grep -q 1; then
        print_test "NeuronDB extension installed" "PASS"
    else
        print_test "NeuronDB extension installed" "WARN" "Extension may not be created yet"
    fi
    
    # Check for required libraries in container
    echo "Checking for required libraries in container..."
    
    LIBRARIES=("libxgboost" "liblightgbm" "libcatboost" "libonnxruntime" "libomp" "libeigen")
    for lib in "${LIBRARIES[@]}"; do
        if docker exec neurondb-cpu find /usr/local/lib -name "*${lib}*" -o -name "*${lib}*.so*" 2>/dev/null | grep -q .; then
            print_test "Library: $lib" "PASS"
        else
            # Check system libs too
            if docker exec neurondb-cpu find /usr/lib -name "*${lib}*" 2>/dev/null | grep -q .; then
                print_test "Library: $lib" "PASS" "Found in /usr/lib"
            else
                print_test "Library: $lib" "WARN" "Not found (may be optional or named differently)"
            fi
        fi
    done
    
    # Check ONNX Runtime
    if docker exec neurondb-cpu test -d /usr/local/onnxruntime 2>/dev/null; then
        ONNX_FILES=$(docker exec neurondb-cpu find /usr/local/onnxruntime/lib -name "*.so*" 2>/dev/null | wc -l)
        print_test "ONNX Runtime installation" "PASS" "Found $ONNX_FILES library files"
    else
        print_test "ONNX Runtime installation" "FAIL" "ONNX Runtime directory not found"
    fi
    
    # Check for neurondb.so
    if docker exec neurondb-cpu find /usr/lib/postgresql -name "neurondb.so" 2>/dev/null | grep -q .; then
        NEURONDB_SO=$(docker exec neurondb-cpu find /usr/lib/postgresql -name "neurondb.so" 2>/dev/null | head -1)
        print_test "neurondb.so extension" "PASS" "Found at: $NEURONDB_SO"
    else
        print_test "neurondb.so extension" "FAIL" "Extension library not found"
    fi
    
    # Test vector operations
    if docker exec neurondb-cpu psql -U neurondb -d neurondb -tAc "SELECT '[1.0, 2.0, 3.0]'::vector;" 2>/dev/null | grep -q "\[1,2,3\]"; then
        print_test "Vector operations" "PASS"
    else
        print_test "Vector operations" "WARN" "Vector type may not be available"
    fi
else
    print_test "Container functionality test" "WARN" "Container not running - start it with: docker compose up -d neurondb"
fi

echo ""

# 7. Check docker-compose configuration
echo -e "${BLUE}7. Docker Compose Configuration${NC}"
echo "----------------------------------------"

if [ -f "docker-compose.yml" ]; then
    # Check if docker-compose can parse the file
    if docker compose config &> /dev/null || docker-compose config &> /dev/null; then
        print_test "docker-compose.yml syntax" "PASS"
        
        # Check for required services
        SERVICES=("neurondb" "neurondb-cuda")
        for service in "${SERVICES[@]}"; do
            if grep -q "^  ${service}:" docker-compose.yml; then
                print_test "Service defined: $service" "PASS"
            else
                print_test "Service defined: $service" "WARN" "Service not found in docker-compose.yml"
            fi
        done
        
        # Check for health checks
        if grep -q "healthcheck:" docker-compose.yml; then
            print_test "Health checks configured" "PASS"
        else
            print_test "Health checks configured" "WARN" "Health checks may not be configured"
        fi
        
        # Check for networks
        if grep -q "neurondb-network" docker-compose.yml; then
            print_test "Network configuration" "PASS"
        else
            print_test "Network configuration" "WARN" "Network may not be configured"
        fi
    else
        print_test "docker-compose.yml syntax" "FAIL" "Configuration file has syntax errors"
    fi
fi

echo ""

# 8. Check for required files and scripts
echo -e "${BLUE}8. Required Files and Scripts${NC}"
echo "----------------------------------------"

REQUIRED_FILES=(
    "dockers/neurondb/docker-entrypoint-initdb.d/10_configure_neurondb.sh"
    "dockers/neurondb/docker-entrypoint-initdb.d/30_ensure_neurondb_extension.sh"
    "dockers/neurondb/docker-entrypoint-neurondb.sh"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        if [ -x "$file" ] || [[ "$file" == *.sh ]]; then
            print_test "Required file: $file" "PASS"
        else
            print_test "Required file: $file" "PASS" "File exists (executable check skipped)"
        fi
    else
        print_test "Required file: $file" "FAIL" "File not found"
    fi
done

echo ""

# 9. Summary
echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    if [ $WARNINGS -eq 0 ]; then
        echo -e "${GREEN}✓ All checks passed! NeuronDB Docker setup is ready.${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠ All critical checks passed, but there are some warnings.${NC}"
        echo -e "${YELLOW}  Review the warnings above and address them if needed.${NC}"
        exit 0
    fi
else
    echo -e "${RED}✗ Some checks failed. Please review the errors above.${NC}"
    exit 1
fi






