#!/bin/bash
# NeuronDB Docker Container Test Script
# Tests both CPU and CUDA Docker containers
#
# Usage: ./test_neurondb_docker.sh [cpu|cuda|both]
#   cpu  - Test only CPU container
#   cuda - Test only CUDA container  
#   both - Test both containers (default)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test mode
MODE="${1:-both}"

# Container configuration
CPU_CONTAINER="neurondb-cpu"
CPU_PORT=5433
CUDA_CONTAINER="neurondb-cuda"
CUDA_PORT=5434
DB_USER="neurondb"
DB_PASSWORD="neurondb"
DB_NAME="neurondb"

# Test counters (initialize to avoid arithmetic errors)
TESTS_PASSED=0
TESTS_FAILED=0

# Function to print test result
print_test() {
    local test_name="$1"
    local result="$2"
    local details="$3"
    
    if [ "$result" = "PASS" ]; then
        echo -e "${GREEN}✓${NC} $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗${NC} $test_name"
        if [ -n "$details" ]; then
            echo -e "  ${RED}Error:${NC} $details"
        fi
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# Function to test container connectivity
test_connectivity() {
    local container="$1"
    local port="$2"
    local name="$3"
    
    if sudo docker ps | grep -q "$container"; then
        if sudo docker exec "$container" pg_isready -U "$DB_USER" >/dev/null 2>&1; then
            print_test "$name container connectivity" "PASS"
            return 0
        else
            print_test "$name container connectivity" "FAIL" "Container not ready"
            return 1
        fi
    else
        print_test "$name container connectivity" "FAIL" "Container not running"
        return 1
    fi
}

# Function to test PostgreSQL connection
test_postgres_connection() {
    local port="$1"
    local name="$2"
    
    if PGPASSWORD="$DB_PASSWORD" psql -h localhost -p "$port" -U "$DB_USER" -d "$DB_NAME" -c "SELECT version();" >/dev/null 2>&1; then
        print_test "$name PostgreSQL connection" "PASS"
        return 0
    else
        print_test "$name PostgreSQL connection" "FAIL" "Cannot connect to database"
        return 1
    fi
}

# Function to test extension installation
test_extension() {
    local port="$1"
    local name="$2"
    
    local result=$(PGPASSWORD="$DB_PASSWORD" psql -h localhost -p "$port" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT extname FROM pg_extension WHERE extname = 'neurondb';" 2>&1)
    
    if echo "$result" | grep -q "neurondb"; then
        print_test "$name extension installed" "PASS"
        return 0
    else
        print_test "$name extension installed" "FAIL" "Extension not found"
        return 1
    fi
}

# Function to test extension version
test_extension_version() {
    local port="$1"
    local name="$2"
    
    local result=$(PGPASSWORD="$DB_PASSWORD" psql -h localhost -p "$port" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT neurondb.version();" 2>&1)
    
    if echo "$result" | grep -q "version"; then
        print_test "$name extension version" "PASS"
        return 0
    else
        print_test "$name extension version" "FAIL" "Cannot get version"
        return 1
    fi
}

# Function to test vector operations
test_vector_operations() {
    local port="$1"
    local name="$2"
    
    local result=$(PGPASSWORD="$DB_PASSWORD" psql -h localhost -p "$port" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT '[1.0, 2.0, 3.0]'::vector;" 2>&1)
    
    if echo "$result" | grep -q "\[1,2,3\]"; then
        print_test "$name vector operations" "PASS"
        return 0
    else
        print_test "$name vector operations" "FAIL" "Vector operation failed"
        return 1
    fi
}

# Function to test GPU functions (CUDA only)
test_gpu_functions() {
    local port="$1"
    local name="$2"
    
    # Try to get GPU device info
    local result=$(PGPASSWORD="$DB_PASSWORD" psql -h localhost -p "$port" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT neurondb.compute_mode();" 2>&1)
    
    if echo "$result" | grep -qE "[0-9]"; then
        print_test "$name GPU compute mode" "PASS"
        return 0
    else
        # GPU functions may not be available if GPU access is not configured
        print_test "$name GPU compute mode" "FAIL" "GPU functions not available (may require GPU access configuration)"
        return 1
    fi
}

# Function to test CPU container
test_cpu_container() {
    echo ""
    echo "=========================================="
    echo "Testing CPU Container"
    echo "=========================================="
    echo ""
    
    test_connectivity "$CPU_CONTAINER" "$CPU_PORT" "CPU"
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Warning:${NC} CPU container is not running. Skipping CPU tests."
        return 1
    fi
    
    test_postgres_connection "$CPU_PORT" "CPU"
    test_extension "$CPU_PORT" "CPU"
    test_extension_version "$CPU_PORT" "CPU"
    test_vector_operations "$CPU_PORT" "CPU"
}

# Function to test CUDA container
test_cuda_container() {
    echo ""
    echo "=========================================="
    echo "Testing CUDA Container"
    echo "=========================================="
    echo ""
    
    test_connectivity "$CUDA_CONTAINER" "$CUDA_PORT" "CUDA"
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Warning:${NC} CUDA container is not running. Skipping CUDA tests."
        return 1
    fi
    
    test_postgres_connection "$CUDA_PORT" "CUDA"
    test_extension "$CUDA_PORT" "CUDA"
    test_extension_version "$CUDA_PORT" "CUDA"
    test_vector_operations "$CUDA_PORT" "CUDA"
    test_gpu_functions "$CUDA_PORT" "CUDA"
}

# Main execution
echo "NeuronDB Docker Container Test Suite"
echo "======================================"
echo ""

case "$MODE" in
    cpu)
        test_cpu_container
        ;;
    cuda)
        test_cuda_container
        ;;
    both|*)
        test_cpu_container
        test_cuda_container
        ;;
esac

# Print summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi

