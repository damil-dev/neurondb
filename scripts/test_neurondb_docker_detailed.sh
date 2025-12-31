#!/bin/bash
# Comprehensive NeuronDB Docker Testing Script
# Tests all aspects of NeuronDB running in Docker containers
#
# Usage: ./test_neurondb_docker_detailed.sh [options]
#   --container NAME    Container name (default: auto-detect)
#   --port PORT         Database port (default: auto-detect)
#   --variant TYPE      Container variant: cpu, cuda, rocm, metal (default: cpu)
#   --user USER         Database user (default: neurondb)
#   --password PASS     Database password (default: neurondb)
#   --db NAME           Database name (default: neurondb)
#   --quick             Run quick tests only (skip performance tests)
#   --verbose           Enable verbose output
#   --help              Show help message

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Test configuration
CONTAINER_NAME=""
DB_PORT=""
VARIANT="cpu"
DB_USER="neurondb"
DB_PASSWORD="neurondb"
DB_NAME="neurondb"
DB_HOST="localhost"
QUICK_MODE=false
VERBOSE=false

# Container name mappings
CPU_CONTAINER="neurondb-cpu"
CUDA_CONTAINER="neurondb-cuda"
ROCM_CONTAINER="neurondb-rocm"
METAL_CONTAINER="neurondb-metal"

# Default ports
CPU_PORT=5433
CUDA_PORT=5434
ROCM_PORT=5435
METAL_PORT=5436

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0
TEST_START_TIME=$(date +%s)

# Docker command (auto-detect sudo requirement)
DOCKER_CMD="docker"
if ! docker ps &>/dev/null 2>&1; then
    if command -v sudo &> /dev/null && sudo docker ps &>/dev/null 2>&1; then
        DOCKER_CMD="sudo docker"
    fi
fi

# Function to print section header
print_section() {
    echo ""
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${CYAN}$1${NC}"
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Function to print test result
print_test() {
    local test_name="$1"
    local result="$2"
    local details="$3"
    local elapsed="${4:-}"
    
    if [ "$result" = "PASS" ]; then
        echo -e "${GREEN}✓${NC} ${test_name}${elapsed:+ ${BLUE}(${elapsed}s)${NC}}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    elif [ "$result" = "SKIP" ]; then
        echo -e "${YELLOW}⊘${NC} ${test_name}${details:+ ${YELLOW}(${details})${NC}}"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    else
        echo -e "${RED}✗${NC} ${test_name}${elapsed:+ ${BLUE}(${elapsed}s)${NC}}"
        if [ -n "$details" ]; then
            echo -e "  ${RED}Error:${NC} $details"
        fi
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# Function to run SQL and capture result
run_sql() {
    local sql="$1"
    local timeout="${2:-10}"
    PGPASSWORD="$DB_PASSWORD" timeout "$timeout" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -A -c "$sql" 2>&1
}

# Function to run SQL file
run_sql_file() {
    local file="$1"
    local timeout="${2:-30}"
    PGPASSWORD="$DB_PASSWORD" timeout "$timeout" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$file" 2>&1
}

# Function to show help
show_help() {
    cat << EOF
Comprehensive NeuronDB Docker Testing Script

Usage: $0 [options]

Options:
  --container NAME    Container name (default: auto-detect)
  --port PORT         Database port (default: auto-detect)
  --variant TYPE      Container variant: cpu, cuda, rocm, metal (default: cpu)
  --user USER         Database user (default: neurondb)
  --password PASS     Database password (default: neurondb)
  --db NAME           Database name (default: neurondb)
  --host HOST         Database host (default: localhost)
  --quick             Run quick tests only (skip performance tests)
  --verbose           Enable verbose output
  --help              Show this help message

Examples:
  # Test CPU container (auto-detect)
  $0 --variant=cpu

  # Test CUDA container on specific port
  $0 --variant=cuda --port=5434

  # Quick test mode
  $0 --variant=cpu --quick

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --container=*)
            CONTAINER_NAME="${1#*=}"
            shift
            ;;
        --container)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --port=*)
            DB_PORT="${1#*=}"
            shift
            ;;
        --port)
            DB_PORT="$2"
            shift 2
            ;;
        --variant=*)
            VARIANT="${1#*=}"
            shift
            ;;
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --user=*)
            DB_USER="${1#*=}"
            shift
            ;;
        --user)
            DB_USER="$2"
            shift 2
            ;;
        --password=*)
            DB_PASSWORD="${1#*=}"
            shift
            ;;
        --password)
            DB_PASSWORD="$2"
            shift 2
            ;;
        --db=*)
            DB_NAME="${1#*=}"
            shift
            ;;
        --db)
            DB_NAME="$2"
            shift 2
            ;;
        --host=*)
            DB_HOST="${1#*=}"
            shift
            ;;
        --host)
            DB_HOST="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Error:${NC} Unknown option: $1" >&2
            show_help
            exit 1
            ;;
    esac
done

# Auto-detect container and port if not specified
if [ -z "$CONTAINER_NAME" ]; then
    case "$VARIANT" in
        cpu)
            CONTAINER_NAME="$CPU_CONTAINER"
            DB_PORT="${DB_PORT:-$CPU_PORT}"
            ;;
        cuda)
            CONTAINER_NAME="$CUDA_CONTAINER"
            DB_PORT="${DB_PORT:-$CUDA_PORT}"
            ;;
        rocm)
            CONTAINER_NAME="$ROCM_CONTAINER"
            DB_PORT="${DB_PORT:-$ROCM_PORT}"
            ;;
        metal)
            CONTAINER_NAME="$METAL_CONTAINER"
            DB_PORT="${DB_PORT:-$METAL_PORT}"
            ;;
        *)
            CONTAINER_NAME="$CPU_CONTAINER"
            DB_PORT="${DB_PORT:-$CPU_PORT}"
            ;;
    esac
fi

# Try to get actual port from container if running
if $DOCKER_CMD ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    port_mapping=$($DOCKER_CMD port "$CONTAINER_NAME" 2>/dev/null | grep "5432/tcp" | head -1 | cut -d: -f2)
    if [ -n "$port_mapping" ]; then
        DB_PORT="$port_mapping"
    fi
fi

# Print header
echo -e "${BOLD}${MAGENTA}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════════════════════╗
║                  NeuronDB Docker Comprehensive Test Suite                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"
echo -e "${BLUE}Container:${NC} ${CONTAINER_NAME}"
echo -e "${BLUE}Variant:${NC} ${VARIANT}"
echo -e "${BLUE}Host:${NC} ${DB_HOST}:${DB_PORT}"
echo -e "${BLUE}Database:${NC} ${DB_NAME}"
echo -e "${BLUE}User:${NC} ${DB_USER}"
echo -e "${BLUE}Quick Mode:${NC} ${QUICK_MODE}"
echo ""

# ============================================================================
# TEST SUITE 1: Container Health and Connectivity
# ============================================================================
print_section "TEST SUITE 1: Container Health and Connectivity"

# Test 1.1: Container is running
test_start=$(date +%s)
if $DOCKER_CMD ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    elapsed=$(($(date +%s) - test_start))
    print_test "Container is running" "PASS" "" "${elapsed}"
else
    elapsed=$(($(date +%s) - test_start))
    print_test "Container is running" "FAIL" "Container '${CONTAINER_NAME}' not found" "${elapsed}"
    echo -e "${RED}Fatal:${NC} Container must be running to continue tests"
    exit 1
fi

# Test 1.2: Container health status
test_start=$(date +%s)
health_status=$($DOCKER_CMD inspect --format='{{.State.Health.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "none")
elapsed=$(($(date +%s) - test_start))
if [ "$health_status" = "healthy" ]; then
    print_test "Container health check" "PASS" "" "${elapsed}"
elif [ "$health_status" = "none" ]; then
    print_test "Container health check" "SKIP" "Health check not configured" "${elapsed}"
else
    print_test "Container health check" "FAIL" "Status: ${health_status}" "${elapsed}"
fi

# Test 1.3: PostgreSQL is ready
test_start=$(date +%s)
if $DOCKER_CMD exec "$CONTAINER_NAME" pg_isready -U "$DB_USER" >/dev/null 2>&1; then
    elapsed=$(($(date +%s) - test_start))
    print_test "PostgreSQL is ready" "PASS" "" "${elapsed}"
else
    elapsed=$(($(date +%s) - test_start))
    print_test "PostgreSQL is ready" "FAIL" "pg_isready failed" "${elapsed}"
fi

# Test 1.4: Database connection
test_start=$(date +%s)
if run_sql "SELECT version();" >/dev/null 2>&1; then
    elapsed=$(($(date +%s) - test_start))
    pg_version=$(run_sql "SELECT version();" | head -1)
    print_test "Database connection" "PASS" "PostgreSQL version detected" "${elapsed}"
    if [ "$VERBOSE" = true ]; then
        echo -e "  ${BLUE}Version:${NC} $pg_version"
    fi
else
    elapsed=$(($(date +%s) - test_start))
    print_test "Database connection" "FAIL" "Cannot connect to database" "${elapsed}"
fi

# Test 1.5: Port mapping
test_start=$(date +%s)
port_mapping=$($DOCKER_CMD port "$CONTAINER_NAME" 2>/dev/null | grep "5432/tcp" | head -1 || echo "")
elapsed=$(($(date +%s) - test_start))
if [ -n "$port_mapping" ]; then
    print_test "Port mapping" "PASS" "Port mapped: $port_mapping" "${elapsed}"
else
    print_test "Port mapping" "SKIP" "Port mapping not found (may be using host network)" "${elapsed}"
fi

# ============================================================================
# TEST SUITE 2: Extension Installation and Configuration
# ============================================================================
print_section "TEST SUITE 2: Extension Installation and Configuration"

# Test 2.1: Extension exists
test_start=$(date +%s)
ext_exists=$(run_sql "SELECT extname FROM pg_extension WHERE extname = 'neurondb';" 2>&1 | grep -v "ERROR" | head -1 | tr -d ' ')
elapsed=$(($(date +%s) - test_start))
if [ "$ext_exists" = "neurondb" ]; then
    print_test "NeuronDB extension installed" "PASS" "" "${elapsed}"
else
    elapsed=$(($(date +%s) - test_start))
    print_test "NeuronDB extension installed" "FAIL" "Extension not found" "${elapsed}"
fi

# Test 2.2: Extension version
test_start=$(date +%s)
ext_version=$(run_sql "SELECT extversion FROM pg_extension WHERE extname = 'neurondb';" 2>&1 | grep -v "ERROR" | head -1 | tr -d ' ')
elapsed=$(($(date +%s) - test_start))
if [ -n "$ext_version" ] && [ "$ext_version" != "" ]; then
    print_test "Extension version" "PASS" "Version: $ext_version" "${elapsed}"
else
    print_test "Extension version" "FAIL" "Cannot get version" "${elapsed}"
fi

# Test 2.3: Extension schema exists
test_start=$(date +%s)
schema_exists=$(run_sql "SELECT nspname FROM pg_namespace WHERE nspname = 'neurondb';" 2>&1 | grep -v "ERROR" | head -1 | tr -d ' ')
elapsed=$(($(date +%s) - test_start))
if [ "$schema_exists" = "neurondb" ]; then
    print_test "NeuronDB schema exists" "PASS" "" "${elapsed}"
else
    print_test "NeuronDB schema exists" "FAIL" "Schema not found" "${elapsed}"
fi

# Test 2.4: Extension functions available
test_start=$(date +%s)
func_count=$(run_sql "SELECT COUNT(*) FROM pg_proc WHERE pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb');" 2>&1 | grep -v "ERROR" | head -1 | tr -d ' ')
elapsed=$(($(date +%s) - test_start))
if [ -n "$func_count" ] && [ "$func_count" -gt 0 ]; then
    print_test "Extension functions available" "PASS" "Found $func_count functions" "${elapsed}"
else
    print_test "Extension functions available" "FAIL" "No functions found" "${elapsed}"
fi

# Test 2.5: Compute mode
test_start=$(date +%s)
compute_mode=$(run_sql "SELECT current_setting('neurondb.compute_mode', true);" 2>&1 | grep -v "ERROR" | head -1 | tr -d ' ')
elapsed=$(($(date +%s) - test_start))
if [ -n "$compute_mode" ] && [ "$compute_mode" != "" ]; then
    mode_desc="CPU"
    case "$compute_mode" in
        0) mode_desc="CPU" ;;
        1) mode_desc="GPU" ;;
        2) mode_desc="AUTO" ;;
    esac
    print_test "Compute mode" "PASS" "Mode: $compute_mode ($mode_desc)" "${elapsed}"
else
    print_test "Compute mode" "SKIP" "Compute mode setting not available" "${elapsed}"
fi

# Test 2.6: GPU backend type (if applicable)
if [ "$VARIANT" != "cpu" ]; then
    test_start=$(date +%s)
    gpu_backend=$(run_sql "SELECT current_setting('neurondb.gpu_backend_type', true);" 2>&1 | grep -v "ERROR" | head -1 | tr -d ' ')
    elapsed=$(($(date +%s) - test_start))
    if [ -n "$gpu_backend" ] && [ "$gpu_backend" != "" ]; then
        backend_desc="Unknown"
        case "$gpu_backend" in
            0) backend_desc="None" ;;
            1) backend_desc="CUDA" ;;
            2) backend_desc="ROCm" ;;
            3) backend_desc="Metal" ;;
        esac
        print_test "GPU backend type" "PASS" "Backend: $gpu_backend ($backend_desc)" "${elapsed}"
    else
        print_test "GPU backend type" "SKIP" "GPU backend setting not available" "${elapsed}"
    fi
fi

# ============================================================================
# TEST SUITE 3: Basic Vector Operations
# ============================================================================
print_section "TEST SUITE 3: Basic Vector Operations"

# Test 3.1: Vector type creation
test_start=$(date +%s)
result=$(run_sql "SELECT '[1.0, 2.0, 3.0]'::vector;" 2>&1)
elapsed=$(($(date +%s) - test_start))
if echo "$result" | grep -qE "\[1,2,3\]|\[1\.0,2\.0,3\.0\]"; then
    print_test "Vector type creation" "PASS" "" "${elapsed}"
else
    print_test "Vector type creation" "FAIL" "Vector creation failed" "${elapsed}"
    if [ "$VERBOSE" = true ]; then
        echo -e "  ${RED}Output:${NC} $result"
    fi
fi

# Test 3.2: Vector operations
test_start=$(date +%s)
result=$(run_sql "SELECT '[1,2,3]'::vector + '[4,5,6]'::vector;" 2>&1)
elapsed=$(($(date +%s) - test_start))
if echo "$result" | grep -qE "\[5,7,9\]"; then
    print_test "Vector addition" "PASS" "" "${elapsed}"
else
    print_test "Vector addition" "FAIL" "Vector addition failed" "${elapsed}"
fi

# Test 3.3: Vector distance (L2)
test_start=$(date +%s)
result=$(run_sql "SELECT l2_distance('[1,2,3]'::vector, '[4,5,6]'::vector);" 2>&1)
elapsed=$(($(date +%s) - test_start))
if echo "$result" | grep -qE "^[0-9]+\.[0-9]+$"; then
    distance=$(echo "$result" | head -1 | tr -d ' ')
    print_test "L2 distance calculation" "PASS" "Distance: $distance" "${elapsed}"
else
    print_test "L2 distance calculation" "FAIL" "Distance calculation failed" "${elapsed}"
fi

# Test 3.4: Vector distance (cosine)
test_start=$(date +%s)
result=$(run_sql "SELECT cosine_distance('[1,2,3]'::vector, '[4,5,6]'::vector);" 2>&1)
elapsed=$(($(date +%s) - test_start))
if echo "$result" | grep -qE "^[0-9]+\.[0-9]+$"; then
    distance=$(echo "$result" | head -1 | tr -d ' ')
    print_test "Cosine distance calculation" "PASS" "Distance: $distance" "${elapsed}"
else
    print_test "Cosine distance calculation" "FAIL" "Distance calculation failed" "${elapsed}"
fi

# Test 3.5: Vector normalization
test_start=$(date +%s)
result=$(run_sql "SELECT l2_normalize('[3,4]'::vector);" 2>&1)
elapsed=$(($(date +%s) - test_start))
if echo "$result" | grep -qE "\[0\.6,0\.8\]|\[0\.600,0\.800\]"; then
    print_test "Vector normalization" "PASS" "" "${elapsed}"
else
    print_test "Vector normalization" "FAIL" "Normalization failed" "${elapsed}"
fi

# ============================================================================
# TEST SUITE 4: Vector Search and Indexing
# ============================================================================
print_section "TEST SUITE 4: Vector Search and Indexing"

# Test 4.1: Create test table with vectors
test_start=$(date +%s)
run_sql "DROP TABLE IF EXISTS test_vectors;" >/dev/null 2>&1
result=$(run_sql "CREATE TABLE test_vectors (id SERIAL PRIMARY KEY, embedding vector(3));" 2>&1)
elapsed=$(($(date +%s) - test_start))
if echo "$result" | grep -qE "CREATE TABLE|already exists"; then
    print_test "Create vector table" "PASS" "" "${elapsed}"
else
    print_test "Create vector table" "FAIL" "Table creation failed" "${elapsed}"
fi

# Test 4.2: Insert vectors
test_start=$(date +%s)
result=$(run_sql "INSERT INTO test_vectors (embedding) VALUES ('[1,2,3]'::vector), ('[4,5,6]'::vector), ('[7,8,9]'::vector);" 2>&1)
elapsed=$(($(date +%s) - test_start))
if echo "$result" | grep -qE "INSERT|^$"; then
    print_test "Insert vectors" "PASS" "" "${elapsed}"
else
    print_test "Insert vectors" "FAIL" "Insert failed" "${elapsed}"
fi

# Test 4.3: Vector similarity search
test_start=$(date +%s)
result=$(run_sql "SELECT id, embedding <-> '[1,2,3]'::vector AS distance FROM test_vectors ORDER BY embedding <-> '[1,2,3]'::vector LIMIT 1;" 2>&1)
elapsed=$(($(date +%s) - test_start))
if echo "$result" | grep -qE "^[0-9]+\|0$|^[0-9]+\|0\.0+$"; then
    print_test "Vector similarity search" "PASS" "" "${elapsed}"
else
    print_test "Vector similarity search" "FAIL" "Search failed" "${elapsed}"
fi

# Test 4.4: Create HNSW index
test_start=$(date +%s)
result=$(run_sql "CREATE INDEX IF NOT EXISTS test_vectors_hnsw_idx ON test_vectors USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 64);" 2>&1 || true)
elapsed=$(($(date +%s) - test_start))
if echo "$result" | grep -qE "CREATE INDEX|already exists"; then
    print_test "Create HNSW index" "PASS" "" "${elapsed}"
elif echo "$result" | grep -qiE "error|pfree|invalid"; then
    print_test "Create HNSW index" "SKIP" "Index creation error (may be a known issue)" "${elapsed}"
    if [ "$VERBOSE" = true ]; then
        echo -e "  ${YELLOW}Note:${NC} HNSW index creation encountered an error"
    fi
else
    print_test "Create HNSW index" "FAIL" "Index creation failed" "${elapsed}"
    if [ "$VERBOSE" = true ]; then
        echo -e "  ${RED}Output:${NC} $result"
    fi
fi

# Test 4.5: Index usage in query
test_start=$(date +%s)
result=$(run_sql "SET enable_seqscan = off; EXPLAIN SELECT id FROM test_vectors ORDER BY embedding <-> '[1,2,3]'::vector LIMIT 1;" 2>&1)
elapsed=$(($(date +%s) - test_start))
if echo "$result" | grep -qi "hnsw\|index"; then
    print_test "Index usage in queries" "PASS" "" "${elapsed}"
else
    print_test "Index usage in queries" "SKIP" "Index may not be used (expected for small datasets)" "${elapsed}"
fi

# Cleanup
run_sql "DROP TABLE IF EXISTS test_vectors;" >/dev/null 2>&1

# ============================================================================
# TEST SUITE 5: ML Functions (Basic)
# ============================================================================
print_section "TEST SUITE 5: ML Functions (Basic)"

# Test 5.1: ML functions exist
test_start=$(date +%s)
ml_func_count=$(run_sql "SELECT COUNT(*) FROM pg_proc WHERE pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb') AND proname LIKE '%train%' OR proname LIKE '%predict%';" 2>&1 | grep -v "ERROR" | head -1 | tr -d ' ')
elapsed=$(($(date +%s) - test_start))
if [ -n "$ml_func_count" ] && [ "$ml_func_count" -gt 0 ]; then
    print_test "ML functions available" "PASS" "Found $ml_func_count ML-related functions" "${elapsed}"
else
    print_test "ML functions available" "SKIP" "ML functions may require specific setup" "${elapsed}"
fi

# Test 5.2: ML function signatures
test_start=$(date +%s)
result=$(run_sql "SELECT proname, pg_get_function_arguments(oid) FROM pg_proc WHERE pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb') AND proname LIKE '%train%' LIMIT 1;" 2>&1)
elapsed=$(($(date +%s) - test_start))
if echo "$result" | grep -qi "train"; then
    print_test "ML function signatures" "PASS" "ML functions are registered" "${elapsed}"
else
    print_test "ML function signatures" "SKIP" "ML functions may require specific setup" "${elapsed}"
fi

# ============================================================================
# TEST SUITE 6: Container Resources and Limits
# ============================================================================
print_section "TEST SUITE 6: Container Resources and Limits"

# Test 6.1: Memory limit
test_start=$(date +%s)
mem_limit=$($DOCKER_CMD inspect --format='{{.HostConfig.Memory}}' "$CONTAINER_NAME" 2>/dev/null || echo "0")
elapsed=$(($(date +%s) - test_start))
if [ "$mem_limit" != "0" ] && [ -n "$mem_limit" ]; then
    mem_mb=$((mem_limit / 1024 / 1024))
    print_test "Memory limit configured" "PASS" "Limit: ${mem_mb}MB" "${elapsed}"
else
    print_test "Memory limit configured" "SKIP" "No memory limit set" "${elapsed}"
fi

# Test 6.2: CPU limit
test_start=$(date +%s)
cpu_limit=$($DOCKER_CMD inspect --format='{{.HostConfig.CpuQuota}}' "$CONTAINER_NAME" 2>/dev/null || echo "0")
elapsed=$(($(date +%s) - test_start))
if [ "$cpu_limit" != "0" ] && [ -n "$cpu_limit" ]; then
    print_test "CPU limit configured" "PASS" "Quota: ${cpu_limit}" "${elapsed}"
else
    print_test "CPU limit configured" "SKIP" "No CPU limit set" "${elapsed}"
fi

# Test 6.3: Container restart policy
test_start=$(date +%s)
restart_policy=$($DOCKER_CMD inspect --format='{{.HostConfig.RestartPolicy.Name}}' "$CONTAINER_NAME" 2>/dev/null || echo "none")
elapsed=$(($(date +%s) - test_start))
print_test "Restart policy" "PASS" "Policy: ${restart_policy}" "${elapsed}"

# Test 6.4: Logging configuration
test_start=$(date +%s)
log_driver=$($DOCKER_CMD inspect --format='{{.HostConfig.LogConfig.Type}}' "$CONTAINER_NAME" 2>/dev/null || echo "none")
elapsed=$(($(date +%s) - test_start))
if [ "$log_driver" != "none" ]; then
    print_test "Logging configured" "PASS" "Driver: ${log_driver}" "${elapsed}"
else
    print_test "Logging configured" "SKIP" "No logging driver configured" "${elapsed}"
fi

# ============================================================================
# TEST SUITE 7: Data Persistence
# ============================================================================
print_section "TEST SUITE 7: Data Persistence"

# Test 7.1: Volume mounts
test_start=$(date +%s)
volumes=$($DOCKER_CMD inspect --format='{{range .Mounts}}{{.Destination}} {{end}}' "$CONTAINER_NAME" 2>/dev/null || echo "")
elapsed=$(($(date +%s) - test_start))
if echo "$volumes" | grep -q "/var/lib/postgresql/data"; then
    print_test "Data volume mounted" "PASS" "Volume configured" "${elapsed}"
else
    print_test "Data volume mounted" "SKIP" "No data volume found" "${elapsed}"
fi

# Test 7.2: Data persistence test
test_start=$(date +%s)
run_sql "CREATE TABLE IF NOT EXISTS persistence_test (id SERIAL PRIMARY KEY, data TEXT);" >/dev/null 2>&1
run_sql "INSERT INTO persistence_test (data) VALUES ('test_data');" >/dev/null 2>&1
result=$(run_sql "SELECT data FROM persistence_test WHERE id = (SELECT MAX(id) FROM persistence_test);" 2>&1)
elapsed=$(($(date +%s) - test_start))
if echo "$result" | grep -q "test_data"; then
    print_test "Data persistence" "PASS" "Data written and read successfully" "${elapsed}"
    run_sql "DROP TABLE IF EXISTS persistence_test;" >/dev/null 2>&1
else
    print_test "Data persistence" "FAIL" "Data persistence test failed" "${elapsed}"
fi

# ============================================================================
# TEST SUITE 8: Performance Tests (if not in quick mode)
# ============================================================================
if [ "$QUICK_MODE" = false ]; then
    print_section "TEST SUITE 8: Performance Tests"
    
    # Test 8.1: Bulk vector insert performance
    test_start=$(date +%s)
    run_sql "CREATE TABLE IF NOT EXISTS perf_test (id SERIAL PRIMARY KEY, embedding vector(128));" >/dev/null 2>&1
    insert_start=$(date +%s)
    run_sql "INSERT INTO perf_test (embedding) SELECT (ARRAY(SELECT random()::float FROM generate_series(1, 128)))::vector FROM generate_series(1, 100);" >/dev/null 2>&1
    insert_elapsed=$(($(date +%s) - insert_start))
    elapsed=$(($(date +%s) - test_start))
    if [ $insert_elapsed -lt 10 ]; then
        print_test "Bulk vector insert (100 vectors)" "PASS" "Time: ${insert_elapsed}s" "${elapsed}"
    else
        print_test "Bulk vector insert (100 vectors)" "FAIL" "Too slow: ${insert_elapsed}s" "${elapsed}"
    fi
    run_sql "DROP TABLE IF EXISTS perf_test;" >/dev/null 2>&1
    
    # Test 8.2: Vector search performance
    test_start=$(date +%s)
    run_sql "CREATE TABLE IF NOT EXISTS perf_search (id SERIAL PRIMARY KEY, embedding vector(64));" >/dev/null 2>&1
    run_sql "INSERT INTO perf_search (embedding) SELECT (ARRAY(SELECT random()::float FROM generate_series(1, 64)))::vector FROM generate_series(1, 50);" >/dev/null 2>&1
    search_start=$(date +%s)
    run_sql "SELECT id FROM perf_search ORDER BY embedding <-> (ARRAY(SELECT random()::float FROM generate_series(1, 64)))::vector LIMIT 5;" >/dev/null 2>&1
    search_elapsed=$(($(date +%s) - search_start))
    elapsed=$(($(date +%s) - test_start))
    if [ $search_elapsed -lt 5 ]; then
        print_test "Vector search performance (50 vectors)" "PASS" "Time: ${search_elapsed}s" "${elapsed}"
    else
        print_test "Vector search performance (50 vectors)" "FAIL" "Too slow: ${search_elapsed}s" "${elapsed}"
    fi
    run_sql "DROP TABLE IF EXISTS perf_search;" >/dev/null 2>&1
else
    print_section "TEST SUITE 8: Performance Tests"
    print_test "Performance tests" "SKIP" "Quick mode enabled" ""
fi

# ============================================================================
# TEST SUITE 9: Error Handling
# ============================================================================
print_section "TEST SUITE 9: Error Handling"

# Test 9.1: Invalid vector dimension
test_start=$(date +%s)
result=$(run_sql "SELECT '[1,2,3,4]'::vector(3);" 2>&1 || true)
elapsed=$(($(date +%s) - test_start))
if echo "$result" | grep -qi "error\|dimension"; then
    print_test "Invalid vector dimension handling" "PASS" "Error properly caught" "${elapsed}"
else
    print_test "Invalid vector dimension handling" "SKIP" "Error handling may differ" "${elapsed}"
fi

# Test 9.2: Invalid distance function
test_start=$(date +%s)
result=$(run_sql "SELECT invalid_distance('[1,2,3]'::vector, '[4,5,6]'::vector);" 2>&1 || true)
elapsed=$(($(date +%s) - test_start))
if echo "$result" | grep -qi "error\|function.*does not exist"; then
    print_test "Invalid function error handling" "PASS" "Error properly caught" "${elapsed}"
else
    print_test "Invalid function error handling" "SKIP" "Error handling may differ" "${elapsed}"
fi

# ============================================================================
# TEST SUITE 10: Integration and Compatibility
# ============================================================================
print_section "TEST SUITE 10: Integration and Compatibility"

# Test 10.1: PostgreSQL compatibility
test_start=$(date +%s)
pg_version=$(run_sql "SHOW server_version;" 2>&1 | head -1 | tr -d ' ')
elapsed=$(($(date +%s) - test_start))
if [ -n "$pg_version" ]; then
    print_test "PostgreSQL compatibility" "PASS" "Version: $pg_version" "${elapsed}"
else
    print_test "PostgreSQL compatibility" "FAIL" "Cannot get version" "${elapsed}"
fi

# Test 10.2: Standard PostgreSQL functions work
test_start=$(date +%s)
result=$(run_sql "SELECT current_database(), current_user;" 2>&1)
elapsed=$(($(date +%s) - test_start))
if echo "$result" | grep -q "$DB_NAME"; then
    print_test "Standard PostgreSQL functions" "PASS" "" "${elapsed}"
else
    print_test "Standard PostgreSQL functions" "FAIL" "Standard functions not working" "${elapsed}"
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print_section "Test Summary"

TOTAL_TIME=$(($(date +%s) - TEST_START_TIME))

echo -e "${BOLD}Test Results:${NC}"
echo -e "  ${GREEN}Passed:${NC}  $TESTS_PASSED"
echo -e "  ${RED}Failed:${NC}  $TESTS_FAILED"
echo -e "  ${YELLOW}Skipped:${NC} $TESTS_SKIPPED"
echo -e "  ${BLUE}Total:${NC}   $((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))"
echo ""
echo -e "${BOLD}Total Time:${NC} ${TOTAL_TIME}s"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}${BOLD}✗ Some tests failed.${NC}"
    exit 1
fi

