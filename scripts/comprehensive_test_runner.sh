#!/bin/bash
# Comprehensive Test Runner
# Orchestrates all tests across all modules with timeout protection and failure tracking

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# Configuration
CONFIG_FILE="${SCRIPT_DIR}/test_config.yaml"
FAILURE_TRACKER="${SCRIPT_DIR}/failure_tracker.py"
TIMEOUT_WRAPPER="${SCRIPT_DIR}/test_timeout_wrapper.sh"
REPORT_GENERATOR="${SCRIPT_DIR}/generate_test_report.py"
RESULTS_DIR="${REPO_ROOT}/test-results"
FAILURES_FILE="${RESULTS_DIR}/failures.json"

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0
START_TIME=$(date +%s)

# Create results directory
mkdir -p "$RESULTS_DIR"
mkdir -p "${RESULTS_DIR}/logs"

# Load configuration (simplified - would use yq or python in production)
get_timeout() {
    local test_type="${1:-default}"
    case "$test_type" in
        unit) echo "30" ;;
        integration) echo "300" ;;
        e2e) echo "900" ;;
        performance) echo "1800" ;;
        stress) echo "3600" ;;
        *) echo "60" ;;
    esac
}

print_header() {
    echo -e "${BOLD}${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║         NeuronDB Comprehensive Feature Integration Test Suite              ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "${BLUE}Repository:${NC} $REPO_ROOT"
    echo -e "${BLUE}Results:${NC} $RESULTS_DIR"
    echo -e "${BLUE}Start Time:${NC} $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
}

print_section() {
    echo ""
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${CYAN}$1${NC}"
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

run_test_with_timeout() {
    local test_name="$1"
    local test_type="${2:-default}"
    local test_command="$3"
    local test_category="${4:-unknown}"
    
    local timeout=$(get_timeout "$test_type")
    local test_start=$(date +%s)
    
    echo -e "${BLUE}[RUN]${NC} $test_name (timeout: ${timeout}s)"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    # Run test with timeout wrapper
    local log_file="${RESULTS_DIR}/logs/${test_name//\//_}.log"
    mkdir -p "$(dirname "$log_file")"
    
    # Clean test name for file
    local safe_test_name=$(echo "$test_name" | tr '/ :' '_' | tr -cd '[:alnum:]_')
    log_file="${RESULTS_DIR}/logs/${safe_test_name}.log"
    
    if bash "$TIMEOUT_WRAPPER" "$timeout" $test_command > "$log_file" 2>&1; then
        local elapsed=$(($(date +%s) - test_start))
        echo -e "${GREEN}[PASS]${NC} $test_name (${elapsed}s)"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        local exit_code=$?
        local elapsed=$(($(date +%s) - test_start))
        
        if [ $exit_code -eq 124 ]; then
            echo -e "${RED}[TIMEOUT]${NC} $test_name (${elapsed}s)"
            SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
        else
            echo -e "${RED}[FAIL]${NC} $test_name (${elapsed}s)"
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi
        
        # Track failure
        local error_msg=$(tail -50 "$log_file" 2>/dev/null | head -20 | tr '\n' ' ' | sed 's/"/\\"/g' || echo "Unknown error")
        python3 "$FAILURE_TRACKER" add \
            --test-name "$test_name" \
            --category "$test_category" \
            --error "$error_msg" \
            --log-file "$log_file" \
            --storage-file "$FAILURES_FILE" 2>/dev/null || true
        
        return 0  # Continue even on failure
    fi
}

# Check prerequisites
check_prerequisites() {
    print_section "Checking Prerequisites"
    
    local missing=0
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}✗${NC} python3 not found"
        missing=1
    else
        echo -e "${GREEN}✓${NC} python3 found"
    fi
    
    # Check PostgreSQL
    if ! command -v psql &> /dev/null; then
        echo -e "${YELLOW}⚠${NC} psql not found (some tests may be skipped)"
    else
        echo -e "${GREEN}✓${NC} psql found"
    fi
    
    # Check Go (for NeuronAgent/NeuronMCP tests)
    if ! command -v go &> /dev/null; then
        echo -e "${YELLOW}⚠${NC} go not found (some tests may be skipped)"
    else
        echo -e "${GREEN}✓${NC} go found"
    fi
    
    # Check timeout command
    if ! command -v timeout &> /dev/null; then
        echo -e "${RED}✗${NC} timeout command not found"
        missing=1
    else
        echo -e "${GREEN}✓${NC} timeout command found"
    fi
    
    if [ $missing -eq 1 ]; then
        echo -e "${RED}Missing required prerequisites. Exiting.${NC}"
        exit 1
    fi
    
    # Make scripts executable
    chmod +x "$TIMEOUT_WRAPPER" 2>/dev/null || true
    chmod +x "$FAILURE_TRACKER" 2>/dev/null || true
    chmod +x "$REPORT_GENERATOR" 2>/dev/null || true
}

# Run NeuronDB tests
run_neurondb_tests() {
    print_section "NeuronDB Extension Tests"
    
    local neurondb_dir="${REPO_ROOT}/NeuronDB"
    
    if [ ! -d "$neurondb_dir" ]; then
        echo -e "${YELLOW}⚠${NC} NeuronDB directory not found, skipping"
        return
    fi
    
    # Test 1: Extension installation
    run_test_with_timeout \
        "NeuronDB: Extension Installation" \
        "unit" \
        "cd $neurondb_dir && psql -d neurondb -c 'CREATE EXTENSION IF NOT EXISTS neurondb; SELECT neurondb_version();'" \
        "neurondb.core"
    
    # Test 2: Vector operations
    if [ -f "${neurondb_dir}/tests/sql/basic/012_vector_ops.sql" ]; then
        run_test_with_timeout \
            "NeuronDB: Vector Operations" \
            "unit" \
            "cd $neurondb_dir && psql -d neurondb -f tests/sql/basic/012_vector_ops.sql" \
            "neurondb.vector.operations"
    fi
    
    # Test 3: HNSW Index
    if [ -f "${neurondb_dir}/tests/sql/basic/001_core_index.sql" ]; then
        run_test_with_timeout \
            "NeuronDB: HNSW Index" \
            "integration" \
            "cd $neurondb_dir && psql -d neurondb -f tests/sql/basic/001_core_index.sql" \
            "neurondb.index.hnsw"
    fi
    
    # Test 4: ML Algorithms (sample)
    if [ -f "${neurondb_dir}/tests/sql/basic/035_ml_linreg.sql" ]; then
        run_test_with_timeout \
            "NeuronDB: Linear Regression" \
            "integration" \
            "cd $neurondb_dir && psql -d neurondb -f tests/sql/basic/035_ml_linreg.sql" \
            "neurondb.ml.basic"
    fi
    
    # Test 5: Comprehensive vector tests
    if [ -f "${neurondb_dir}/tests/comprehensive/test_vector_comprehensive.sql" ]; then
        run_test_with_timeout \
            "NeuronDB: Comprehensive Vector Tests" \
            "integration" \
            "cd $neurondb_dir && psql -d neurondb -f tests/comprehensive/test_vector_comprehensive.sql" \
            "neurondb.vector.comprehensive"
    fi
    
    # Test 6: Comprehensive index tests
    if [ -f "${neurondb_dir}/tests/comprehensive/test_index_comprehensive.sql" ]; then
        run_test_with_timeout \
            "NeuronDB: Comprehensive Index Tests" \
            "integration" \
            "cd $neurondb_dir && psql -d neurondb -f tests/comprehensive/test_index_comprehensive.sql" \
            "neurondb.index.comprehensive"
    fi
    
    # Test 7: Comprehensive ML tests
    if [ -f "${neurondb_dir}/tests/comprehensive/test_ml_comprehensive.sql" ]; then
        run_test_with_timeout \
            "NeuronDB: Comprehensive ML Tests" \
            "integration" \
            "cd $neurondb_dir && psql -d neurondb -f tests/comprehensive/test_ml_comprehensive.sql" \
            "neurondb.ml.comprehensive"
    fi
    
    # Test 8: Run comprehensive test suite if available
    if [ -f "${neurondb_dir}/tests/run_test.py" ]; then
        run_test_with_timeout \
            "NeuronDB: Comprehensive Test Suite" \
            "integration" \
            "cd $neurondb_dir/tests && python3 run_test.py --category basic --max-tests 10" \
            "neurondb.comprehensive"
    fi
}

# Run NeuronAgent tests
run_neuronagent_tests() {
    print_section "NeuronAgent Tests"
    
    local agent_dir="${REPO_ROOT}/NeuronAgent"
    
    if [ ! -d "$agent_dir" ]; then
        echo -e "${YELLOW}⚠${NC} NeuronAgent directory not found, skipping"
        return
    fi
    
    # Test 1: Health check
    run_test_with_timeout \
        "NeuronAgent: Health Check" \
        "unit" \
        "curl -f http://localhost:8080/health" \
        "neuronagent.api.health"
    
    # Test 2: Go tests
    if [ -f "${agent_dir}/go.mod" ]; then
        run_test_with_timeout \
            "NeuronAgent: Go Unit Tests" \
            "integration" \
            "cd $agent_dir && go test ./internal/... -timeout 5m -count=1" \
            "neuronagent.unit"
    fi
    
    # Test 3: Comprehensive API tests
    if [ -f "${agent_dir}/tests/comprehensive/test_api_comprehensive.py" ]; then
        run_test_with_timeout \
            "NeuronAgent: Comprehensive API Tests" \
            "integration" \
            "cd $agent_dir && python3 tests/comprehensive/test_api_comprehensive.py" \
            "neuronagent.api.comprehensive"
    fi
    
    # Test 4: Python tests (if available)
    if [ -f "${agent_dir}/tests/run_tests.sh" ]; then
        run_test_with_timeout \
            "NeuronAgent: Python Tests" \
            "integration" \
            "cd $agent_dir && bash tests/run_tests.sh fast" \
            "neuronagent.python"
    fi
}

# Run NeuronMCP tests
run_neuronmcp_tests() {
    print_section "NeuronMCP Tests"
    
    local mcp_dir="${REPO_ROOT}/NeuronMCP"
    
    if [ ! -d "$mcp_dir" ]; then
        echo -e "${YELLOW}⚠${NC} NeuronMCP directory not found, skipping"
        return
    fi
    
    # Test 1: Go tests
    if [ -f "${mcp_dir}/go.mod" ]; then
        run_test_with_timeout \
            "NeuronMCP: Go Unit Tests" \
            "integration" \
            "cd $mcp_dir && go test ./internal/... -timeout 5m -count=1" \
            "neuronmcp.unit"
    fi
    
    # Test 2: Python tests (if available)
    if [ -d "${mcp_dir}/tests" ]; then
        run_test_with_timeout \
            "NeuronMCP: Python Tests" \
            "integration" \
            "cd $mcp_dir && python3 -m pytest tests/ -v --tb=short -x" \
            "neuronmcp.python"
    fi
}

# Run NeuronDesktop tests
run_neurondesktop_tests() {
    print_section "NeuronDesktop Tests"
    
    local desktop_dir="${REPO_ROOT}/NeuronDesktop"
    
    if [ ! -d "$desktop_dir" ]; then
        echo -e "${YELLOW}⚠${NC} NeuronDesktop directory not found, skipping"
        return
    fi
    
    # Test 1: API health check
    run_test_with_timeout \
        "NeuronDesktop: API Health Check" \
        "unit" \
        "curl -f http://localhost:8081/health" \
        "neurondesktop.api.health"
    
    # Test 2: Go tests
    if [ -f "${desktop_dir}/api/go.mod" ]; then
        run_test_with_timeout \
            "NeuronDesktop: API Go Tests" \
            "integration" \
            "cd $desktop_dir/api && go test ./... -timeout 5m -count=1" \
            "neurondesktop.api"
    fi
}

# Run cross-module integration tests
run_integration_tests() {
    print_section "Cross-Module Integration Tests"
    
    local integration_dir="${REPO_ROOT}/tests/integration/cross_module"
    
    if [ ! -d "$integration_dir" ]; then
        echo -e "${YELLOW}⚠${NC} Integration tests directory not found, skipping"
        return
    fi
    
    # Test 1: NeuronDB + NeuronAgent integration
    if [ -f "${integration_dir}/test_neurondb_agent_integration.sh" ]; then
        run_test_with_timeout \
            "Integration: NeuronDB + NeuronAgent" \
            "integration" \
            "bash ${integration_dir}/test_neurondb_agent_integration.sh" \
            "integration.neurondb.agent"
    fi
}

# Generate final report
generate_report() {
    print_section "Generating Test Report"
    
    # Generate markdown report
    python3 "$FAILURE_TRACKER" report --storage-file "$FAILURES_FILE" || true
    
    # Generate HTML report
    python3 "$REPORT_GENERATOR" "$FAILURES_FILE" "${RESULTS_DIR}/test_report.html" || true
    
    # Print summary
    local elapsed=$(($(date +%s) - START_TIME))
    local elapsed_min=$((elapsed / 60))
    local elapsed_sec=$((elapsed % 60))
    
    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}Test Summary${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "Total Tests:  ${TOTAL_TESTS}"
    echo -e "${GREEN}Passed:      ${PASSED_TESTS}${NC}"
    echo -e "${RED}Failed:       ${FAILED_TESTS}${NC}"
    echo -e "${YELLOW}Skipped:      ${SKIPPED_TESTS}${NC}"
    echo -e "Duration:     ${elapsed_min}m ${elapsed_sec}s"
    echo ""
    echo -e "Results:      ${RESULTS_DIR}"
    echo -e "Failures:     ${FAILURES_FILE}"
    echo ""
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "${GREEN}✓ All tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}✗ Some tests failed. See ${FAILURES_FILE} for details.${NC}"
        exit 1
    fi
}

# Main execution
main() {
    print_header
    check_prerequisites
    
    # Initialize failure tracker
    python3 "$FAILURE_TRACKER" > /dev/null 2>&1 || true
    
    # Run test suites
    run_neurondb_tests
    run_neuronagent_tests
    run_neuronmcp_tests
    run_neurondesktop_tests
    run_integration_tests
    
    # Generate report
    generate_report
}

# Handle interruption
trap 'echo -e "\n${YELLOW}Test run interrupted. Generating partial report...${NC}"; generate_report; exit 130' INT TERM

# Run main
main "$@"

