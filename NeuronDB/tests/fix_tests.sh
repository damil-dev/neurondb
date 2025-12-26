#!/bin/bash
# Automated test fixing script
# Runs all tests individually, detects failures, and provides workflow for fixing

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$SCRIPT_DIR/sql"
TEST_RUNNER="$SCRIPT_DIR/run_test.py"
NEURONDB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CORE_DIR="/tmp/core"
ERROR_DIR="$SCRIPT_DIR/error"
OUTPUT_DIR="$SCRIPT_DIR/output"
LOG_FILE="$SCRIPT_DIR/fix_tests.log"
STATE_FILE="$SCRIPT_DIR/fix_tests.state"
DB_NAME="${DB_NAME:-neurondb}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "$LOG_FILE"
}

# Get all test files
get_all_test_files() {
    local test_files=()
    while IFS= read -r -d '' file; do
        # Get relative path from sql directory
        rel_path="${file#$TESTS_DIR/}"
        test_files+=("$rel_path")
    done < <(find "$TESTS_DIR" -name "*.sql" -type f -print0 | sort -z)
    printf '%s\n' "${test_files[@]}"
}

# Extract test name from file path for --test parameter
# Test name is the basename without .sql extension
get_test_name() {
    local file_path="$1"
    basename "$file_path" .sql
}

# Run a single test
run_single_test() {
    local test_file="$1"
    local test_name=$(get_test_name "$test_file")
    
    log "Running test: $test_file (test_name: $test_name)"
    
    # Run the test and capture output and exit code
    local output
    local exit_code=0
    
    # Use --test parameter with --category=all to run the specific test
    # Note: --test matches test names, so we use the basename
    output=$(python3 "$TEST_RUNNER" \
        --test="$test_name" \
        --category=all \
        --compute=cpu \
        --db="$DB_NAME" \
        2>&1) || exit_code=$?
    
    # Log output for debugging
    echo "$output" | tee -a "$LOG_FILE"
    
    # Check exit code first (non-zero means failure)
    if [[ $exit_code -ne 0 ]]; then
        return 1  # Test failed
    fi
    
    # Check output for test result indicators
    # run_test.py uses ✓ for pass and ✗ for fail
    # Note: grep in if conditions won't trigger set -e
    if echo "$output" | grep -q "✗\|Failed:"; then
        return 1  # Test failed
    fi
    
    if echo "$output" | grep -q "✓"; then
        return 0  # Test passed
    fi
    
    # If no clear indicator but exit code is 0, check for "Test Report" summary
    if echo "$output" | grep -q "Failed:.*[1-9]"; then
        return 1  # Has failures
    fi
    
    if echo "$output" | grep -q "Passed:.*[1-9]"; then
        if ! echo "$output" | grep -q "Failed:.*[1-9]"; then
            return 0  # Only passes, no failures
        fi
    fi
    
    # Ambiguous case - assume success if exit code was 0
    return 0
}

# Check for core dumps
check_core_dumps() {
    local test_name="$1"
    local cores=()
    
    if [[ -d "$CORE_DIR" ]]; then
        while IFS= read -r -d '' core; do
            cores+=("$core")
        done < <(find "$CORE_DIR" -name "core.*" -type f -print0 2>/dev/null)
    fi
    
    if [[ ${#cores[@]} -gt 0 ]]; then
        log_warning "Core dumps found for test $test_name:"
        for core in "${cores[@]}"; do
            log_warning "  - $core"
        done
        return 0
    fi
    return 1
}

# Check error files
check_error_files() {
    local test_name="$1"
    local error_file="$ERROR_DIR/${test_name}.err"
    
    if [[ -f "$error_file" ]]; then
        log_warning "Error file found: $error_file"
        log "Last 20 lines of error file:"
        tail -n 20 "$error_file" | tee -a "$LOG_FILE"
        return 0
    fi
    return 1
}

# Check PostgreSQL log
check_pg_log() {
    local pg_log="$NEURONDB_DIR/pg.log"
    if [[ -f "$pg_log" ]]; then
        log "Checking PostgreSQL log for recent errors..."
        tail -n 50 "$pg_log" | grep -i "error\|fatal\|panic" | tail -n 10 | tee -a "$LOG_FILE" || true
    fi
}

# Restart PostgreSQL
restart_postgresql() {
    log "Restarting PostgreSQL..."
    
    # Try to find pg_ctl
    local pg_ctl
    pg_ctl=$(command -v pg_ctl 2>/dev/null || echo "")
    
    if [[ -z "$pg_ctl" ]]; then
        log_error "pg_ctl not found in PATH"
        return 1
    fi
    
    # Try to find PGDATA
    local pgdata
    pgdata="${PGDATA:-}"
    
    if [[ -z "$pgdata" ]]; then
        # Try common locations
        for loc in "$HOME/pgdata" "$HOME/postgres_data" "/var/lib/postgresql"*; do
            if [[ -d "$loc" ]] && [[ -f "$loc/postgresql.conf" ]]; then
                pgdata="$loc"
                break
            fi
        done
    fi
    
    if [[ -z "$pgdata" ]]; then
        log_error "PGDATA not set and cannot be found automatically"
        log_error "Please set PGDATA environment variable or restart PostgreSQL manually"
        return 1
    fi
    
    log "Using PGDATA: $pgdata"
    
    # Restart PostgreSQL
    if "$pg_ctl" restart -D "$pgdata" -l "$pgdata/pg.log" -w; then
        sleep 2
        log_success "PostgreSQL restarted successfully"
        return 0
    else
        log_error "Failed to restart PostgreSQL"
        return 1
    fi
}

# Build NeuronDB
build_neurondb() {
    log "Building NeuronDB..."
    cd "$NEURONDB_DIR"
    
    if make -j12 install >> "$LOG_FILE" 2>&1; then
        log_success "NeuronDB built successfully"
        return 0
    else
        log_error "Build failed. Check $LOG_FILE for details."
        return 1
    fi
}

# Process a single test with fixing workflow
process_test() {
    local test_file="$1"
    local test_name=$(get_test_name "$test_file")
    local max_attempts=10
    local attempt=1
    
    log "========================================"
    log "Processing test: $test_file"
    log "========================================"
    
    while [[ $attempt -le $max_attempts ]]; do
        log "Attempt $attempt/$max_attempts"
        
        # Run the test
        if run_single_test "$test_file"; then
            log_success "Test PASSED: $test_file"
            echo "$test_file" >> "$STATE_FILE"
            return 0
        fi
        
        log_error "Test FAILED: $test_file (attempt $attempt/$max_attempts)"
        
        # Check for core dumps
        if check_core_dumps "$test_name"; then
            log "Core dump detected - this indicates a crash/segfault"
        fi
        
        # Check error files
        if check_error_files "$test_name"; then
            log "Error file contains details about the failure"
        fi
        
        # Check PostgreSQL log
        check_pg_log
        
        # If this is not the last attempt, prompt for fix
        if [[ $attempt -lt $max_attempts ]]; then
            log_warning "Test failed. Please fix the issue and then:"
            log "1. Fix the source code"
            log "2. Rebuild with: make -j12 install (in $NEURONDB_DIR)"
            log "3. Restart PostgreSQL if needed"
            log ""
            log "Press Enter after fixing and rebuilding to retry the test..."
            read -r
            
            # Optionally rebuild and restart
            log "Do you want to rebuild NeuronDB? (y/n)"
            read -r rebuild
            if [[ "$rebuild" == "y" ]]; then
                build_neurondb
                log "Do you want to restart PostgreSQL? (y/n)"
                read -r restart
                if [[ "$restart" == "y" ]]; then
                    restart_postgresql
                fi
            fi
        else
            log_error "Test failed after $max_attempts attempts: $test_file"
            log_error "Moving to next test. Please fix manually later."
            return 1
        fi
        
        ((attempt++))
    done
    
    return 1
}

# Get list of tests to run (skip already passed tests)
get_tests_to_run() {
    local all_tests
    local passed_tests=()
    
    # Read passed tests from state file
    if [[ -f "$STATE_FILE" ]]; then
        mapfile -t passed_tests < "$STATE_FILE"
    fi
    
    # Get all tests
    mapfile -t all_tests < <(get_all_test_files)
    
    # Filter out passed tests
    local tests_to_run=()
    for test in "${all_tests[@]}"; do
        local skip=0
        for passed in "${passed_tests[@]}"; do
            if [[ "$test" == "$passed" ]]; then
                skip=1
                break
            fi
        done
        if [[ $skip -eq 0 ]]; then
            tests_to_run+=("$test")
        fi
    done
    
    printf '%s\n' "${tests_to_run[@]}"
}

# Main function
main() {
    log "Starting test fixing workflow"
    log "Log file: $LOG_FILE"
    log "State file: $STATE_FILE"
    
    # Ensure directories exist
    mkdir -p "$CORE_DIR" "$ERROR_DIR" "$OUTPUT_DIR"
    
    # Clear state file if starting fresh
    if [[ "${1:-}" == "--fresh" ]]; then
        log "Starting fresh (clearing state file)"
        > "$STATE_FILE"
    fi
    
    # Get tests to run
    local tests_to_run
    mapfile -t tests_to_run < <(get_tests_to_run)
    
    local total=${#tests_to_run[@]}
    log "Found $total tests to run"
    
    if [[ $total -eq 0 ]]; then
        log_success "All tests have already passed!"
        return 0
    fi
    
    local current=0
    local passed=0
    local failed=0
    
    # Process each test
    for test_file in "${tests_to_run[@]}"; do
        ((current++))
        log "Progress: $current/$total"
        
        if process_test "$test_file"; then
            ((passed++))
        else
            ((failed++))
        fi
    done
    
    # Summary
    log "========================================"
    log "Summary:"
    log "  Total: $total"
    log "  Passed: $passed"
    log "  Failed: $failed"
    log "========================================"
    
    if [[ $failed -eq 0 ]]; then
        log_success "All tests passed!"
        return 0
    else
        log_error "$failed tests still need fixing"
        return 1
    fi
}

# Run main function
main "$@"

