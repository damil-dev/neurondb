#!/bin/bash
#
# NeuronDB Health Check Script
# Self-sufficient script for all health checking and integration testing
#
# Usage:
#   ./neurondb-healthcheck.sh COMMAND [OPTIONS]
#
# Commands:
#   health         Check health of all components
#   integration    Run integration tests
#   smoke          Run smoke tests
#   quick          Quick health check
#
# This script is completely self-sufficient with no external dependencies.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_NAME=$(basename "$0")

# Colors (inline - no external dependency)
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Default configuration
COMMAND=""
VERBOSE=false
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-neurondb}"
DB_USER="${DB_USER:-neurondb}"
DB_PASSWORD="${DB_PASSWORD:-neurondb}"
AGENT_URL="${AGENT_URL:-http://localhost:8080}"

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

log_info() {
    echo -e "${CYAN}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

print_test() {
    local status="$1"
    local message="$2"
    
    case "$status" in
        PASS|pass)
            echo -e "${GREEN}✓${NC} $message"
            ((TESTS_PASSED++))
            ;;
        FAIL|fail)
            echo -e "${RED}✗${NC} $message"
            ((TESTS_FAILED++))
            ;;
        *)
            echo -e "${CYAN}ℹ${NC} $message"
            ;;
    esac
}

print_header() {
    echo ""
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  ${BOLD}NeuronDB Health Check${NC}                                  ${BLUE}║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

show_help() {
    cat << EOF
${BOLD}NeuronDB Health Check${NC}

${BOLD}Usage:${NC}
    ${SCRIPT_NAME} COMMAND [OPTIONS]

${BOLD}Commands:${NC}
    health         Check health of all components
    integration    Run integration tests
    smoke          Run smoke tests
    quick          Quick health check

${BOLD}Options:${NC}
    --host HOST         Database host (default: localhost)
    --port PORT         Database port (default: 5432)
    --database NAME     Database name (default: neurondb)
    --user USER         Database user (default: neurondb)
    --password PASS     Database password (or use DB_PASSWORD env var)
    --agent-url URL     NeuronAgent URL (default: http://localhost:8080)
    -h, --help          Show this help message
    -v, --verbose       Enable verbose output
    -V, --version       Show version information

${BOLD}Examples:${NC}
    # Quick health check
    ${SCRIPT_NAME} quick

    # Full health check
    ${SCRIPT_NAME} health

    # Integration tests
    ${SCRIPT_NAME} integration

    # Smoke tests
    ${SCRIPT_NAME} smoke

EOF
}

health_command() {
    shift
    print_header
    
    log_info "Checking NeuronDB ecosystem health..."
    echo ""
    
    # Check PostgreSQL/NeuronDB
    log_info "Checking NeuronDB (PostgreSQL)..."
    export PGPASSWORD="$DB_PASSWORD"
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" >/dev/null 2>&1; then
        print_test "PASS" "NeuronDB connection successful"
        
        # Check extension
        local ext_check=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -A -c "SELECT extname FROM pg_extension WHERE extname = 'neurondb';" 2>/dev/null | xargs || echo "")
        if [[ "$ext_check" == "neurondb" ]]; then
            local version=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -A -c "SELECT neurondb.version();" 2>/dev/null | xargs || echo "unknown")
            print_test "PASS" "NeuronDB extension installed (version: $version)"
        else
            print_test "FAIL" "NeuronDB extension not found"
        fi
    else
        print_test "FAIL" "NeuronDB connection failed"
    fi
    
    # Check NeuronAgent
    log_info "Checking NeuronAgent..."
    local health_code=$(curl -s -o /dev/null -w "%{http_code}" "$AGENT_URL/health" 2>/dev/null || echo "000")
    if [[ "$health_code" == "200" ]]; then
        print_test "PASS" "NeuronAgent health check (HTTP $health_code)"
    else
        print_test "FAIL" "NeuronAgent not responding (HTTP $health_code)"
    fi
    
    echo ""
    echo "Tests passed: $TESTS_PASSED"
    echo "Tests failed: $TESTS_FAILED"
    
    [[ $TESTS_FAILED -eq 0 ]] && return 0 || return 1
}

integration_command() {
    shift
    print_header
    
    log_info "Running integration tests..."
    echo ""
    
    # Run health checks first
    health_command "$@"
    
    # Additional integration tests
    log_info "Running component integration tests..."
    
    # Test NeuronDB -> NeuronAgent integration
    if curl -s "$AGENT_URL/health" >/dev/null 2>&1; then
        export PGPASSWORD="$DB_PASSWORD"
        if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" >/dev/null 2>&1; then
            print_test "PASS" "NeuronDB -> NeuronAgent integration"
        else
            print_test "FAIL" "NeuronDB -> NeuronAgent integration failed"
        fi
    fi
    
    echo ""
    echo "Integration tests passed: $TESTS_PASSED"
    echo "Integration tests failed: $TESTS_FAILED"
    
    [[ $TESTS_FAILED -eq 0 ]] && return 0 || return 1
}

smoke_command() {
    shift
    print_header
    
    log_info "Running smoke tests..."
    echo ""
    
    TESTS_PASSED=0
    TESTS_FAILED=0
    
    # Test 1: SQL Query
    log_info "Test 1: NeuronDB SQL Query"
    export PGPASSWORD="$DB_PASSWORD"
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT neurondb.version();" >/dev/null 2>&1; then
        local version=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -A -c "SELECT neurondb.version();" 2>/dev/null | xargs || echo "")
        if [[ -n "$version" ]]; then
            print_test "PASS" "NeuronDB SQL query (version: $version)"
        else
            print_test "FAIL" "NeuronDB SQL query returned empty"
        fi
    else
        print_test "FAIL" "NeuronDB SQL query failed"
    fi
    
    # Test 2: REST API
    log_info "Test 2: NeuronAgent REST API"
    local health_code=$(curl -s -o /dev/null -w "%{http_code}" "$AGENT_URL/health" 2>/dev/null || echo "000")
    if [[ "$health_code" == "200" ]]; then
        print_test "PASS" "NeuronAgent REST API (HTTP $health_code)"
    else
        print_test "FAIL" "NeuronAgent REST API failed (HTTP $health_code)"
    fi
    
    echo ""
    echo "Smoke tests passed: $TESTS_PASSED"
    echo "Smoke tests failed: $TESTS_FAILED"
    
    [[ $TESTS_FAILED -eq 0 ]] && return 0 || return 1
}

quick_command() {
    shift
    print_header
    
    log_info "Quick health check..."
    echo ""
    
    TESTS_PASSED=0
    TESTS_FAILED=0
    
    # Quick database check
    export PGPASSWORD="$DB_PASSWORD"
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" >/dev/null 2>&1; then
        print_test "PASS" "Database connection"
    else
        print_test "FAIL" "Database connection"
    fi
    
    # Quick agent check
    if curl -s "$AGENT_URL/health" >/dev/null 2>&1; then
        print_test "PASS" "NeuronAgent health"
    else
        print_test "FAIL" "NeuronAgent health"
    fi
    
    echo ""
    [[ $TESTS_FAILED -eq 0 ]] && log_success "All checks passed!" || log_error "Some checks failed!"
    
    [[ $TESTS_FAILED -eq 0 ]] && return 0 || return 1
}

parse_arguments() {
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi
    
    COMMAND="$1"
    shift
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --host)
                DB_HOST="$2"
                shift 2
                ;;
            --port)
                DB_PORT="$2"
                shift 2
                ;;
            --database)
                DB_NAME="$2"
                shift 2
                ;;
            --user)
                DB_USER="$2"
                shift 2
                ;;
            --password)
                DB_PASSWORD="$2"
                shift 2
                ;;
            --agent-url)
                AGENT_URL="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -V|--version)
                echo "${SCRIPT_NAME} version 2.0.0"
                exit 0
                ;;
            *)
                break
                ;;
        esac
    done
}

main() {
    parse_arguments "$@"
    
    case "$COMMAND" in
        health)
            health_command "$@"
            ;;
        integration)
            integration_command "$@"
            ;;
        smoke)
            smoke_command "$@"
            ;;
        quick)
            quick_command "$@"
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

main "$@"

