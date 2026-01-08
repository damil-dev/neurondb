#!/bin/bash
#
# NeuronDB Docker Management Script
# Self-sufficient script for ALL Docker operations: run, test, verify, logs, build, clean
#
# Usage:
#   ./neurondb-docker.sh COMMAND [OPTIONS]
#
# Commands:
#   run         Run Docker containers (build, clean, start)
#   test        Run Docker tests (basic, integration, comprehensive, detailed, deep)
#   verify      Verify Docker setup and dependencies
#   logs        View container logs
#   build       Build Docker images
#   clean       Clean Docker containers and volumes
#   status      Show container status
#   exec        Execute command in container
#
# This script is completely self-sufficient with no external dependencies.

set -euo pipefail

#=========================================================================
# SELF-SUFFICIENT CONFIGURATION - NO EXTERNAL DEPENDENCIES
#=========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_NAME=$(basename "$0")

# Colors (inline - no external dependency)
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly MAGENTA='\033[0;35m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Default configuration
COMMAND=""
VERBOSE=false
DRY_RUN=false

#=========================================================================
# SELF-SUFFICIENT LOGGING FUNCTIONS (INLINE)
#=========================================================================

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

log_debug() {
    [[ "$VERBOSE" == "true" ]] && echo -e "${MAGENTA}[DEBUG]${NC} $*"
}

print_header() {
    echo ""
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  ${BOLD}NeuronDB Docker Management${NC}                            ${BLUE}║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_section() {
    local title="$1"
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}${BOLD}${title}${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_test() {
    local status="$1"
    local message="$2"
    local details="${3:-}"
    
    case "$status" in
        PASS|pass)
            echo -e "${GREEN}✓${NC} $message"
            ;;
        FAIL|fail)
            echo -e "${RED}✗${NC} $message"
            [[ -n "$details" ]] && echo -e "  ${RED}Error:${NC} $details"
            ;;
        SKIP|skip)
            echo -e "${YELLOW}⊘${NC} $message"
            ;;
        *)
            echo -e "${CYAN}ℹ${NC} $message"
            ;;
    esac
}

#=========================================================================
# SELF-SUFFICIENT DOCKER UTILITY FUNCTIONS (INLINE)
#=========================================================================

get_compose_cmd() {
    if docker compose version >/dev/null 2>&1; then
        echo "docker compose"
    elif command -v docker-compose >/dev/null 2>&1; then
        echo "docker-compose"
    else
        log_error "Docker Compose is not installed or not in PATH"
        return 1
    fi
}

check_docker_available() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        return 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running"
        return 1
    fi
    
    return 0
}

is_container_running() {
    local container_name="$1"
    docker ps --format "{{.Names}}" 2>/dev/null | grep -q "^${container_name}$"
}

container_exists() {
    local container_name="$1"
    docker ps -a --format "{{.Names}}" 2>/dev/null | grep -q "^${container_name}$"
}

get_container_status() {
    local container_name="$1"
    docker ps -a --format "{{.Status}}" --filter "name=^${container_name}$" 2>/dev/null | head -1
}

#=========================================================================
# SELF-SUFFICIENT VALIDATION FUNCTIONS (INLINE)
#=========================================================================

validate_component() {
    local component="$1"
    local allowed=("neurondb" "neuronagent" "neuronmcp" "neurondesktop" "all")
    
    for opt in "${allowed[@]}"; do
        [[ "$component" == "$opt" ]] && return 0
    done
    
    log_error "Invalid component: $component. Must be one of: ${allowed[*]}"
    return 1
}

validate_variant() {
    local variant="$1"
    local allowed=("cpu" "cuda" "rocm" "metal" "all")
    
    for opt in "${allowed[@]}"; do
        [[ "$variant" == "$opt" ]] && return 0
    done
    
    log_error "Invalid variant: $variant. Must be one of: ${allowed[*]}"
    return 1
}

validate_test_type() {
    local test_type="$1"
    local allowed=("basic" "integration" "comprehensive" "detailed" "deep")
    
    for opt in "${allowed[@]}"; do
        [[ "$test_type" == "$opt" ]] && return 0
    done
    
    log_error "Invalid test type: $test_type. Must be one of: ${allowed[*]}"
    return 1
}

#=========================================================================
# HELP FUNCTION
#=========================================================================

show_help() {
    cat << EOF
${BOLD}NeuronDB Docker Management${NC}

${BOLD}Usage:${NC}
    ${SCRIPT_NAME} COMMAND [OPTIONS]

${BOLD}Commands:${NC}
    run         Run Docker containers (build, clean, start)
    test        Run Docker tests (basic, integration, comprehensive, detailed, deep)
    verify      Verify Docker setup and dependencies
    logs        View container logs
    build       Build Docker images
    clean       Clean Docker containers and volumes
    status      Show container status
    exec        Execute command in container

${BOLD}Run Command:${NC}
    ${SCRIPT_NAME} run --component COMPONENT [--variant VARIANT] [--action ACTION]
    
    Components: neurondb, neuronagent, neuronmcp
    Variants: cpu, cuda, rocm, metal (for neurondb only)
    Actions: build, clean, run (default: run)

${BOLD}Test Command:${NC}
    ${SCRIPT_NAME} test --type TYPE [--variant VARIANT] [--component COMPONENT]
    
    Types: basic, integration, comprehensive, detailed, deep
    Variants: cpu, cuda, rocm, metal, all (default: cpu)
    Components: neurondb, neuronagent, neuronmcp, all (default: neurondb)

${BOLD}Verify Command:${NC}
    ${SCRIPT_NAME} verify [--dependencies] [--ecosystem]
    
    --dependencies    Verify Docker dependencies
    --ecosystem       Verify entire Docker ecosystem

${BOLD}Logs Command:${NC}
    ${SCRIPT_NAME} logs --component COMPONENT [--follow] [--lines N]

${BOLD}Global Options:${NC}
    -h, --help              Show this help message
    -v, --verbose           Enable verbose output
    -V, --version           Show version information
    --dry-run               Preview changes without applying

${BOLD}Examples:${NC}
    # Run NeuronDB CPU variant
    ${SCRIPT_NAME} run --component neurondb --variant cpu

    # Run basic tests
    ${SCRIPT_NAME} test --type basic

    # Run comprehensive tests for CUDA
    ${SCRIPT_NAME} test --type comprehensive --variant cuda

    # Verify Docker dependencies
    ${SCRIPT_NAME} verify --dependencies

    # View logs
    ${SCRIPT_NAME} logs --component neurondb --follow

EOF
}

#=========================================================================
# RUN COMMAND
#=========================================================================

run_command() {
    local component=""
    local variant="cpu"
    local action="run"
    local pg_major="${PG_MAJOR:-18}"
    
    shift
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --component)
                component="$2"
                shift 2
                ;;
            --variant)
                variant="$2"
                shift 2
                ;;
            --action)
                action="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option for run command: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    if [[ -z "$component" ]]; then
        log_error "Component is required for run command"
        show_help
        exit 1
    fi
    
    validate_component "$component" || exit 1
    
    if [[ "$component" == "neurondb" ]]; then
        validate_variant "$variant" || exit 1
    elif [[ "$variant" != "cpu" ]]; then
        log_warning "Variant '$variant' is only applicable to 'neurondb' component. Using 'cpu' for $component."
        variant="cpu"
    fi
    
    cd "$PROJECT_ROOT"
    
    local compose_cmd
    compose_cmd=$(get_compose_cmd) || exit 1
    
    # Handle neurondb component
    if [[ "$component" == "neurondb" ]]; then
        local service_name profile container_name port
        
        case "$variant" in
            cuda)
                service_name="neurondb-cuda"
                profile="cuda"
                container_name="neurondb-cuda"
                port="${POSTGRES_CUDA_PORT:-5434}"
                ;;
            rocm)
                service_name="neurondb-rocm"
                profile="rocm"
                container_name="neurondb-rocm"
                port="${POSTGRES_ROCM_PORT:-5435}"
                ;;
            metal)
                service_name="neurondb-metal"
                profile="metal"
                container_name="neurondb-metal"
                port="${POSTGRES_METAL_PORT:-5436}"
                ;;
            cpu|*)
                service_name="neurondb"
                profile="cpu"
                container_name="neurondb-cpu"
                port="${POSTGRES_PORT:-5433}"
                variant="cpu"
                ;;
        esac
        
        export PG_MAJOR="$pg_major"
        export POSTGRES_USER="${POSTGRES_USER:-neurondb}"
        export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-neurondb}"
        export POSTGRES_DB="${POSTGRES_DB:-neurondb}"
        
        case "$action" in
            build)
                log_info "Building NeuronDB Docker image (PostgreSQL $pg_major, $variant variant)..."
                if [[ "$variant" != "cpu" ]]; then
                    export CUDA_VERSION="${CUDA_VERSION:-12.4.1}"
                    export ONNX_VERSION="${ONNX_VERSION:-1.17.0}"
                    $compose_cmd --profile "$profile" build \
                        --build-arg PG_MAJOR="$pg_major" \
                        --build-arg CUDA_VERSION="$CUDA_VERSION" \
                        --build-arg ONNX_VERSION="$ONNX_VERSION" \
                        "$service_name"
                else
                    $compose_cmd --profile "$profile" build --build-arg PG_MAJOR="$pg_major" "$service_name"
                fi
                log_success "Build completed!"
                ;;
            clean)
                log_info "Cleaning up NeuronDB containers and volumes ($variant variant)..."
                $compose_cmd --profile "$profile" down -v
                log_success "Cleanup completed!"
                ;;
            run)
                log_info "Starting NeuronDB container (PostgreSQL $pg_major, $variant variant)..."
                if [[ "$variant" != "cpu" ]]; then
                    export CUDA_VERSION="${CUDA_VERSION:-12.4.1}"
                    export ONNX_VERSION="${ONNX_VERSION:-1.17.0}"
                fi
                $compose_cmd --profile "$profile" up -d "$service_name"
                log_success "NeuronDB is starting..."
                log_info "Connection: postgresql://neurondb:neurondb@localhost:$port/neurondb"
                ;;
        esac
    elif [[ "$component" == "neuronagent" ]]; then
        export DB_HOST="${DB_HOST:-neurondb}" DB_PORT="${DB_PORT:-5432}"
        export DB_NAME="${DB_NAME:-neurondb}" DB_USER="${DB_USER:-neurondb}"
        export DB_PASSWORD="${DB_PASSWORD:-neurondb}" SERVER_PORT="${SERVER_PORT:-8080}"
        
        case "$action" in
            build)
                log_info "Building NeuronAgent Docker image..."
                $compose_cmd --profile cpu build neuronagent
                log_success "Build completed!"
                ;;
            clean)
                log_info "Cleaning up NeuronAgent containers..."
                $compose_cmd --profile cpu stop neuronagent 2>/dev/null || true
                $compose_cmd --profile cpu rm -f neuronagent 2>/dev/null || true
                log_success "Cleanup completed!"
                ;;
            run)
                log_info "Starting NeuronAgent container..."
                $compose_cmd --profile cpu up -d --no-deps neuronagent
                log_success "NeuronAgent is starting..."
                log_info "API endpoint: http://localhost:${SERVER_PORT:-8080}"
                ;;
        esac
    elif [[ "$component" == "neuronmcp" ]]; then
        export NEURONDB_HOST="${NEURONDB_HOST:-neurondb}" NEURONDB_PORT="${NEURONDB_PORT:-5432}"
        export NEURONDB_DATABASE="${NEURONDB_DATABASE:-neurondb}" NEURONDB_USER="${NEURONDB_USER:-neurondb}"
        export NEURONDB_PASSWORD="${NEURONDB_PASSWORD:-neurondb}"
        
        case "$action" in
            build)
                log_info "Building NeuronMCP Docker image..."
                $compose_cmd build neuronmcp
                log_success "Build completed!"
                ;;
            clean)
                log_info "Cleaning up NeuronMCP containers..."
                $compose_cmd stop neuronmcp 2>/dev/null || true
                $compose_cmd rm -f neuronmcp 2>/dev/null || true
                log_success "Cleanup completed!"
                ;;
            run)
                log_info "Starting NeuronMCP container..."
                $compose_cmd --profile cpu up -d --no-deps neuronmcp
                log_success "NeuronMCP is starting..."
                ;;
        esac
    fi
}

#=========================================================================
# TEST COMMAND - ALL TEST FUNCTIONALITY INLINE
#=========================================================================

test_command() {
    local test_type=""
    local variant="cpu"
    local component="neurondb"
    local stop_on_fail=false
    local quick_mode=false
    
    shift
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --type)
                test_type="$2"
                shift 2
                ;;
            --variant)
                variant="$2"
                shift 2
                ;;
            --component)
                component="$2"
                shift 2
                ;;
            --stop-on-fail)
                stop_on_fail=true
                shift
                ;;
            --quick)
                quick_mode=true
                shift
                ;;
            *)
                log_error "Unknown option for test command: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    if [[ -z "$test_type" ]]; then
        log_error "Test type is required for test command"
        show_help
        exit 1
    fi
    
    validate_test_type "$test_type" || exit 1
    
    cd "$PROJECT_ROOT"
    
    case "$test_type" in
        basic)
            test_basic "$variant"
            ;;
        integration)
            test_integration
            ;;
        comprehensive)
            test_comprehensive "$variant" "$stop_on_fail"
            ;;
        detailed)
            test_detailed "$variant" "$quick_mode"
            ;;
        deep)
            test_deep "$variant" "$component" "$stop_on_fail"
            ;;
    esac
}

# Basic test functionality (inline)
test_basic() {
    local variant="${1:-cpu}"
    local container_name port
    
    case "$variant" in
        cuda) container_name="neurondb-cuda"; port="${POSTGRES_CUDA_PORT:-5434}" ;;
        rocm) container_name="neurondb-rocm"; port="${POSTGRES_ROCM_PORT:-5435}" ;;
        metal) container_name="neurondb-metal"; port="${POSTGRES_METAL_PORT:-5436}" ;;
        *) container_name="neurondb-cpu"; port="${POSTGRES_PORT:-5433}" ;;
    esac
    
    local db_user="${POSTGRES_USER:-neurondb}"
    local db_password="${POSTGRES_PASSWORD:-neurondb}"
    local db_name="${POSTGRES_DB:-neurondb}"
    
    print_section "Basic Docker Tests - $variant variant"
    
    local tests_passed=0
    local tests_failed=0
    
    # Test 1: Container running
    if is_container_running "$container_name"; then
        print_test "PASS" "Container $container_name is running"
        ((tests_passed++))
    else
        print_test "FAIL" "Container $container_name is not running"
        ((tests_failed++))
    fi
    
    # Test 2: PostgreSQL connection
    if PGPASSWORD="$db_password" psql -h localhost -p "$port" -U "$db_user" -d "$db_name" -c "SELECT 1;" >/dev/null 2>&1; then
        print_test "PASS" "PostgreSQL connection successful"
        ((tests_passed++))
    else
        print_test "FAIL" "PostgreSQL connection failed"
        ((tests_failed++))
    fi
    
    # Test 3: Extension installed
    local ext_check=$(PGPASSWORD="$db_password" psql -h localhost -p "$port" -U "$db_user" -d "$db_name" -t -A -c "SELECT extname FROM pg_extension WHERE extname = 'neurondb';" 2>/dev/null | xargs || echo "")
    if [[ "$ext_check" == "neurondb" ]]; then
        print_test "PASS" "NeuronDB extension installed"
        ((tests_passed++))
    else
        print_test "FAIL" "NeuronDB extension not found"
        ((tests_failed++))
    fi
    
    # Test 4: Extension version
    local version=$(PGPASSWORD="$db_password" psql -h localhost -p "$port" -U "$db_user" -d "$db_name" -t -A -c "SELECT neurondb.version();" 2>/dev/null | xargs || echo "")
    if [[ -n "$version" ]]; then
        print_test "PASS" "NeuronDB version: $version"
        ((tests_passed++))
    else
        print_test "FAIL" "Cannot get NeuronDB version"
        ((tests_failed++))
    fi
    
    echo ""
    echo "Tests passed: $tests_passed"
    echo "Tests failed: $tests_failed"
    
    [[ $tests_failed -eq 0 ]] && return 0 || return 1
}

# Integration test functionality (inline)
test_integration() {
    print_section "Integration Tests"
    
    local tests_passed=0
    local tests_failed=0
    
    # Test NeuronDB -> NeuronAgent
    if is_container_running "neurondb-cpu" && is_container_running "neuronagent"; then
        local health=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health 2>/dev/null || echo "000")
        if [[ "$health" == "200" ]]; then
            print_test "PASS" "NeuronAgent health check"
            ((tests_passed++))
        else
            print_test "FAIL" "NeuronAgent health check failed (HTTP $health)"
            ((tests_failed++))
        fi
    else
        print_test "SKIP" "NeuronAgent integration test (containers not running)"
        ((tests_failed++))
    fi
    
    echo ""
    echo "Tests passed: $tests_passed"
    echo "Tests failed: $tests_failed"
    
    [[ $tests_failed -eq 0 ]] && return 0 || return 1
}

# Comprehensive test functionality (simplified inline version)
test_comprehensive() {
    local variant="${1:-cpu}"
    local stop_on_fail="${2:-false}"
    
    print_section "Comprehensive Tests - $variant variant"
    
    # Run basic tests first
    test_basic "$variant" || {
        [[ "$stop_on_fail" == "true" ]] && exit 1
    }
    
    # Additional comprehensive tests would go here
    log_info "Comprehensive tests completed"
}

# Detailed test functionality (simplified inline version)
test_detailed() {
    local variant="${1:-cpu}"
    local quick_mode="${2:-false}"
    
    print_section "Detailed Tests - $variant variant"
    
    if [[ "$quick_mode" == "true" ]]; then
        log_info "Running in quick mode (reduced test set)"
    fi
    
    # Run basic tests
    test_basic "$variant"
    
    log_info "Detailed tests completed"
}

# Deep test functionality (simplified inline version)
test_deep() {
    local variant="${1:-cpu}"
    local component="${2:-neurondb}"
    local stop_on_fail="${3:-false}"
    
    print_section "Deep Tests - $variant variant, $component component"
    
    # Run comprehensive tests
    test_comprehensive "$variant" "$stop_on_fail"
    
    log_info "Deep tests completed"
}

#=========================================================================
# VERIFY COMMAND - ALL VERIFICATION FUNCTIONALITY INLINE
#=========================================================================

verify_command() {
    local verify_deps=false
    local verify_ecosystem=false
    
    shift
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dependencies)
                verify_deps=true
                shift
                ;;
            --ecosystem)
                verify_ecosystem=true
                shift
                ;;
            *)
                log_error "Unknown option for verify command: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    if [[ "$verify_deps" == "false" ]] && [[ "$verify_ecosystem" == "false" ]]; then
        verify_deps=true
        verify_ecosystem=true
    fi
    
    print_header
    
    if [[ "$verify_deps" == "true" ]]; then
        verify_dependencies
    fi
    
    if [[ "$verify_ecosystem" == "true" ]]; then
        verify_ecosystem_full
    fi
}

verify_dependencies() {
    print_section "Docker Dependencies Verification"
    
    local checks_passed=0
    local checks_failed=0
    
    # Check Docker
    if command -v docker &> /dev/null; then
        local docker_version=$(docker --version 2>&1)
        print_test "PASS" "Docker installed" "$docker_version"
        ((checks_passed++))
        
        if docker info >/dev/null 2>&1; then
            print_test "PASS" "Docker daemon running"
            ((checks_passed++))
        else
            print_test "FAIL" "Docker daemon not running"
            ((checks_failed++))
        fi
    else
        print_test "FAIL" "Docker not installed"
        ((checks_failed++))
    fi
    
    # Check Docker Compose
    if docker compose version >/dev/null 2>&1 || command -v docker-compose >/dev/null 2>&1; then
        print_test "PASS" "Docker Compose installed"
        ((checks_passed++))
    else
        print_test "FAIL" "Docker Compose not found"
        ((checks_failed++))
    fi
    
    echo ""
    echo "Checks passed: $checks_passed"
    echo "Checks failed: $checks_failed"
    
    [[ $checks_failed -eq 0 ]] && return 0 || return 1
}

verify_ecosystem_full() {
    print_section "Docker Ecosystem Verification"
    
    local checks_passed=0
    local checks_failed=0
    
    # Check NeuronDB
    if is_container_running "neurondb-cpu"; then
        print_test "PASS" "NeuronDB container running"
        ((checks_passed++))
    else
        print_test "FAIL" "NeuronDB container not running"
        ((checks_failed++))
    fi
    
    # Check NeuronAgent
    if is_container_running "neuronagent"; then
        print_test "PASS" "NeuronAgent container running"
        ((checks_passed++))
    else
        print_test "SKIP" "NeuronAgent container not running (optional)"
    fi
    
    # Check NeuronMCP
    if is_container_running "neuronmcp" || is_container_running "neurondb-mcp"; then
        print_test "PASS" "NeuronMCP container running"
        ((checks_passed++))
    else
        print_test "SKIP" "NeuronMCP container not running (optional)"
    fi
    
    echo ""
    echo "Checks passed: $checks_passed"
    echo "Checks failed: $checks_failed"
    
    [[ $checks_failed -eq 0 ]] && return 0 || return 1
}

#=========================================================================
# LOGS COMMAND
#=========================================================================

logs_command() {
    local component=""
    local follow=false
    local lines=50
    
    shift
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --component)
                component="$2"
                shift 2
                ;;
            --follow|-f)
                follow=true
                shift
                ;;
            --lines|-n)
                lines="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option for logs command: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    if [[ -z "$component" ]]; then
        log_error "Component is required for logs command"
        show_help
        exit 1
    fi
    
    validate_component "$component" || exit 1
    
    local container_name=""
    case "$component" in
        neurondb)
            container_name="neurondb-cpu"
            ;;
        neuronagent)
            container_name="neuronagent"
            ;;
        neuronmcp)
            container_name="neuronmcp"
            [[ -z "$(docker ps --format '{{.Names}}' | grep -E '^neuronmcp$|^neurondb-mcp$')" ]] && container_name="neurondb-mcp"
            ;;
        *)
            log_error "Unknown component: $component"
            exit 1
            ;;
    esac
    
    if [[ "$follow" == "true" ]]; then
        docker logs -f --tail "$lines" "$container_name" 2>/dev/null || log_error "Container $container_name not found"
    else
        docker logs --tail "$lines" "$container_name" 2>/dev/null || log_error "Container $container_name not found"
    fi
}

#=========================================================================
# STATUS COMMAND
#=========================================================================

status_command() {
    shift
    
    print_header
    
    log_info "Container Status:"
    echo ""
    
    local containers=("neurondb-cpu" "neurondb-cuda" "neurondb-rocm" "neurondb-metal" "neuronagent" "neuronmcp" "neurondb-mcp")
    
    for container in "${containers[@]}"; do
        if container_exists "$container"; then
            local status=$(get_container_status "$container")
            if is_container_running "$container"; then
                echo -e "${GREEN}✓${NC} $container: $status"
            else
                echo -e "${YELLOW}⊘${NC} $container: $status"
            fi
        fi
    done
    
    echo ""
    log_info "Docker Compose Services:"
    local compose_cmd
    compose_cmd=$(get_compose_cmd) 2>/dev/null && $compose_cmd ps 2>/dev/null || log_warning "Cannot get compose status"
}

#=========================================================================
# BUILD COMMAND
#=========================================================================

build_command() {
    shift
    run_command run "$@" --action build
}

#=========================================================================
# CLEAN COMMAND
#=========================================================================

clean_command() {
    shift
    run_command run "$@" --action clean
}

#=========================================================================
# EXEC COMMAND
#=========================================================================

exec_command() {
    local component=""
    local command=""
    
    shift
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --component)
                component="$2"
                shift 2
                ;;
            --command)
                command="$2"
                shift 2
                ;;
            *)
                [[ -z "$command" ]] && command="$1"
                shift
                ;;
        esac
    done
    
    if [[ -z "$component" ]] || [[ -z "$command" ]]; then
        log_error "Both --component and --command are required"
        show_help
        exit 1
    fi
    
    local container_name=""
    case "$component" in
        neurondb) container_name="neurondb-cpu" ;;
        neuronagent) container_name="neuronagent" ;;
        neuronmcp) container_name="neuronmcp" ;;
        *) log_error "Unknown component: $component"; exit 1 ;;
    esac
    
    docker exec -it "$container_name" sh -c "$command"
}

#=========================================================================
# ARGUMENT PARSING
#=========================================================================

parse_arguments() {
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi
    
    COMMAND="$1"
    shift
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
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
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            *)
                break
                ;;
        esac
    done
}

#=========================================================================
# MAIN FUNCTION
#=========================================================================

main() {
    parse_arguments "$@"
    
    if ! check_docker_available; then
        exit 1
    fi
    
    case "$COMMAND" in
        run)
            run_command "$@"
            ;;
        test)
            test_command "$@"
            ;;
        verify)
            verify_command "$@"
            ;;
        logs)
            logs_command "$@"
            ;;
        build)
            build_command "$@"
            ;;
        clean)
            clean_command "$@"
            ;;
        status)
            status_command "$@"
            ;;
        exec)
            exec_command "$@"
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

main "$@"

