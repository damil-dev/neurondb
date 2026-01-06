#!/bin/bash
#
# NeuronDB Ecosystem Logs Viewer
# View and tail logs from all NeuronDB components
#
# Usage:
#   ./view-logs.sh [COMPONENT] [OPTIONS]
#
# Components:
#   neurondb, neuronagent, neuronmcp, neurondesktop, all
#
# Options:
#   --follow, -f         Follow log output (tail -f)
#   --lines N            Number of lines to show (default: 50)
#   --mode [docker|native]  Deployment mode (auto-detected)
#   --help, -h           Show this help

set -euo pipefail

# ============================================================================
# Configuration and Constants
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_NAME=$(basename "$0")

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly MAGENTA='\033[0;35m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Options
COMPONENT=""
FOLLOW=false
LINES=50
MODE=""

# ============================================================================
# Logging Functions
# ============================================================================

log_info() {
    echo -e "${CYAN}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $*"
}

log_error() {
    echo -e "${RED}[✗]${NC} $*" >&2
}

print_header() {
    echo ""
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  ${BOLD}NeuronDB Ecosystem Logs Viewer${NC}                           ${BLUE}║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_component_header() {
    local component=$1
    echo ""
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}  $component Logs${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# ============================================================================
# Utility Functions
# ============================================================================

show_help() {
    cat << EOF
${BOLD}NeuronDB Ecosystem Logs Viewer${NC}

${BOLD}Usage:${NC}
    $SCRIPT_NAME [COMPONENT] [OPTIONS]

${BOLD}Components:${NC}
    neurondb         Show NeuronDB logs
    neuronagent      Show NeuronAgent logs
    neuronmcp        Show NeuronMCP logs
    neurondesktop    Show NeuronDesktop logs
    all              Show logs from all components (default)

${BOLD}Options:${NC}
    --follow, -f     Follow log output in real-time (like tail -f)
    --lines N        Number of lines to show (default: 50)
    --mode MODE      Deployment mode: docker or native (auto-detected)
    --help, -h       Show this help message

${BOLD}Examples:${NC}
    # View last 50 lines of all logs
    $SCRIPT_NAME

    # Follow NeuronAgent logs
    $SCRIPT_NAME neuronagent --follow

    # View last 100 lines of NeuronDB logs
    $SCRIPT_NAME neurondb --lines 100

    # Follow all logs in Docker mode
    $SCRIPT_NAME all --follow --mode docker

EOF
}

detect_mode() {
    if [ -n "$MODE" ]; then
        return 0
    fi
    
    # Check if Docker is available and containers are running
    if command -v docker &> /dev/null && docker ps &> /dev/null; then
        if docker ps --format "{{.Names}}" | grep -q "neurondb"; then
            MODE="docker"
            return 0
        fi
    fi
    
    MODE="native"
}

# ============================================================================
# Log Viewing Functions
# ============================================================================

view_docker_logs() {
    local container=$1
    local component_name=$2
    
    if ! docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
        log_warning "$component_name container not running: $container"
        return 1
    fi
    
    print_component_header "$component_name"
    
    if [ "$FOLLOW" = true ]; then
        log_info "Following $component_name logs (Ctrl+C to exit)..."
        docker logs -f --tail "$LINES" "$container" 2>&1
    else
        docker logs --tail "$LINES" "$container" 2>&1
    fi
}

view_native_logs() {
    local log_path=$1
    local component_name=$2
    
    if [ ! -f "$log_path" ]; then
        log_warning "$component_name log file not found: $log_path"
        return 1
    fi
    
    print_component_header "$component_name"
    
    if [ "$FOLLOW" = true ]; then
        log_info "Following $component_name logs (Ctrl+C to exit)..."
        tail -f -n "$LINES" "$log_path"
    else
        tail -n "$LINES" "$log_path"
    fi
}

view_neurondb_logs() {
    if [ "$MODE" = "docker" ]; then
        view_docker_logs "neurondb-cpu" "NeuronDB"
    else
        # Try common PostgreSQL log locations
        local pg_log_dir=$(psql -U postgres -d postgres -t -c "SHOW log_directory;" 2>/dev/null | xargs || echo "/var/log/postgresql")
        local pg_log_file=$(find "$pg_log_dir" -name "*.log" -type f 2>/dev/null | head -1 || echo "")
        
        if [ -n "$pg_log_file" ]; then
            view_native_logs "$pg_log_file" "NeuronDB (PostgreSQL)"
        else
            log_warning "PostgreSQL log file not found"
            log_info "Try: sudo journalctl -u postgresql -n $LINES"
        fi
    fi
}

view_neuronagent_logs() {
    if [ "$MODE" = "docker" ]; then
        view_docker_logs "neuronagent" "NeuronAgent"
    else
        local log_locations=(
            "$PROJECT_ROOT/NeuronAgent/neuronagent.log"
            "$PROJECT_ROOT/NeuronAgent/logs/neuronagent.log"
            "/var/log/neuronagent/neuronagent.log"
        )
        
        local found=false
        for log_file in "${log_locations[@]}"; do
            if [ -f "$log_file" ]; then
                view_native_logs "$log_file" "NeuronAgent"
                found=true
                break
            fi
        done
        
        if [ "$found" = false ]; then
            log_warning "NeuronAgent log file not found"
            log_info "Try: sudo journalctl -u neuronagent -n $LINES"
        fi
    fi
}

view_neuronmcp_logs() {
    if [ "$MODE" = "docker" ]; then
        view_docker_logs "neurondb-mcp" "NeuronMCP"
    else
        local log_locations=(
            "$PROJECT_ROOT/NeuronMCP/neuronmcp.log"
            "$PROJECT_ROOT/NeuronMCP/logs/neuronmcp.log"
            "/var/log/neuronmcp/neuronmcp.log"
        )
        
        local found=false
        for log_file in "${log_locations[@]}"; do
            if [ -f "$log_file" ]; then
                view_native_logs "$log_file" "NeuronMCP"
                found=true
                break
            fi
        done
        
        if [ "$found" = false ]; then
            log_warning "NeuronMCP log file not found (MCP runs as stdio process)"
        fi
    fi
}

view_neurondesktop_logs() {
    if [ "$MODE" = "docker" ]; then
        view_docker_logs "neurondesktop" "NeuronDesktop"
    else
        local log_locations=(
            "$PROJECT_ROOT/NeuronDesktop/neurondesktop.log"
            "$PROJECT_ROOT/NeuronDesktop/logs/neurondesktop.log"
            "/var/log/neurondesktop/neurondesktop.log"
        )
        
        local found=false
        for log_file in "${log_locations[@]}"; do
            if [ -f "$log_file" ]; then
                view_native_logs "$log_file" "NeuronDesktop"
                found=true
                break
            fi
        done
        
        if [ "$found" = false ]; then
            log_warning "NeuronDesktop log file not found"
        fi
    fi
}

view_all_logs() {
    if [ "$MODE" = "docker" ] && [ "$FOLLOW" = true ]; then
        # For Docker follow mode, use docker compose logs
        cd "$PROJECT_ROOT"
        if [ -f "docker-compose.yml" ]; then
            log_info "Following all Docker logs (Ctrl+C to exit)..."
            docker compose logs -f --tail "$LINES" 2>&1 || docker-compose logs -f --tail "$LINES" 2>&1
        else
            log_error "docker-compose.yml not found"
            exit 1
        fi
    else
        # Show each component separately
        view_neurondb_logs || true
        view_neuronagent_logs || true
        view_neuronmcp_logs || true
        view_neurondesktop_logs || true
    fi
}

# ============================================================================
# Main Functions
# ============================================================================

parse_arguments() {
    # First argument might be component name (if not starting with --)
    if [ $# -gt 0 ] && [[ ! "$1" =~ ^-- ]]; then
        COMPONENT="$1"
        shift
    else
        COMPONENT="all"
    fi
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --follow|-f)
                FOLLOW=true
                shift
                ;;
            --lines)
                LINES="$2"
                shift 2
                ;;
            --mode)
                MODE="$2"
                shift 2
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Validate component
    if [[ ! "$COMPONENT" =~ ^(neurondb|neuronagent|neuronmcp|neurondesktop|all)$ ]]; then
        log_error "Invalid component: $COMPONENT"
        log_info "Valid components: neurondb, neuronagent, neuronmcp, neurondesktop, all"
        exit 1
    fi
}

main() {
    parse_arguments "$@"
    
    print_header
    
    detect_mode
    log_info "Deployment mode: $MODE"
    
    case "$COMPONENT" in
        neurondb)
            view_neurondb_logs
            ;;
        neuronagent)
            view_neuronagent_logs
            ;;
        neuronmcp)
            view_neuronmcp_logs
            ;;
        neurondesktop)
            view_neurondesktop_logs
            ;;
        all)
            view_all_logs
            ;;
    esac
}

main "$@"



