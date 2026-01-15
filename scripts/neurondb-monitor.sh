#!/bin/bash
#
# NeuronDB Monitoring Script
# Self-sufficient script for all monitoring operations: status, logs, metrics
#
# Usage:
#   ./neurondb-monitor.sh COMMAND [OPTIONS]
#
# Commands:
#   status         Show status of all components
#   logs           View container logs
#   watch          Watch status continuously
#   metrics        Show metrics and statistics
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
FOLLOW=false
LINES=50
WATCH_INTERVAL=5
MODE=""


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

print_header() {
    echo ""
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  ${BOLD}NeuronDB Monitoring${NC}                                  ${BLUE}║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}


show_help() {
    cat << EOF
${BOLD}NeuronDB Monitoring${NC}

${BOLD}Usage:${NC}
    ${SCRIPT_NAME} COMMAND [OPTIONS]

${BOLD}Commands:${NC}
    status         Show status of all components
    logs           View container logs
    watch          Watch status continuously
    metrics        Show metrics and statistics

${BOLD}Logs Options:${NC}
    --component COMPONENT   Component name (neurondb, neuronagent, neuronmcp)
    --follow, -f            Follow log output
    --lines N               Number of lines to show (default: 50)

${BOLD}Watch Options:${NC}
    --interval SECONDS      Refresh interval in seconds (default: 5)

${BOLD}Global Options:${NC}
    --mode [docker|native]  Deployment mode (auto-detected)
    -h, --help              Show this help message
    -v, --verbose           Enable verbose output
    -V, --version           Show version information

${BOLD}Examples:${NC}
    # Show status
    ${SCRIPT_NAME} status

    # View logs
    ${SCRIPT_NAME} logs --component neurondb --follow

    # Watch status
    ${SCRIPT_NAME} watch --interval 3

    # Show metrics
    ${SCRIPT_NAME} metrics

EOF
}


detect_mode() {
    if [[ -n "$MODE" ]]; then
        return 0
    fi
    
    if command -v docker &> /dev/null && docker ps &> /dev/null; then
        if docker ps --format "{{.Names}}" | grep -q "neurondb"; then
            MODE="docker"
            return 0
        fi
    fi
    
    if pgrep -f "neuronagent" &> /dev/null || pgrep -f "postgres.*neurondb" &> /dev/null; then
        MODE="native"
        return 0
    fi
    
    MODE="unknown"
    return 1
}

is_container_running() {
    local container="$1"
    docker ps --format "{{.Names}}" 2>/dev/null | grep -q "^${container}$"
}


status_command() {
    shift
    print_header
    
    detect_mode || log_warning "Cannot detect deployment mode"
    
    log_info "Component Status (mode: $MODE)"
    echo ""
    
    if [[ "$MODE" == "docker" ]]; then
        local containers=("neurondb-cpu" "neurondb-cuda" "neurondb-rocm" "neurondb-metal" "neuronagent" "neuronmcp" "neurondb-mcp")
        
        for container in "${containers[@]}"; do
            if is_container_running "$container"; then
                local status=$(docker ps --format "{{.Status}}" --filter "name=^${container}$" 2>/dev/null | head -1)
                echo -e "${GREEN}✓${NC} $container: $status"
            fi
        done
    elif [[ "$MODE" == "native" ]]; then
        if pgrep -f "postgres.*neurondb" &> /dev/null; then
            echo -e "${GREEN}✓${NC} NeuronDB (native PostgreSQL)"
        fi
        
        if pgrep -f "neuronagent" &> /dev/null; then
            echo -e "${GREEN}✓${NC} NeuronAgent (native)"
        fi
    else
        log_warning "No components detected"
    fi
    
    echo ""
}


logs_command() {
    shift
    
    local component=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --component)
                component="$2"
                shift 2
                ;;
            --follow|-f)
                FOLLOW=true
                shift
                ;;
            --lines|-n)
                LINES="$2"
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done
    
    if [[ -z "$component" ]]; then
        log_error "Component is required for logs command"
        show_help
        exit 1
    fi
    
    detect_mode || log_warning "Cannot detect deployment mode"
    
    if [[ "$MODE" == "docker" ]]; then
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
        
        if [[ "$FOLLOW" == "true" ]]; then
            docker logs -f --tail "$LINES" "$container_name" 2>/dev/null || log_error "Container $container_name not found"
        else
            docker logs --tail "$LINES" "$container_name" 2>/dev/null || log_error "Container $container_name not found"
        fi
    else
        log_warning "Logs command requires Docker mode"
        exit 1
    fi
}


watch_command() {
    shift
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --interval)
                WATCH_INTERVAL="$2"
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done
    
    while true; do
        clear
        status_command
        sleep "$WATCH_INTERVAL"
    done
}


metrics_command() {
    shift
    print_header
    
    log_info "Component Metrics"
    echo ""
    
    detect_mode || log_warning "Cannot detect deployment mode"
    
    if [[ "$MODE" == "docker" ]]; then
        log_info "Docker Container Stats:"
        docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" $(docker ps --format "{{.Names}}" | grep -E "neuron(db|agent|mcp)") 2>/dev/null || log_warning "No containers running"
    else
        log_warning "Metrics command requires Docker mode"
    fi
    
    echo ""
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
            --component)
                # Handled by logs command
                shift 2
                ;;
            --follow|-f)
                # Handled by logs command
                shift
                ;;
            --lines|-n)
                # Handled by logs command
                shift 2
                ;;
            --interval)
                # Handled by watch command
                shift 2
                ;;
            --mode)
                MODE="$2"
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
        status)
            status_command "$@"
            ;;
        logs)
            logs_command "$@"
            ;;
        watch)
            watch_command "$@"
            ;;
        metrics)
            metrics_command "$@"
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

main "$@"

