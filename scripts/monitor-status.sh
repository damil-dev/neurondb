#!/bin/bash
#
# NeuronDB Ecosystem Status Monitoring Script
# Real-time monitoring of all NeuronDB components with health checks
#
# Usage:
#   ./monitor-status.sh [OPTIONS]
#
# Options:
#   --mode [docker|native]    Deployment mode (default: auto-detect)
#   --watch                   Continuous monitoring with updates every 5 seconds
#   --json                    Output in JSON format
#   --help, -h                Show this help

set -euo pipefail

# ============================================================================
# Configuration and Constants
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME=$(basename "$0")

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly MAGENTA='\033[0;35m'
readonly BOLD='\033[1m'
readonly DIM='\033[2m'
readonly NC='\033[0m'

# Default configuration
MODE=""
WATCH_MODE=false
JSON_OUTPUT=false

# Component ports
NEURONDB_PORT=5433
NEURONAGENT_PORT=8080
NEURONDESKTOP_PORT=8081
NEURONDESKTOP_UI_PORT=3000

# Status tracking
declare -A COMPONENT_STATUS
declare -A COMPONENT_VERSION
declare -A COMPONENT_UPTIME
declare -A COMPONENT_MEMORY
declare -A COMPONENT_CPU

# ============================================================================
# Logging Functions
# ============================================================================

log_info() {
    if [ "$JSON_OUTPUT" = false ]; then
        echo -e "${CYAN}[INFO]${NC} $*"
    fi
}

log_success() {
    if [ "$JSON_OUTPUT" = false ]; then
        echo -e "${GREEN}[✓]${NC} $*"
    fi
}

log_warning() {
    if [ "$JSON_OUTPUT" = false ]; then
        echo -e "${YELLOW}[⚠]${NC} $*"
    fi
}

log_error() {
    if [ "$JSON_OUTPUT" = false ]; then
        echo -e "${RED}[✗]${NC} $*"
    fi
}

print_header() {
    if [ "$JSON_OUTPUT" = false ]; then
        clear
        echo ""
        echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${BLUE}║${NC}  ${BOLD}NeuronDB Ecosystem Status Monitor${NC}                        ${BLUE}║${NC}"
        echo -e "${BLUE}║${NC}  ${DIM}$(date '+%Y-%m-%d %H:%M:%S')${NC}                                    ${BLUE}║${NC}"
        echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
        echo ""
    fi
}

# ============================================================================
# Detection Functions
# ============================================================================

detect_mode() {
    if [ -n "$MODE" ]; then
        log_info "Using specified mode: $MODE"
        return 0
    fi
    
    # Check if Docker is available and containers are running
    if command -v docker &> /dev/null && docker ps &> /dev/null; then
        if docker ps --format "{{.Names}}" | grep -q "neurondb"; then
            MODE="docker"
            log_info "Detected Docker deployment"
            return 0
        fi
    fi
    
    # Check if native processes are running
    if pgrep -f "neuronagent" &> /dev/null || pgrep -f "postgres.*neurondb" &> /dev/null; then
        MODE="native"
        log_info "Detected native deployment"
        return 0
    fi
    
    MODE="unknown"
    log_warning "Cannot detect deployment mode"
    return 1
}

# ============================================================================
# Health Check Functions
# ============================================================================

check_neurondb_health() {
    local container_name="neurondb-cpu"
    
    if [ "$MODE" = "docker" ]; then
        # Docker mode
        if docker ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
            # Check if container is healthy
            local health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "none")
            local container_status=$(docker inspect --format='{{.State.Status}}' "$container_name" 2>/dev/null || echo "not found")
            
            if [ "$container_status" = "running" ]; then
                # Try to connect and get version
                if docker exec "$container_name" psql -U neurondb -d neurondb -t -c "SELECT neurondb.version();" &> /dev/null; then
                    local version=$(docker exec "$container_name" psql -U neurondb -d neurondb -t -c "SELECT neurondb.version();" 2>/dev/null | xargs || echo "unknown")
                    COMPONENT_STATUS["NeuronDB"]="healthy"
                    COMPONENT_VERSION["NeuronDB"]="$version"
                    
                    # Get container stats
                    local stats=$(docker stats --no-stream --format "{{.MemUsage}}|{{.CPUPerc}}" "$container_name" 2>/dev/null || echo "N/A|N/A")
                    COMPONENT_MEMORY["NeuronDB"]=$(echo "$stats" | cut -d'|' -f1)
                    COMPONENT_CPU["NeuronDB"]=$(echo "$stats" | cut -d'|' -f2)
                    return 0
                else
                    COMPONENT_STATUS["NeuronDB"]="unhealthy"
                    return 1
                fi
            else
                COMPONENT_STATUS["NeuronDB"]="stopped"
                return 1
            fi
        else
            COMPONENT_STATUS["NeuronDB"]="not found"
            return 1
        fi
    else
        # Native mode
        if pgrep -f "postgres.*neurondb" &> /dev/null; then
            # Try to connect
            if psql -h localhost -p 5432 -U postgres -d neurondb -t -c "SELECT neurondb.version();" &> /dev/null 2>&1; then
                local version=$(psql -h localhost -p 5432 -U postgres -d neurondb -t -c "SELECT neurondb.version();" 2>/dev/null | xargs || echo "unknown")
                COMPONENT_STATUS["NeuronDB"]="healthy"
                COMPONENT_VERSION["NeuronDB"]="$version"
                return 0
            else
                COMPONENT_STATUS["NeuronDB"]="unhealthy"
                return 1
            fi
        else
            COMPONENT_STATUS["NeuronDB"]="stopped"
            return 1
        fi
    fi
}

check_neuronagent_health() {
    local container_name="neuronagent"
    
    if [ "$MODE" = "docker" ]; then
        # Docker mode
        if docker ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
            local container_status=$(docker inspect --format='{{.State.Status}}' "$container_name" 2>/dev/null || echo "not found")
            
            if [ "$container_status" = "running" ]; then
                # Check health endpoint
                if curl -sf "http://localhost:$NEURONAGENT_PORT/health" &> /dev/null; then
                    COMPONENT_STATUS["NeuronAgent"]="healthy"
                    COMPONENT_VERSION["NeuronAgent"]=$(curl -s "http://localhost:$NEURONAGENT_PORT/health" | grep -o '"version":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
                    
                    # Get container stats
                    local stats=$(docker stats --no-stream --format "{{.MemUsage}}|{{.CPUPerc}}" "$container_name" 2>/dev/null || echo "N/A|N/A")
                    COMPONENT_MEMORY["NeuronAgent"]=$(echo "$stats" | cut -d'|' -f1)
                    COMPONENT_CPU["NeuronAgent"]=$(echo "$stats" | cut -d'|' -f2)
                    return 0
                else
                    COMPONENT_STATUS["NeuronAgent"]="unhealthy"
                    return 1
                fi
            else
                COMPONENT_STATUS["NeuronAgent"]="stopped"
                return 1
            fi
        else
            COMPONENT_STATUS["NeuronAgent"]="not found"
            return 1
        fi
    else
        # Native mode
        if pgrep -f "neuronagent" &> /dev/null; then
            if curl -sf "http://localhost:$NEURONAGENT_PORT/health" &> /dev/null; then
                COMPONENT_STATUS["NeuronAgent"]="healthy"
                COMPONENT_VERSION["NeuronAgent"]=$(curl -s "http://localhost:$NEURONAGENT_PORT/health" | grep -o '"version":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
                return 0
            else
                COMPONENT_STATUS["NeuronAgent"]="unhealthy"
                return 1
            fi
        else
            COMPONENT_STATUS["NeuronAgent"]="stopped"
            return 1
        fi
    fi
}

check_neuronmcp_health() {
    local container_name="neurondb-mcp"
    
    if [ "$MODE" = "docker" ]; then
        # Docker mode
        if docker ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
            local container_status=$(docker inspect --format='{{.State.Status}}' "$container_name" 2>/dev/null || echo "not found")
            
            if [ "$container_status" = "running" ]; then
                # Check if binary exists
                if docker exec "$container_name" test -x /app/neurondb-mcp 2>/dev/null; then
                    COMPONENT_STATUS["NeuronMCP"]="healthy"
                    COMPONENT_VERSION["NeuronMCP"]="stdio"
                    
                    # Get container stats
                    local stats=$(docker stats --no-stream --format "{{.MemUsage}}|{{.CPUPerc}}" "$container_name" 2>/dev/null || echo "N/A|N/A")
                    COMPONENT_MEMORY["NeuronMCP"]=$(echo "$stats" | cut -d'|' -f1)
                    COMPONENT_CPU["NeuronMCP"]=$(echo "$stats" | cut -d'|' -f2)
                    return 0
                else
                    COMPONENT_STATUS["NeuronMCP"]="unhealthy"
                    return 1
                fi
            else
                COMPONENT_STATUS["NeuronMCP"]="stopped"
                return 1
            fi
        else
            COMPONENT_STATUS["NeuronMCP"]="not found"
            return 1
        fi
    else
        # Native mode
        if command -v neurondb-mcp &> /dev/null || [ -f "/usr/local/bin/neurondb-mcp" ]; then
            COMPONENT_STATUS["NeuronMCP"]="healthy"
            COMPONENT_VERSION["NeuronMCP"]="stdio"
            return 0
        else
            COMPONENT_STATUS["NeuronMCP"]="not found"
            return 1
        fi
    fi
}

check_neurondesktop_health() {
    local container_name="neurondesktop"
    
    if [ "$MODE" = "docker" ]; then
        # Docker mode
        if docker ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
            local container_status=$(docker inspect --format='{{.State.Status}}' "$container_name" 2>/dev/null || echo "not found")
            
            if [ "$container_status" = "running" ]; then
                # Check API health
                if curl -sf "http://localhost:$NEURONDESKTOP_PORT/health" &> /dev/null; then
                    COMPONENT_STATUS["NeuronDesktop"]="healthy"
                    COMPONENT_VERSION["NeuronDesktop"]=$(curl -s "http://localhost:$NEURONDESKTOP_PORT/health" | grep -o '"version":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
                    
                    # Get container stats
                    local stats=$(docker stats --no-stream --format "{{.MemUsage}}|{{.CPUPerc}}" "$container_name" 2>/dev/null || echo "N/A|N/A")
                    COMPONENT_MEMORY["NeuronDesktop"]=$(echo "$stats" | cut -d'|' -f1)
                    COMPONENT_CPU["NeuronDesktop"]=$(echo "$stats" | cut -d'|' -f2)
                    return 0
                else
                    COMPONENT_STATUS["NeuronDesktop"]="unhealthy"
                    return 1
                fi
            else
                COMPONENT_STATUS["NeuronDesktop"]="stopped"
                return 1
            fi
        else
            COMPONENT_STATUS["NeuronDesktop"]="not found"
            return 1
        fi
    else
        # Native mode
        if curl -sf "http://localhost:$NEURONDESKTOP_PORT/health" &> /dev/null; then
            COMPONENT_STATUS["NeuronDesktop"]="healthy"
            COMPONENT_VERSION["NeuronDesktop"]=$(curl -s "http://localhost:$NEURONDESKTOP_PORT/health" | grep -o '"version":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
            return 0
        else
            COMPONENT_STATUS["NeuronDesktop"]="not found"
            return 1
        fi
    fi
}

# ============================================================================
# Display Functions
# ============================================================================

display_status() {
    if [ "$JSON_OUTPUT" = true ]; then
        display_status_json
    else
        display_status_text
    fi
}

display_status_text() {
    print_header
    
    echo -e "${BOLD}Deployment Mode:${NC} $MODE"
    echo ""
    
    echo -e "${BOLD}┌─────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BOLD}│                    Component Status                         │${NC}"
    echo -e "${BOLD}├─────────────────┬─────────────┬───────────┬────────┬────────┤${NC}"
    printf "${BOLD}│ %-15s │ %-11s │ %-9s │ %-6s │ %-6s │${NC}\n" "Component" "Status" "Version" "Memory" "CPU"
    echo -e "${BOLD}├─────────────────┼─────────���───┼───────────┼────────┼────────┤${NC}"
    
    for component in "NeuronDB" "NeuronAgent" "NeuronMCP" "NeuronDesktop"; do
        local status="${COMPONENT_STATUS[$component]:-unknown}"
        local version="${COMPONENT_VERSION[$component]:-N/A}"
        local memory="${COMPONENT_MEMORY[$component]:-N/A}"
        local cpu="${COMPONENT_CPU[$component]:-N/A}"
        
        # Truncate version if too long
        if [ ${#version} -gt 9 ]; then
            version="${version:0:6}..."
        fi
        
        local color="$NC"
        case "$status" in
            "healthy") color="$GREEN" ;;
            "unhealthy") color="$YELLOW" ;;
            "stopped"|"not found") color="$RED" ;;
        esac
        
        printf "│ %-15s │ ${color}%-11s${NC} │ %-9s │ %-6s │ %-6s │\n" "$component" "$status" "$version" "$memory" "$cpu"
    done
    
    echo -e "${BOLD}└─────────────────┴─────────────┴───────────┴────────┴────────┘${NC}"
    echo ""
    
    # Display endpoints
    echo -e "${BOLD}Service Endpoints:${NC}"
    if [ "${COMPONENT_STATUS[NeuronDB]}" = "healthy" ]; then
        echo -e "  ${GREEN}●${NC} NeuronDB:        postgresql://localhost:$NEURONDB_PORT/neurondb"
    else
        echo -e "  ${RED}●${NC} NeuronDB:        postgresql://localhost:$NEURONDB_PORT/neurondb"
    fi
    
    if [ "${COMPONENT_STATUS[NeuronAgent]}" = "healthy" ]; then
        echo -e "  ${GREEN}●${NC} NeuronAgent:     http://localhost:$NEURONAGENT_PORT"
    else
        echo -e "  ${RED}●${NC} NeuronAgent:     http://localhost:$NEURONAGENT_PORT"
    fi
    
    if [ "${COMPONENT_STATUS[NeuronMCP]}" = "healthy" ]; then
        echo -e "  ${GREEN}●${NC} NeuronMCP:       stdio protocol"
    else
        echo -e "  ${RED}●${NC} NeuronMCP:       stdio protocol"
    fi
    
    if [ "${COMPONENT_STATUS[NeuronDesktop]}" = "healthy" ]; then
        echo -e "  ${GREEN}●${NC} NeuronDesktop:   http://localhost:$NEURONDESKTOP_PORT (API)"
        echo -e "  ${GREEN}●${NC}                  http://localhost:$NEURONDESKTOP_UI_PORT (UI)"
    else
        echo -e "  ${RED}●${NC} NeuronDesktop:   http://localhost:$NEURONDESKTOP_PORT (API)"
        echo -e "  ${RED}●${NC}                  http://localhost:$NEURONDESKTOP_UI_PORT (UI)"
    fi
    
    echo ""
    
    if [ "$WATCH_MODE" = false ]; then
        echo -e "${DIM}Run with --watch for continuous monitoring${NC}"
    fi
}

display_status_json() {
    cat << EOF
{
  "mode": "$MODE",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "components": {
    "neurondb": {
      "status": "${COMPONENT_STATUS[NeuronDB]:-unknown}",
      "version": "${COMPONENT_VERSION[NeuronDB]:-unknown}",
      "memory": "${COMPONENT_MEMORY[NeuronDB]:-N/A}",
      "cpu": "${COMPONENT_CPU[NeuronDB]:-N/A}",
      "endpoint": "postgresql://localhost:$NEURONDB_PORT/neurondb"
    },
    "neuronagent": {
      "status": "${COMPONENT_STATUS[NeuronAgent]:-unknown}",
      "version": "${COMPONENT_VERSION[NeuronAgent]:-unknown}",
      "memory": "${COMPONENT_MEMORY[NeuronAgent]:-N/A}",
      "cpu": "${COMPONENT_CPU[NeuronAgent]:-N/A}",
      "endpoint": "http://localhost:$NEURONAGENT_PORT"
    },
    "neuronmcp": {
      "status": "${COMPONENT_STATUS[NeuronMCP]:-unknown}",
      "version": "${COMPONENT_VERSION[NeuronMCP]:-unknown}",
      "memory": "${COMPONENT_MEMORY[NeuronMCP]:-N/A}",
      "cpu": "${COMPONENT_CPU[NeuronMCP]:-N/A}",
      "endpoint": "stdio"
    },
    "neurondesktop": {
      "status": "${COMPONENT_STATUS[NeuronDesktop]:-unknown}",
      "version": "${COMPONENT_VERSION[NeuronDesktop]:-unknown}",
      "memory": "${COMPONENT_MEMORY[NeuronDesktop]:-N/A}",
      "cpu": "${COMPONENT_CPU[NeuronDesktop]:-N/A}",
      "endpoints": {
        "api": "http://localhost:$NEURONDESKTOP_PORT",
        "ui": "http://localhost:$NEURONDESKTOP_UI_PORT"
      }
    }
  }
}
EOF
}

# ============================================================================
# Main Functions
# ============================================================================

run_checks() {
    check_neurondb_health
    check_neuronagent_health
    check_neuronmcp_health
    check_neurondesktop_health
}

show_help() {
    cat << EOF
${BOLD}NeuronDB Ecosystem Status Monitor${NC}

${BOLD}Usage:${NC}
    $SCRIPT_NAME [OPTIONS]

${BOLD}Options:${NC}
    --mode MODE       Deployment mode: docker or native (auto-detected if not specified)
    --watch           Continuous monitoring with updates every 5 seconds
    --json            Output in JSON format
    --help, -h        Show this help message

${BOLD}Examples:${NC}
    # Single status check
    $SCRIPT_NAME

    # Continuous monitoring
    $SCRIPT_NAME --watch

    # JSON output for integration
    $SCRIPT_NAME --json

    # Specify deployment mode
    $SCRIPT_NAME --mode docker --watch

EOF
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                MODE="$2"
                shift 2
                ;;
            --watch)
                WATCH_MODE=true
                shift
                ;;
            --json)
                JSON_OUTPUT=true
                shift
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
}

main() {
    parse_arguments "$@"
    
    detect_mode
    
    if [ "$WATCH_MODE" = true ]; then
        while true; do
            run_checks
            display_status
            
            if [ "$JSON_OUTPUT" = false ]; then
                echo -e "${DIM}Refreshing in 5 seconds... (Ctrl+C to exit)${NC}"
            fi
            sleep 5
        done
    else
        run_checks
        display_status
    fi
}

main "$@"



