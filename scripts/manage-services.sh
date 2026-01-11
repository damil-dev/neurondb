#!/bin/bash
# ====================================================================
# Service Management Script
# ====================================================================
# Unified script for managing NeuronDB ecosystem services
# ====================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME=$(basename "$0")

# Source helper functions
source "$SCRIPT_DIR/install-helpers.sh"

# Available services
SERVICES="neuronmcp neuronagent neurondesktop-api"

# Detect init system
INIT_SYSTEM=$(detect_init_system)

show_help() {
    cat << EOF
Service Management Script

Usage: $SCRIPT_NAME COMMAND [SERVICE...]

Commands:
    start       Start service(s)
    stop        Stop service(s)
    restart     Restart service(s)
    status      Show service status
    enable      Enable service(s) to start on boot
    disable     Disable service(s) from starting on boot
    logs        Show service logs
    health      Check service health

Services:
    neuronmcp          NeuronMCP server
    neuronagent        NeuronAgent server
    neurondesktop-api  NeuronDesktop API server
    all                All services (default)

Examples:
    # Start all services
    $SCRIPT_NAME start

    # Start specific service
    $SCRIPT_NAME start neuronmcp

    # Check status
    $SCRIPT_NAME status

    # View logs
    $SCRIPT_NAME logs neuronagent

EOF
}

# systemd functions
systemd_start() {
    local service=$1
    if ! systemctl list-unit-files | grep -q "^${service}.service"; then
        print_warning "$service is not installed. Install it first with: sudo ./scripts/install-*.sh --enable-service"
        return 1
    fi
    if systemctl is-active --quiet "$service"; then
        print_info "$service is already running"
    else
        if sudo systemctl start "$service" 2>/dev/null; then
            print_success "$service started"
        else
            print_error "Failed to start $service. Check logs: sudo journalctl -u $service"
            return 1
        fi
    fi
}

systemd_stop() {
    local service=$1
    if ! systemctl list-unit-files | grep -q "^${service}.service"; then
        print_warning "$service is not installed"
        return 1
    fi
    if systemctl is-active --quiet "$service"; then
        if sudo systemctl stop "$service" 2>/dev/null; then
            print_success "$service stopped"
        else
            print_error "Failed to stop $service"
            return 1
        fi
    else
        print_info "$service is not running"
    fi
}

systemd_restart() {
    local service=$1
    if ! systemctl list-unit-files | grep -q "^${service}.service"; then
        print_warning "$service is not installed. Install it first with: sudo ./scripts/install-*.sh --enable-service"
        return 1
    fi
    if sudo systemctl restart "$service" 2>/dev/null; then
        print_success "$service restarted"
    else
        print_error "Failed to restart $service. Check logs: sudo journalctl -u $service"
        return 1
    fi
}

systemd_status() {
    local service=$1
    if ! systemctl list-unit-files | grep -q "^${service}.service"; then
        print_warning "$service is not installed"
        return 1
    fi
    systemctl status "$service" --no-pager -l || true
}

systemd_enable() {
    local service=$1
    if ! systemctl list-unit-files | grep -q "^${service}.service"; then
        print_warning "$service is not installed. Install it first with: sudo ./scripts/install-*.sh --enable-service"
        return 1
    fi
    if sudo systemctl enable "$service" 2>/dev/null; then
        print_success "$service enabled"
    else
        print_error "Failed to enable $service"
        return 1
    fi
}

systemd_disable() {
    local service=$1
    if ! systemctl list-unit-files | grep -q "^${service}.service"; then
        print_warning "$service is not installed"
        return 1
    fi
    if sudo systemctl disable "$service" 2>/dev/null; then
        print_success "$service disabled"
    else
        print_error "Failed to disable $service"
        return 1
    fi
}

systemd_logs() {
    local service=$1
    if ! systemctl list-unit-files | grep -q "^${service}.service"; then
        print_warning "$service is not installed"
        return 1
    fi
    sudo journalctl -u "$service" -f --no-pager
}

# launchd functions (macOS)
launchd_start() {
    local service=$1
    local plist="com.neurondb.$service"
    if ! launchctl list | grep -q "$plist" && ! sudo launchctl list | grep -q "$plist" 2>/dev/null; then
        print_warning "$service is not installed. Install it first with: sudo ./scripts/install-*.sh --enable-service"
        return 1
    fi
    if launchctl start "$plist" 2>/dev/null || sudo launchctl start "$plist" 2>/dev/null; then
        print_success "$service started"
    else
        print_error "Failed to start $service"
        return 1
    fi
}

launchd_stop() {
    local service=$1
    local plist="com.neurondb.$service"
    if ! launchctl list | grep -q "$plist" && ! sudo launchctl list | grep -q "$plist" 2>/dev/null; then
        print_warning "$service is not installed"
        return 1
    fi
    if launchctl stop "$plist" 2>/dev/null || sudo launchctl stop "$plist" 2>/dev/null; then
        print_success "$service stopped"
    else
        print_error "Failed to stop $service"
        return 1
    fi
}

launchd_restart() {
    local service=$1
    launchd_stop "$service"
    sleep 1
    launchd_start "$service"
}

launchd_status() {
    local service=$1
    local plist="com.neurondb.$service"
    launchctl list | grep "$plist" || print_info "$service is not loaded"
}

launchd_enable() {
    local plist="com.neurondb.$1"
    if [ -f "$HOME/Library/LaunchAgents/$plist.plist" ]; then
        launchctl load "$HOME/Library/LaunchAgents/$plist.plist"
    elif [ -f "/Library/LaunchDaemons/$plist.plist" ]; then
        sudo launchctl load "/Library/LaunchDaemons/$plist.plist"
    else
        print_error "Service file not found for $1"
        return 1
    fi
    print_success "$1 enabled"
}

launchd_disable() {
    local plist="com.neurondb.$1"
    launchctl unload "$HOME/Library/LaunchAgents/$plist.plist" 2>/dev/null || \
        sudo launchctl unload "/Library/LaunchDaemons/$plist.plist" 2>/dev/null || true
    print_success "$1 disabled"
}

launchd_logs() {
    local service=$1
    tail -f "$HOME/Library/Logs/neurondb/$service.log" 2>/dev/null || \
        tail -f "/usr/local/var/log/neurondb/$service.log" 2>/dev/null || \
        print_error "Log file not found for $service"
}

# Health check
check_health() {
    local service=$1
    case $service in
        neuronagent)
            if curl -sf http://localhost:8080/health >/dev/null 2>&1; then
                print_success "NeuronAgent is healthy"
            else
                print_error "NeuronAgent health check failed"
            fi
            ;;
        neurondesktop-api)
            if curl -sf http://localhost:8081/health >/dev/null 2>&1; then
                print_success "NeuronDesktop API is healthy"
            else
                print_error "NeuronDesktop API health check failed"
            fi
            ;;
        neuronmcp)
            print_info "NeuronMCP uses stdio protocol - cannot check health via HTTP"
            ;;
    esac
}

# Main command handler
handle_command() {
    local cmd=$1
    shift
    local services=${@:-all}
    
    if [ "$services" = "all" ]; then
        services=$SERVICES
    fi
    
    case $INIT_SYSTEM in
        systemd)
            for service in $services; do
                case $cmd in
                    start) systemd_start "$service" ;;
                    stop) systemd_stop "$service" ;;
                    restart) systemd_restart "$service" ;;
                    status) systemd_status "$service" ;;
                    enable) systemd_enable "$service" ;;
                    disable) systemd_disable "$service" ;;
                    logs) systemd_logs "$service" ;;
                    health) check_health "$service" ;;
                    *) print_error "Unknown command: $cmd" ; exit 1 ;;
                esac
            done
            ;;
        launchd)
            for service in $services; do
                # Map service names
                case $service in
                    neurondesktop-api) service="neurondesktop-api" ;;
                esac
                
                case $cmd in
                    start) launchd_start "$service" ;;
                    stop) launchd_stop "$service" ;;
                    restart) launchd_restart "$service" ;;
                    status) launchd_status "$service" ;;
                    enable) launchd_enable "$service" ;;
                    disable) launchd_disable "$service" ;;
                    logs) launchd_logs "$service" ;;
                    health) check_health "$service" ;;
                    *) print_error "Unknown command: $cmd" ; exit 1 ;;
                esac
            done
            ;;
        *)
            print_error "Unknown init system: $INIT_SYSTEM"
            print_info "Supported systems: systemd, launchd"
            exit 1
            ;;
    esac
}

# Parse arguments
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

COMMAND=$1
shift

case $COMMAND in
    -h|--help)
        show_help
        exit 0
        ;;
    start|stop|restart|status|enable|disable|logs|health)
        handle_command "$COMMAND" "$@"
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

