#!/bin/bash
#
# NeuronDB Docker Management Script
# Modular and clean script for managing NeuronDB ecosystem containers
#
# Usage:
#   ./docker.sh --version | -v                    # Show version
#   ./docker.sh --list                             # List available services
#   ./docker.sh --build [service] [--profile]      # Build containers
#   ./docker.sh --run [service] [--profile]        # Run containers
#   ./docker.sh --all --build [--profile]          # Build all services
#   ./docker.sh --all --run [--profile]            # Run all services
#

set -euo pipefail

# Script metadata
SCRIPT_VERSION="1.0.0"
SCRIPT_NAME="docker.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Available services
AVAILABLE_SERVICES=("neurondb" "neuronagent" "neuronmcp" "neurondesktop")
AVAILABLE_PROFILES=("cpu" "cuda" "rocm" "metal" "default")

# Default values
PROFILE="default"
ACTION=""
SERVICES=()
ALL_SERVICES=false

# ============================================================================
# Utility Functions
# ============================================================================

print_error() {
    echo -e "${RED}ERROR:${NC} $1" >&2
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# ============================================================================
# Validation Functions
# ============================================================================

validate_compose_file() {
    if [[ ! -f "${COMPOSE_FILE}" ]]; then
        print_error "Docker compose file not found: ${COMPOSE_FILE}"
        exit 1
    fi
}

validate_service() {
    local service=$1
    if [[ ! " ${AVAILABLE_SERVICES[@]} " =~ " ${service} " ]]; then
        print_error "Unknown service: ${service}"
        print_info "Available services: ${AVAILABLE_SERVICES[*]}"
        exit 1
    fi
}

validate_profile() {
    local profile=$1
    if [[ ! " ${AVAILABLE_PROFILES[@]} " =~ " ${profile} " ]]; then
        print_error "Unknown profile: ${profile}"
        print_info "Available profiles: ${AVAILABLE_PROFILES[*]}"
        exit 1
    fi
}

# ============================================================================
# Core Functions
# ============================================================================

get_compose_cmd() {
    if command -v docker-compose &> /dev/null; then
        echo "docker-compose"
    elif docker compose version &> /dev/null 2>&1; then
        echo "docker compose"
    else
        print_error "Docker Compose is not available"
        exit 1
    fi
}

is_compose_v2() {
    local cmd=$(get_compose_cmd)
    if [[ "${cmd}" == "docker compose" ]]; then
        return 0
    else
        return 1
    fi
}

run_compose() {
    local compose_cmd=$(get_compose_cmd)
    cd "${PROJECT_ROOT}"
    
    # For docker-compose v1, profile flag comes before the subcommand
    # For docker compose v2, profile flag comes after the subcommand
    if is_compose_v2; then
        ${compose_cmd} -f "${COMPOSE_FILE}" "$@"
    else
        # docker-compose v1 syntax
        ${compose_cmd} -f "${COMPOSE_FILE}" "$@"
    fi
    
    local exit_code=$?
    cd - > /dev/null
    return ${exit_code}
}

show_version() {
    echo "NeuronDB Docker Management Script"
    echo "Version: ${SCRIPT_VERSION}"
    echo ""
    echo "Docker Compose file: ${COMPOSE_FILE}"
    if command -v docker-compose &> /dev/null || command -v docker &> /dev/null; then
        if command -v docker &> /dev/null; then
            echo "Docker version: $(docker --version)"
        fi
        if docker compose version &> /dev/null; then
            echo "Docker Compose version: $(docker compose version)"
        fi
    fi
}

list_services() {
    print_info "Available services:"
    for service in "${AVAILABLE_SERVICES[@]}"; do
        echo "  - ${service}"
    done
    echo ""
    print_info "Available profiles:"
    for profile in "${AVAILABLE_PROFILES[@]}"; do
        echo "  - ${profile}"
    done
}

build_service() {
    local service=$1
    local profile=$2
    
    print_info "Building ${service} with profile: ${profile}"
    
    validate_service "${service}"
    validate_profile "${profile}"
    
    # Map service names to compose service names
    local compose_service=""
    case "${service}" in
        neurondb)
            case "${profile}" in
                cuda) compose_service="neurondb-cuda" ;;
                rocm) compose_service="neurondb-rocm" ;;
                metal) compose_service="neurondb-metal" ;;
                *) compose_service="neurondb" ;;
            esac
            ;;
        neuronagent)
            case "${profile}" in
                cuda) compose_service="neuronagent-cuda" ;;
                rocm) compose_service="neuronagent-rocm" ;;
                metal) compose_service="neuronagent-metal" ;;
                *) compose_service="neuronagent" ;;
            esac
            ;;
        neuronmcp)
            case "${profile}" in
                cuda) compose_service="neuronmcp-cuda" ;;
                rocm) compose_service="neuronmcp-rocm" ;;
                metal) compose_service="neuronmcp-metal" ;;
                *) compose_service="neuronmcp" ;;
            esac
            ;;
        neurondesktop)
            compose_service="neurondesk-api neurondesk-frontend"
            ;;
    esac
    
    if [[ -z "${compose_service}" ]]; then
        print_error "No compose service mapped for ${service} with profile ${profile}"
        return 1
    fi
    
    # Split multiple services if needed
    local services_array=(${compose_service})
    local compose_cmd=$(get_compose_cmd)
    
    if is_compose_v2; then
        # docker compose v2: --profile comes after subcommand
        cd "${PROJECT_ROOT}"
        ${compose_cmd} -f "${COMPOSE_FILE}" build --profile "${profile}" "${services_array[@]}"
        local exit_code=$?
        cd - > /dev/null
    else
        # docker-compose v1: --profile comes before subcommand
        cd "${PROJECT_ROOT}"
        ${compose_cmd} -f "${COMPOSE_FILE}" --profile "${profile}" build "${services_array[@]}"
        local exit_code=$?
        cd - > /dev/null
    fi
    
    if [[ ${exit_code} -eq 0 ]]; then
        print_success "Built ${service} (${profile})"
    else
        print_error "Failed to build ${service} (${profile})"
        return 1
    fi
}

run_service() {
    local service=$1
    local profile=$2
    
    print_info "Running ${service} with profile: ${profile}"
    
    validate_service "${service}"
    validate_profile "${profile}"
    
    # Map service names to compose service names
    local compose_service=""
    case "${service}" in
        neurondb)
            case "${profile}" in
                cuda) compose_service="neurondb-cuda" ;;
                rocm) compose_service="neurondb-rocm" ;;
                metal) compose_service="neurondb-metal" ;;
                *) compose_service="neurondb" ;;
            esac
            ;;
        neuronagent)
            case "${profile}" in
                cuda) compose_service="neuronagent-cuda" ;;
                rocm) compose_service="neuronagent-rocm" ;;
                metal) compose_service="neuronagent-metal" ;;
                *) compose_service="neuronagent" ;;
            esac
            ;;
        neuronmcp)
            case "${profile}" in
                cuda) compose_service="neuronmcp-cuda" ;;
                rocm) compose_service="neuronmcp-rocm" ;;
                metal) compose_service="neuronmcp-metal" ;;
                *) compose_service="neuronmcp" ;;
            esac
            ;;
        neurondesktop)
            compose_service="neurondesk-init neurondesk-api neurondesk-frontend"
            ;;
    esac
    
    if [[ -z "${compose_service}" ]]; then
        print_error "No compose service mapped for ${service} with profile ${profile}"
        return 1
    fi
    
    # Split multiple services if needed
    local services_array=(${compose_service})
    local compose_cmd=$(get_compose_cmd)
    
    if is_compose_v2; then
        # docker compose v2: --profile comes after subcommand
        cd "${PROJECT_ROOT}"
        ${compose_cmd} -f "${COMPOSE_FILE}" up -d --profile "${profile}" "${services_array[@]}"
        local exit_code=$?
        cd - > /dev/null
    else
        # docker-compose v1: --profile comes before subcommand
        cd "${PROJECT_ROOT}"
        ${compose_cmd} -f "${COMPOSE_FILE}" --profile "${profile}" up -d "${services_array[@]}"
        local exit_code=$?
        cd - > /dev/null
    fi
    
    if [[ ${exit_code} -eq 0 ]]; then
        print_success "Started ${service} (${profile})"
    else
        print_error "Failed to start ${service} (${profile})"
        return 1
    fi
}

build_all() {
    local profile=$1
    print_info "Building all services with profile: ${profile}"
    
    validate_profile "${profile}"
    
    local compose_cmd=$(get_compose_cmd)
    
    if is_compose_v2; then
        cd "${PROJECT_ROOT}"
        ${compose_cmd} -f "${COMPOSE_FILE}" build --profile "${profile}"
        local exit_code=$?
        cd - > /dev/null
    else
        cd "${PROJECT_ROOT}"
        ${compose_cmd} -f "${COMPOSE_FILE}" --profile "${profile}" build
        local exit_code=$?
        cd - > /dev/null
    fi
    
    if [[ ${exit_code} -eq 0 ]]; then
        print_success "Built all services (${profile})"
    else
        print_error "Failed to build all services (${profile})"
        return 1
    fi
}

run_all() {
    local profile=$1
    print_info "Running all services with profile: ${profile}"
    
    validate_profile "${profile}"
    
    local compose_cmd=$(get_compose_cmd)
    
    if is_compose_v2; then
        cd "${PROJECT_ROOT}"
        ${compose_cmd} -f "${COMPOSE_FILE}" up -d --profile "${profile}"
        local exit_code=$?
        cd - > /dev/null
    else
        cd "${PROJECT_ROOT}"
        ${compose_cmd} -f "${COMPOSE_FILE}" --profile "${profile}" up -d
        local exit_code=$?
        cd - > /dev/null
    fi
    
    if [[ ${exit_code} -eq 0 ]]; then
        print_success "Started all services (${profile})"
    else
        print_error "Failed to start all services (${profile})"
        return 1
    fi
}

# ============================================================================
# Argument Parsing
# ============================================================================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --version|-v)
                show_version
                exit 0
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            --list)
                list_services
                exit 0
                ;;
            --build)
                ACTION="build"
                shift
                ;;
            --run)
                ACTION="run"
                shift
                ;;
            --all)
                ALL_SERVICES=true
                shift
                ;;
            --profile)
                PROFILE="$2"
                shift 2
                ;;
            neurondb|neuronagent|neuronmcp|neurondesktop)
                if [[ "${ALL_SERVICES}" == false ]]; then
                    SERVICES+=("$1")
                fi
                shift
                ;;
            *)
                print_error "Unknown argument: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

show_usage() {
    cat << EOF
Usage: ${SCRIPT_NAME} [OPTIONS] [SERVICE...]

Options:
    --version, -v              Show script version and docker info
    --list                     List available services and profiles
    --build                    Build container(s)
    --run                      Run container(s)
    --all                      Apply action to all services
    --profile PROFILE          Specify profile (cpu, cuda, rocm, metal, default)
    
Services:
    neurondb                   NeuronDB database service
    neuronagent                NeuronAgent service
    neuronmcp                  NeuronMCP service
    neurondesktop              NeuronDesktop service

Examples:
    ${SCRIPT_NAME} --version
    ${SCRIPT_NAME} --list
    ${SCRIPT_NAME} --build neurondb --profile cpu
    ${SCRIPT_NAME} --run neuronagent --profile cuda
    ${SCRIPT_NAME} --all --build --profile default
    ${SCRIPT_NAME} --all --run --profile cuda
    ${SCRIPT_NAME} neurondb neuronagent --build --profile rocm

EOF
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    # Check if docker is available
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if docker compose is available (will be checked by get_compose_cmd when needed)
    
    # Validate compose file exists
    validate_compose_file
    
    # Parse arguments
    if [[ $# -eq 0 ]]; then
        show_usage
        exit 0
    fi
    
    parse_arguments "$@"
    
    # Execute action
    if [[ -z "${ACTION}" ]]; then
        print_error "No action specified. Use --build or --run"
        show_usage
        exit 1
    fi
    
    if [[ "${ALL_SERVICES}" == true ]]; then
        # Handle --all
        case "${ACTION}" in
            build)
                build_all "${PROFILE}"
                ;;
            run)
                run_all "${PROFILE}"
                ;;
        esac
    elif [[ ${#SERVICES[@]} -eq 0 ]]; then
        print_error "No services specified. Use --all or specify service names"
        show_usage
        exit 1
    else
        # Handle specific services
        local failed=0
        for service in "${SERVICES[@]}"; do
            case "${ACTION}" in
                build)
                    if ! build_service "${service}" "${PROFILE}"; then
                        ((failed++))
                    fi
                    ;;
                run)
                    if ! run_service "${service}" "${PROFILE}"; then
                        ((failed++))
                    fi
                    ;;
            esac
        done
        
        if [[ ${failed} -gt 0 ]]; then
            print_error "${failed} service(s) failed"
            exit 1
        fi
    fi
}

# Run main function
main "$@"

