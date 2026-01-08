#!/bin/bash
#
# NeuronDB Setup Script
# Self-sufficient script for all setup and installation operations
#
# Usage:
#   ./neurondb-setup.sh COMMAND [OPTIONS]
#
# Commands:
#   install        Install NeuronDB ecosystem
#   vagrant-deps   Install Vagrant dependencies
#   ecosystem      Setup complete ecosystem
#   verify         Verify installation
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
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Default configuration
COMMAND=""
VERBOSE=false
DRY_RUN=false
MODE="docker"
COMPONENTS="all"

#=========================================================================
# SELF-SUFFICIENT LOGGING FUNCTIONS
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

print_header() {
    echo ""
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  ${BOLD}NeuronDB Setup${NC}                                        ${BLUE}║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

#=========================================================================
# HELP FUNCTION
#=========================================================================

show_help() {
    cat << EOF
${BOLD}NeuronDB Setup${NC}

${BOLD}Usage:${NC}
    ${SCRIPT_NAME} COMMAND [OPTIONS]

${BOLD}Commands:${NC}
    install        Install NeuronDB ecosystem
    vagrant-deps   Install Vagrant dependencies
    ecosystem      Setup complete ecosystem
    verify         Verify installation
    generate-passwords Generate secure passwords for deployment

${BOLD}Install Options:${NC}
    --mode MODE         Installation mode: docker, deb, rpm (default: docker)
    --components LIST   Comma-separated list: neurondb,neuronagent,neuronmcp,neurondesktop (default: all)

${BOLD}Global Options:${NC}
    --dry-run           Preview changes without applying
    -h, --help          Show this help message
    -v, --verbose       Enable verbose output
    -V, --version       Show version information

${BOLD}Examples:${NC}
    # Install with Docker
    ${SCRIPT_NAME} install --mode docker

    # Install specific components
    ${SCRIPT_NAME} install --components neurondb,neuronagent

    # Setup complete ecosystem
    ${SCRIPT_NAME} ecosystem

    # Verify installation
    ${SCRIPT_NAME} verify

    # Generate secure passwords
    ${SCRIPT_NAME} generate-passwords > .env.secure

EOF
}

#=========================================================================
# INSTALL COMMAND
#=========================================================================

install_command() {
    shift
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --mode)
                MODE="$2"
                shift 2
                ;;
            --components)
                COMPONENTS="$2"
                shift 2
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
    
    print_header
    log_info "Installing NeuronDB ecosystem (mode: $MODE, components: $COMPONENTS)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would install with mode=$MODE, components=$COMPONENTS"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    
    case "$MODE" in
        docker)
            log_info "Installing via Docker..."
            if command -v docker &> /dev/null && command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
                if docker compose version &> /dev/null; then
                    docker compose up -d
                else
                    docker-compose up -d
                fi
                log_success "Docker installation completed"
            else
                log_error "Docker or Docker Compose not found"
                exit 1
            fi
            ;;
        deb)
            log_info "Installing DEB packages..."
            log_warning "DEB package installation not yet implemented"
            ;;
        rpm)
            log_info "Installing RPM packages..."
            log_warning "RPM package installation not yet implemented"
            ;;
        *)
            log_error "Unknown installation mode: $MODE"
            exit 1
            ;;
    esac
}

#=========================================================================
# VAGRANT-DEPS COMMAND
#=========================================================================

vagrant_deps_command() {
    shift
    
    print_header
    log_info "Installing Vagrant dependencies..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would install Vagrant dependencies"
        return 0
    fi
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            log_info "Installing on Debian/Ubuntu..."
            sudo apt-get update
            sudo apt-get install -y virtualbox vagrant
        elif command -v yum &> /dev/null || command -v dnf &> /dev/null; then
            log_info "Installing on RHEL/CentOS/Rocky..."
            sudo yum install -y VirtualBox vagrant || sudo dnf install -y VirtualBox vagrant
        else
            log_error "Unsupported Linux distribution"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        log_info "Installing on macOS..."
        if command -v brew &> /dev/null; then
            brew install --cask virtualbox vagrant
        else
            log_error "Homebrew not found. Install VirtualBox and Vagrant manually."
            exit 1
        fi
    else
        log_error "Unsupported operating system"
        exit 1
    fi
    
    log_success "Vagrant dependencies installed"
}

#=========================================================================
# ECOSYSTEM COMMAND
#=========================================================================

ecosystem_command() {
    shift
    
    print_header
    log_info "Setting up complete NeuronDB ecosystem..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would setup ecosystem"
        return 0
    fi
    
    # Check prerequisites
    log_info "Checking prerequisites..."
    
    local missing_deps=()
    
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        missing_deps+=("docker-compose")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Install them first or use: ${SCRIPT_NAME} vagrant-deps"
        exit 1
    fi
    
    # Install
    install_command --mode docker --components all
    
    log_success "Ecosystem setup completed"
}

#=========================================================================
# VERIFY COMMAND
#=========================================================================

verify_command() {
    shift
    
    print_header
    log_info "Verifying installation..."
    
    local checks_passed=0
    local checks_failed=0
    
    # Check Docker
    if command -v docker &> /dev/null; then
        if docker info >/dev/null 2>&1; then
            log_success "Docker is running"
            ((checks_passed++))
        else
            log_error "Docker daemon not running"
            ((checks_failed++))
        fi
    else
        log_error "Docker not installed"
        ((checks_failed++))
    fi
    
    # Check containers
    if docker ps --format "{{.Names}}" | grep -q "neurondb"; then
        log_success "NeuronDB containers running"
        ((checks_passed++))
    else
        log_warning "NeuronDB containers not running"
        ((checks_failed++))
    fi
    
    echo ""
    echo "Checks passed: $checks_passed"
    echo "Checks failed: $checks_failed"
    
    [[ $checks_failed -eq 0 ]] && return 0 || return 1
}

#=========================================================================
# GENERATE-PASSWORDS COMMAND
#=========================================================================

generate_passwords_command() {
    shift
    
    # Generate secure passwords
    echo "# Secure passwords generated on $(date)"
    echo "# Copy these values to your .env file"
    echo ""
    echo "# PostgreSQL / NeuronDB"
    echo "POSTGRES_USER=neurondb"
    echo "POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -d '\n')"
    echo "POSTGRES_DB=neurondb"
    echo ""
    echo "# NeuronAgent (must match POSTGRES_PASSWORD)"
    echo "DB_PASSWORD=\${POSTGRES_PASSWORD}"
    echo ""
    echo "# NeuronMCP (must match POSTGRES_PASSWORD)"
    echo "NEURONDB_PASSWORD=\${POSTGRES_PASSWORD}"
    echo ""
    echo "# Generate NeuronAgent API key (if needed)"
    echo "# Use: openssl rand -hex 32"
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
            --mode)
                # Handled by install command
                shift 2
                ;;
            --components)
                # Handled by install command
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
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

#=========================================================================
# MAIN FUNCTION
#=========================================================================

main() {
    parse_arguments "$@"
    
    case "$COMMAND" in
        install)
            install_command "$@"
            ;;
        vagrant-deps)
            vagrant_deps_command "$@"
            ;;
        ecosystem)
            ecosystem_command "$@"
            ;;
        verify)
            verify_command "$@"
            ;;
        generate-passwords)
            generate_passwords_command "$@"
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

main "$@"

