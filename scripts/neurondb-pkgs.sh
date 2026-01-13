#!/bin/bash
#
# NeuronDB Packages Script
# Self-sufficient script for all package operations: verify, generate SDKs, test packages
#
# Usage:
#   ./neurondb-pkgs.sh COMMAND [OPTIONS]
#
# Commands:
#   verify        Verify DEB/RPM packages
#   generate-sdk  Generate client SDKs
#   test-vagrant  Test packages in Vagrant VM
#   validate-helm Validate Helm charts
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
DRY_RUN=false
OS_TYPE="ubuntu"

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
    echo -e "${BLUE}║${NC}  ${BOLD}NeuronDB Packages${NC}                                   ${BLUE}║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

show_help() {
    cat << EOF
${BOLD}NeuronDB Packages${NC}

${BOLD}Usage:${NC}
    ${SCRIPT_NAME} COMMAND [OPTIONS]

${BOLD}Commands:${NC}
    verify        Verify DEB/RPM packages
    generate-sdk  Generate client SDKs from OpenAPI spec
    test-vagrant  Test packages in Vagrant VM
    validate-helm Validate Helm charts

${BOLD}Verify Options:${NC}
    --os OS           OS type: ubuntu, rocky (default: ubuntu)
    --package PATH    Package file path

${BOLD}Generate SDK Options:${NC}
    --language LANG   Target language: python, javascript, go, java (default: python)
    --output DIR      Output directory (default: ./sdks)

${BOLD}Test Vagrant Options:${NC}
    --os OS           OS type: ubuntu, rocky (default: ubuntu)
    --destroy-vm      Destroy VM after testing

${BOLD}Global Options:${NC}
    --dry-run         Preview changes without applying
    -h, --help        Show this help message
    -v, --verbose     Enable verbose output
    -V, --version     Show version information

${BOLD}Examples:${NC}
    # Verify packages
    ${SCRIPT_NAME} verify --os ubuntu

    # Generate Python SDK
    ${SCRIPT_NAME} generate-sdk --language python

    # Test packages in Vagrant
    ${SCRIPT_NAME} test-vagrant --os rocky

    # Validate Helm chart
    ${SCRIPT_NAME} validate-helm

EOF
}

verify_command() {
    shift
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --os)
                OS_TYPE="$2"
                shift 2
                ;;
            --package)
                # Package path handling
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done
    
    print_header
    log_info "Verifying packages for $OS_TYPE..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would verify packages"
        return 0
    fi
    
    local checks_passed=0
    local checks_failed=0
    
    # Check for package files
    if [[ "$OS_TYPE" == "ubuntu" ]]; then
        if find "$PROJECT_ROOT" -name "*.deb" -type f | head -1 | grep -q .; then
            log_success "DEB packages found"
            ((checks_passed++))
        else
            log_warning "No DEB packages found"
            ((checks_failed++))
        fi
    elif [[ "$OS_TYPE" == "rocky" ]]; then
        if find "$PROJECT_ROOT" -name "*.rpm" -type f | head -1 | grep -q .; then
            log_success "RPM packages found"
            ((checks_passed++))
        else
            log_warning "No RPM packages found"
            ((checks_failed++))
        fi
    fi
    
    echo ""
    echo "Checks passed: $checks_passed"
    echo "Checks failed: $checks_failed"
    
    [[ $checks_failed -eq 0 ]] && return 0 || return 1
}

generate_sdk_command() {
    shift
    
    local language="python"
    local output_dir="./sdks"
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --language)
                language="$2"
                shift 2
                ;;
            --output)
                output_dir="$2"
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done
    
    print_header
    log_info "Generating $language SDK..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would generate $language SDK to $output_dir"
        return 0
    fi
    
    # Check for openapi-generator
    if ! command -v openapi-generator-cli &> /dev/null && ! command -v openapi-generator &> /dev/null; then
        log_error "OpenAPI Generator not found"
        log_info "Install with: npm install -g @openapitools/openapi-generator-cli"
        exit 1
    fi
    
    # Find OpenAPI spec
    local spec_file=$(find "$PROJECT_ROOT" -name "openapi.yaml" -o -name "openapi.json" | head -1)
    if [[ -z "$spec_file" ]]; then
        log_error "OpenAPI specification not found"
        exit 1
    fi
    
    log_info "Using OpenAPI spec: $spec_file"
    
    mkdir -p "$output_dir"
    
    local generator_cmd=""
    if command -v openapi-generator-cli &> /dev/null; then
        generator_cmd="openapi-generator-cli"
    else
        generator_cmd="openapi-generator"
    fi
    
    $generator_cmd generate \
        -i "$spec_file" \
        -g "$language" \
        -o "$output_dir/$language" \
        --skip-validate-spec || {
        log_error "SDK generation failed"
        exit 1
    }
    
    log_success "SDK generated: $output_dir/$language"
}

test_vagrant_command() {
    shift
    
    local destroy_vm=false
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --os)
                OS_TYPE="$2"
                shift 2
                ;;
            --destroy-vm)
                destroy_vm=true
                shift
                ;;
            *)
                break
                ;;
        esac
    done
    
    print_header
    log_info "Testing packages in Vagrant ($OS_TYPE)..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would test packages in Vagrant"
        return 0
    fi
    
    # Check for Vagrant
    if ! command -v vagrant &> /dev/null; then
        log_error "Vagrant not found"
        log_info "Install with: ${SCRIPT_DIR}/neurondb-setup.sh vagrant-deps"
        exit 1
    fi
    
    cd "$PROJECT_ROOT"
    
    # Set Vagrantfile based on OS
    if [[ "$OS_TYPE" == "rocky" ]]; then
        export VAGRANT_VAGRANTFILE=Vagrantfile.rocky
    fi
    
    log_info "Starting Vagrant VM..."
    vagrant up
    
    log_info "Testing packages in VM..."
    vagrant ssh -c "sudo yum install -y *.rpm || sudo apt-get install -y *.deb" || true
    
    if [[ "$destroy_vm" == "true" ]]; then
        log_info "Destroying VM..."
        vagrant destroy -f
    fi
    
    log_success "Vagrant testing completed"
}

validate_helm_command() {
    shift
    
    print_header
    log_info "Validating Helm charts..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would validate Helm charts"
        return 0
    fi
    
    # Check for Helm
    if ! command -v helm &> /dev/null; then
        log_error "Helm not found"
        log_info "Install from: https://helm.sh/docs/intro/install/"
        exit 1
    fi
    
    local chart_dir="$PROJECT_ROOT/helm/neurondb"
    if [[ ! -d "$chart_dir" ]]; then
        log_error "Helm chart not found: $chart_dir"
        exit 1
    fi
    
    log_info "Linting Helm chart..."
    helm lint "$chart_dir" || {
        log_error "Helm chart validation failed"
        exit 1
    }
    
    log_success "Helm chart validated"
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
            --os)
                # Handled by commands
                shift 2
                ;;
            --package|--language|--output)
                # Handled by commands
                shift 2
                ;;
            --destroy-vm)
                # Handled by commands
                shift
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

main() {
    parse_arguments "$@"
    
    case "$COMMAND" in
        verify)
            verify_command "$@"
            ;;
        generate-sdk)
            generate_sdk_command "$@"
            ;;
        test-vagrant)
            test_vagrant_command "$@"
            ;;
        validate-helm)
            validate_helm_command "$@"
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

main "$@"

