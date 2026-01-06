#!/bin/bash
#
# Validate Helm Chart
# Checks Helm chart for common issues and validates templates
#
# Usage: ./scripts/validate-helm-chart.sh [--dry-run]
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_VERSION="2.0.0"
SCRIPT_NAME="$(basename "$0")"

DRY_RUN=false
VERBOSE=false

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[✗]${NC} $*" >&2
}

log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "[DEBUG] $*"
    fi
}

show_version() {
    echo "$SCRIPT_NAME version $SCRIPT_VERSION"
    exit 0
}

show_help() {
    cat << EOF
Usage: $SCRIPT_NAME [OPTIONS]

Validates the NeuronDB Helm chart for common issues.

Options:
    --dry-run        Only check, don't render templates
    -h, --help       Show this help message
    -V, --version    Show version information
    -v, --verbose    Enable verbose output

Examples:
    $SCRIPT_NAME
    $SCRIPT_NAME --dry-run --verbose
EOF
    exit 0
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                show_help
                ;;
            -V|--version)
                show_version
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

check_helm_installed() {
    if ! command -v helm &> /dev/null; then
        log_error "Helm is not installed. Please install Helm 3.8+ first."
        log_info "Visit: https://helm.sh/docs/intro/install/"
        exit 1
    fi
    
    local helm_version
    helm_version=$(helm version --short 2>/dev/null || echo "unknown")
    log_verbose "Helm version: $helm_version"
}

validate_chart_structure() {
    log_info "Validating chart structure..."
    
    local chart_dir="helm/neurondb"
    local required_files=(
        "$chart_dir/Chart.yaml"
        "$chart_dir/values.yaml"
        "$chart_dir/templates/_helpers.tpl"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "Required file missing: $file"
            exit 1
        fi
    done
    
    log_success "Chart structure is valid"
}

validate_templates() {
    log_info "Validating Helm templates..."
    
    local chart_dir="helm/neurondb"
    
    if [ "$DRY_RUN" = false ]; then
        log_info "Rendering templates with default values..."
        if helm template "$chart_dir" > /dev/null 2>&1; then
            log_success "Templates render successfully"
        else
            log_error "Template rendering failed"
            helm template "$chart_dir" 2>&1 | head -20
            exit 1
        fi
        
        log_info "Linting chart..."
        if helm lint "$chart_dir" 2>&1; then
            log_success "Chart linting passed"
        else
            log_warn "Chart linting found issues (may be non-critical)"
        fi
    else
        log_info "Skipping template rendering (dry-run mode)"
    fi
}

check_required_values() {
    log_info "Checking required values..."
    
    local chart_dir="helm/neurondb"
    local values_file="$chart_dir/values.yaml"
    
    # Check for critical values
    local critical_keys=(
        "neurondb.image.tag"
        "neurondb.postgresql.username"
        "secrets.postgresPassword"
    )
    
    for key in "${critical_keys[@]}"; do
        if ! grep -q "^[[:space:]]*${key//./\[[:space:]]*}:.*" "$values_file" 2>/dev/null; then
            log_warn "Key '$key' may be missing or commented out"
        fi
    done
    
    log_success "Values check complete"
}

main() {
    parse_args "$@"
    
    log_info "Starting Helm chart validation..."
    
    check_helm_installed
    validate_chart_structure
    validate_templates
    check_required_values
    
    log_success "Helm chart validation complete!"
}

main "$@"

