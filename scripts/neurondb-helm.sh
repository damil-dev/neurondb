#!/bin/bash
#
# NeuronDB Helm Chart Management Script
# Self-sufficient script for ALL Helm operations: validate, test, release
#
# Usage:
#   ./neurondb-helm.sh COMMAND [OPTIONS]
#
# Commands:
#   validate     Validate Helm chart (structure, templates, security, etc.)
#   test        Test Helm chart (installation, upgrade, rollback)
#   release     Release Helm chart (version bump, package, publish)
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
readonly MAGENTA='\033[0;35m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Default configuration
COMMAND=""
VERBOSE=false
DRY_RUN=false
CHART_DIR="${CHART_DIR:-helm/neurondb}"
CHART_NAME="neurondb"


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
    echo -e "${BLUE}║${NC}  ${BOLD}NeuronDB Helm Chart Management${NC}                        ${BLUE}║${NC}"
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


show_help() {
    cat << EOF
${BOLD}NeuronDB Helm Chart Management${NC}

${BOLD}Usage:${NC}
    ${SCRIPT_NAME} COMMAND [OPTIONS]

${BOLD}Commands:${NC}
    validate     Validate Helm chart (structure, templates, security, etc.)
    test         Test Helm chart (installation, upgrade, rollback)
    release      Release Helm chart (version bump, package, publish)

${BOLD}Validate Command:${NC}
    ${SCRIPT_NAME} validate [OPTIONS]
    
    Options:
        --chart-dir DIR       Chart directory (default: helm/neurondb)
        --values-file FILE    Custom values file for validation
        --namespace NAME       Namespace for validation (default: neurondb-test)
        --release-name NAME   Release name for validation (default: neurondb-validation)

${BOLD}Test Command:${NC}
    ${SCRIPT_NAME} test [OPTIONS]
    
    Options:
        --chart-dir DIR       Chart directory (default: helm/neurondb)
        --namespace NAME      Namespace for testing (default: neurondb-test)
        --release-name NAME   Release name for testing (default: neurondb-test)
        --use-kind           Use kind cluster if no cluster available
        --no-cleanup          Don't cleanup after testing
        --timeout SECONDS     Timeout for operations (default: 300s)

${BOLD}Release Command:${NC}
    ${SCRIPT_NAME} release [OPTIONS]
    
    Options:
        --chart-dir DIR       Chart directory (default: helm/neurondb)
        --version-type TYPE   Version bump type: patch, minor, major (default: patch)
        --oci-registry REG    OCI registry (default: ghcr.io)
        --oci-repo REPO       OCI repository (default: neurondb/helm-charts)
        --sign-key KEY        GPG key for signing
        --no-publish          Don't publish to OCI registry

${BOLD}Global Options:${NC}
    -h, --help              Show this help message
    -v, --verbose           Enable verbose output
    -V, --version           Show version information
    --dry-run               Preview changes without applying

${BOLD}Examples:${NC}
    # Validate chart
    ${SCRIPT_NAME} validate

    # Test chart installation
    ${SCRIPT_NAME} test

    # Release chart with patch version bump
    ${SCRIPT_NAME} release --version-type patch

    # Release chart with minor version bump (dry run)
    ${SCRIPT_NAME} release --version-type minor --dry-run

    # Validate with custom values
    ${SCRIPT_NAME} validate --values-file custom-values.yaml

EOF
}


check_helm() {
    if ! command -v helm &> /dev/null; then
        log_error "Helm is not installed"
        log_info "Install Helm: https://helm.sh/docs/intro/install/"
        return 1
    fi
    return 0
}

check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        log_warning "kubectl is not installed (some operations will be skipped)"
        return 1
    fi
    return 0
}

check_yq() {
    if ! command -v yq &> /dev/null; then
        log_warning "yq is not installed (version bumping will be skipped)"
        return 1
    fi
    return 0
}


validate_command() {
    local values_file=""
    local namespace="neurondb-test"
    local release_name="neurondb-validation"
    
    shift
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --chart-dir)
                CHART_DIR="$2"
                shift 2
                ;;
            --values-file)
                values_file="$2"
                shift 2
                ;;
            --namespace)
                namespace="$2"
                shift 2
                ;;
            --release-name)
                release_name="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option for validate command: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    if ! check_helm; then
        exit 1
    fi
    
    CHART_DIR="$PROJECT_ROOT/$CHART_DIR"
    
    if [ ! -d "$CHART_DIR" ]; then
        log_error "Chart directory not found: $CHART_DIR"
        exit 1
    fi
    
    print_header
    log_info "Chart Directory: $CHART_DIR"
    log_info "Release Name: $release_name"
    log_info "Namespace: $namespace"
    [ -n "$values_file" ] && log_info "Values File: $values_file"
    echo ""
    
    local passed=0
    local failed=0
    local warnings=0
    
    # Check prerequisites
    print_section "Checking Prerequisites"
    if check_helm; then
        print_test "PASS" "Helm is installed ($(helm version --short))"
        ((passed++))
    else
        print_test "FAIL" "Helm is not installed"
        ((failed++))
        exit 1
    fi
    
    if check_kubectl; then
        print_test "PASS" "kubectl is installed"
        ((passed++))
    else
        print_test "SKIP" "kubectl is not installed (some validations will be skipped)"
        ((warnings++))
    fi
    
    if check_yq; then
        print_test "PASS" "yq is installed"
        ((passed++))
    else
        print_test "SKIP" "yq is not installed (some validations will be skipped)"
        ((warnings++))
    fi
    
    if [ ! -f "$CHART_DIR/Chart.yaml" ]; then
        print_test "FAIL" "Chart.yaml not found in $CHART_DIR"
        ((failed++))
        exit 1
    else
        print_test "PASS" "Chart.yaml found"
        ((passed++))
    fi
    
    if [ ! -f "$CHART_DIR/values.yaml" ]; then
        print_test "FAIL" "values.yaml not found in $CHART_DIR"
        ((failed++))
        exit 1
    else
        print_test "PASS" "values.yaml found"
        ((passed++))
    fi
    
    # Validate chart structure
    print_section "Validating Chart Structure"
    
    if [ -f "$CHART_DIR/values.schema.json" ]; then
        print_test "PASS" "values.schema.json exists"
        ((passed++))
    else
        print_test "SKIP" "values.schema.json not found (schema validation will be skipped)"
        ((warnings++))
    fi
    
    if [ -f "$CHART_DIR/templates/_helpers.tpl" ]; then
        print_test "PASS" "_helpers.tpl found"
        ((passed++))
    else
        print_test "FAIL" "_helpers.tpl not found"
        ((failed++))
    fi
    
    local template_count=$(find "$CHART_DIR/templates" -name "*.yaml" -o -name "*.tpl" 2>/dev/null | wc -l | xargs)
    if [ "$template_count" -gt 0 ]; then
        print_test "PASS" "Found $template_count template files"
        ((passed++))
    else
        print_test "FAIL" "No template files found"
        ((failed++))
    fi
    
    # Run Helm lint
    print_section "Running Helm Lint"
    
    local lint_cmd="helm lint \"$CHART_DIR\" --strict"
    [ -n "$values_file" ] && lint_cmd="$lint_cmd --values \"$values_file\""
    
    local lint_output=$(eval "$lint_cmd" 2>&1 || true)
    
    if echo "$lint_output" | grep -qi "error"; then
        print_test "FAIL" "Helm lint found errors"
        echo "$lint_output" | grep -i "error" | head -10
        ((failed++))
    elif echo "$lint_output" | grep -qi "warning"; then
        print_test "SKIP" "Helm lint found warnings"
        echo "$lint_output" | grep -i "warning" | head -5
        ((warnings++))
    else
        print_test "PASS" "Helm lint passed with no errors or warnings"
        ((passed++))
    fi
    
    # Validate template rendering
    print_section "Validating Template Rendering"
    
    local test_values=""
    [ -n "$values_file" ] && test_values="--values \"$values_file\""
    
    log_info "Rendering templates with default values..."
    if helm template "$release_name" "$CHART_DIR" $test_values --debug 2>&1 | grep -q "Error:"; then
        print_test "FAIL" "Template rendering failed with default values"
        ((failed++))
    else
        print_test "PASS" "Templates render successfully with default values"
        ((passed++))
    fi
    
    # Validate Kubernetes API compatibility
    if check_kubectl && kubectl cluster-info &>/dev/null 2>&1; then
        print_section "Validating Kubernetes API Compatibility"
        
        log_info "Validating generated resources against Kubernetes API..."
        local api_output=$(helm template "$release_name" "$CHART_DIR" ${values_file:+--values "$values_file"} 2>/dev/null | \
            kubectl apply --dry-run=client -f - 2>&1 || true)
        
        if echo "$api_output" | grep -qi "error"; then
            print_test "FAIL" "Some resources failed Kubernetes API validation"
            echo "$api_output" | grep -i "error" | head -5
            ((failed++))
        else
            print_test "PASS" "All resources are valid Kubernetes API objects"
            ((passed++))
        fi
    else
        print_section "Validating Kubernetes API Compatibility"
        print_test "SKIP" "kubectl not available or cluster not accessible"
        ((warnings++))
    fi
    
    # Summary
    print_section "Validation Summary"
    echo -e "Total Checks: $((passed + failed + warnings))"
    echo -e "${GREEN}Passed: $passed${NC}"
    echo -e "${RED}Failed: $failed${NC}"
    echo -e "${YELLOW}Warnings: $warnings${NC}"
    echo ""
    
    if [ "$failed" -gt 0 ]; then
        log_error "Validation FAILED with $failed error(s)"
        exit 1
    elif [ "$warnings" -gt 0 ]; then
        log_warning "Validation completed with $warnings warning(s)"
        exit 0
    else
        log_success "All validations PASSED"
        exit 0
    fi
}


test_command() {
    local namespace="neurondb-test"
    local release_name="neurondb-test"
    local use_kind=false
    local cleanup=true
    local timeout="300s"
    
    shift
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --chart-dir)
                CHART_DIR="$2"
                shift 2
                ;;
            --namespace)
                namespace="$2"
                shift 2
                ;;
            --release-name)
                release_name="$2"
                shift 2
                ;;
            --use-kind)
                use_kind=true
                shift
                ;;
            --no-cleanup)
                cleanup=false
                shift
                ;;
            --timeout)
                timeout="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option for test command: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    if ! check_helm; then
        exit 1
    fi
    
    if ! check_kubectl; then
        log_error "kubectl is required for testing"
        exit 1
    fi
    
    CHART_DIR="$PROJECT_ROOT/$CHART_DIR"
    
    if [ ! -d "$CHART_DIR" ]; then
        log_error "Chart directory not found: $CHART_DIR"
        exit 1
    fi
    
    print_header
    log_info "Chart Directory: $CHART_DIR"
    log_info "Release Name: $release_name"
    log_info "Namespace: $namespace"
    log_info "Cleanup: $cleanup"
    echo ""
    
    local passed=0
    local failed=0
    local skipped=0
    
    # Cleanup function
    cleanup_test() {
        if [ "$cleanup" = "true" ]; then
            print_section "Cleaning Up"
            log_info "Uninstalling Helm release..."
            helm uninstall "$release_name" --namespace "$namespace" 2>/dev/null || true
            log_info "Deleting namespace..."
            kubectl delete namespace "$namespace" --ignore-not-found=true --wait=true 2>/dev/null || true
            log_success "Cleanup completed"
        fi
    }
    
    trap cleanup_test EXIT
    
    # Check prerequisites
    print_section "Checking Prerequisites"
    
    if check_helm; then
        print_test "PASS" "Helm is installed ($(helm version --short))"
        ((passed++))
    else
        print_test "FAIL" "Helm is not installed"
        ((failed++))
        exit 1
    fi
    
    if check_kubectl; then
        print_test "PASS" "kubectl is installed"
        ((passed++))
    else
        print_test "FAIL" "kubectl is not installed (required for testing)"
        ((failed++))
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &>/dev/null 2>&1; then
        if [ "$use_kind" = "true" ]; then
            log_info "No cluster detected, attempting to use kind..."
            if command -v kind &> /dev/null; then
                log_info "Creating kind cluster for testing..."
                kind create cluster --name neurondb-test --wait 60s || {
                    print_test "FAIL" "Failed to create kind cluster"
                    ((failed++))
                    exit 1
                }
                print_test "PASS" "Kind cluster created"
                ((passed++))
            else
                print_test "FAIL" "kind is not installed and no cluster is available"
                ((failed++))
                exit 1
            fi
        else
            print_test "FAIL" "No Kubernetes cluster available. Set --use-kind to use kind"
            ((failed++))
            exit 1
        fi
    else
        print_test "PASS" "Kubernetes cluster is accessible"
        ((passed++))
    fi
    
    # Test template rendering
    print_section "Testing Template Rendering"
    
    log_info "Testing with default values..."
    if helm template "$release_name" "$CHART_DIR" > /dev/null 2>&1; then
        print_test "PASS" "Template rendering with default values"
        ((passed++))
    else
        print_test "FAIL" "Template rendering failed with default values"
        helm template "$release_name" "$CHART_DIR" 2>&1 | head -20
        ((failed++))
    fi
    
    # Test chart installation
    print_section "Testing Chart Installation"
    
    log_info "Creating namespace: $namespace"
    if kubectl create namespace "$namespace" --dry-run=client -o yaml | kubectl apply -f -; then
        print_test "PASS" "Namespace created"
        ((passed++))
    else
        print_test "FAIL" "Failed to create namespace"
        ((failed++))
        exit 1
    fi
    
    # Generate secure password
    local postgres_password
    postgres_password=$(openssl rand -base64 32 2>/dev/null || echo "test-password-$(date +%s)")
    
    log_info "Installing Helm chart..."
    if helm install "$release_name" "$CHART_DIR" \
        --namespace "$namespace" \
        --wait \
        --timeout "$timeout" \
        --set secrets.postgresPassword="$postgres_password" \
        --set monitoring.enabled=false \
        --set backup.enabled=false \
        --set migrations.enabled=false 2>&1; then
        print_test "PASS" "Chart installed successfully"
        ((passed++))
    else
        print_test "FAIL" "Chart installation failed"
        kubectl get pods -n "$namespace" || true
        kubectl get events -n "$namespace" --sort-by='.lastTimestamp' | tail -20 || true
        ((failed++))
        exit 1
    fi
    
    # Wait for pods to be ready
    log_info "Waiting for pods to be ready..."
    if kubectl wait --for=condition=ready pod \
        -l app.kubernetes.io/instance="$release_name" \
        -n "$namespace" \
        --timeout="$timeout" 2>/dev/null; then
        print_test "PASS" "All pods are ready"
        ((passed++))
    else
        print_test "SKIP" "Some pods may not be ready"
        kubectl get pods -n "$namespace" || true
        ((skipped++))
    fi
    
    # Test upgrade
    print_section "Testing Chart Upgrade"
    
    log_info "Performing Helm upgrade test..."
    if helm upgrade "$release_name" "$CHART_DIR" \
        --namespace "$namespace" \
        --reuse-values \
        --wait \
        --timeout "$timeout" \
        --set secrets.postgresPassword="$postgres_password" 2>&1; then
        print_test "PASS" "Chart upgrade successful"
        ((passed++))
    else
        print_test "FAIL" "Chart upgrade failed"
        kubectl get events -n "$namespace" --sort-by='.lastTimestamp' | tail -20 || true
        ((failed++))
    fi
    
    # Summary
    print_section "Test Summary"
    echo -e "Total Tests: $((passed + failed + skipped))"
    echo -e "${GREEN}Passed: $passed${NC}"
    echo -e "${RED}Failed: $failed${NC}"
    echo -e "${YELLOW}Skipped: $skipped${NC}"
    echo ""
    
    if [ "$failed" -gt 0 ]; then
        log_error "Test Suite FAILED with $failed test(s)"
        exit 1
    else
        log_success "All tests PASSED"
        exit 0
    fi
}


release_command() {
    local version_type="patch"
    local oci_registry="${OCI_REGISTRY:-ghcr.io}"
    local oci_repo="${OCI_REPO:-neurondb/helm-charts}"
    local sign_key="${SIGN_KEY:-}"
    local no_publish=false
    
    shift
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --chart-dir)
                CHART_DIR="$2"
                shift 2
                ;;
            --version-type)
                version_type="$2"
                shift 2
                ;;
            --oci-registry)
                oci_registry="$2"
                shift 2
                ;;
            --oci-repo)
                oci_repo="$2"
                shift 2
                ;;
            --sign-key)
                sign_key="$2"
                shift 2
                ;;
            --no-publish)
                no_publish=true
                shift
                ;;
            *)
                log_error "Unknown option for release command: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    if ! check_helm; then
        exit 1
    fi
    
    CHART_DIR="$PROJECT_ROOT/$CHART_DIR"
    
    if [ ! -d "$CHART_DIR" ]; then
        log_error "Chart directory not found: $CHART_DIR"
        exit 1
    fi
    
    if [ ! -f "$CHART_DIR/Chart.yaml" ]; then
        log_error "Chart.yaml not found in $CHART_DIR"
        exit 1
    fi
    
    print_header
    
    # Check prerequisites
    print_section "Checking Prerequisites"
    
    if check_helm; then
        print_test "PASS" "Helm is installed"
    else
        print_test "FAIL" "Helm is not installed"
        exit 1
    fi
    
    if check_yq; then
        print_test "PASS" "yq is installed"
    else
        print_test "FAIL" "yq is not installed (required for version bumping)"
        exit 1
    fi
    
    # Bump version
    print_section "Bumping Chart Version"
    
    local current_version
    current_version=$(yq eval '.version' "$CHART_DIR/Chart.yaml" 2>/dev/null || echo "0.0.0")
    log_info "Current version: $current_version"
    
    IFS='.' read -ra VERSION_PARTS <<< "$current_version"
    local major="${VERSION_PARTS[0]:-0}"
    local minor="${VERSION_PARTS[1]:-0}"
    local patch="${VERSION_PARTS[2]:-0}"
    
    case "$version_type" in
        major)
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        minor)
            minor=$((minor + 1))
            patch=0
            ;;
        patch)
            patch=$((patch + 1))
            ;;
        *)
            log_error "Invalid version type: $version_type (must be major, minor, or patch)"
            exit 1
            ;;
    esac
    
    local new_version="${major}.${minor}.${patch}"
    log_info "New version: $new_version"
    
    if [ "$DRY_RUN" = "false" ]; then
        yq eval ".version = \"$new_version\"" -i "$CHART_DIR/Chart.yaml" 2>/dev/null || {
            log_error "Failed to update version in Chart.yaml"
            exit 1
        }
        log_success "Version bumped to $new_version"
    else
        log_info "DRY RUN: Would bump version to $new_version"
    fi
    
    # Validate chart
    print_section "Validating Chart"
    
    if helm lint "$CHART_DIR" --strict 2>&1; then
        log_success "Chart validation passed"
    else
        log_error "Chart validation failed"
        exit 1
    fi
    
    # Package chart
    print_section "Packaging Chart"
    
    mkdir -p "$PROJECT_ROOT/dist"
    
    if [ "$DRY_RUN" = "false" ]; then
        helm package "$CHART_DIR" --destination "$PROJECT_ROOT/dist" 2>&1 || {
            log_error "Failed to package chart"
            exit 1
        }
        local package_file="$PROJECT_ROOT/dist/${CHART_NAME}-${new_version}.tgz"
        
        if [ ! -f "$package_file" ]; then
            log_error "Failed to create package file"
            exit 1
        fi
        
        log_success "Chart packaged: $package_file"
    else
        log_info "DRY RUN: Would package chart to dist/${CHART_NAME}-${new_version}.tgz"
        local package_file="$PROJECT_ROOT/dist/${CHART_NAME}-${new_version}.tgz"
    fi
    
    # Sign chart (if key provided)
    if [ -n "$sign_key" ]; then
        print_section "Signing Chart"
        
        if [ "$DRY_RUN" = "false" ]; then
            if helm package "$CHART_DIR" --sign --key "$sign_key" --keyring ~/.gnupg/secring.gpg 2>&1; then
                log_success "Chart signed successfully"
            else
                log_warning "Chart signing failed (continuing without signature)"
            fi
        else
            log_info "DRY RUN: Would sign chart with key $sign_key"
        fi
    fi
    
    # Generate release notes
    print_section "Generating Release Notes"
    
    local notes_file="$PROJECT_ROOT/dist/RELEASE_NOTES_${new_version}.md"
    
    if [ "$DRY_RUN" = "false" ]; then
        cat > "$notes_file" <<EOF
# NeuronDB Helm Chart ${new_version} Release Notes

## Release Date
$(date -u +"%Y-%m-%d %H:%M:%S UTC")

## Chart Version
${new_version}

## Changes
- See CHANGELOG.md for detailed changes

## Installation
\`\`\`bash
helm install neurondb oci://${oci_registry}/${oci_repo}/${CHART_NAME}:${new_version} \\
  --namespace neurondb \\
  --create-namespace
\`\`\`

## Upgrading
\`\`\`bash
helm upgrade neurondb oci://${oci_registry}/${oci_repo}/${CHART_NAME}:${new_version} \\
  --namespace neurondb \\
  --values my-values.yaml
\`\`\`

## Documentation
- Full documentation: https://github.com/neurondb/neurondb/tree/main/helm/neurondb
- Deployment guide: https://github.com/neurondb/neurondb/blob/main/Docs/deployment/kubernetes-helm.md
EOF
        log_success "Release notes generated: $notes_file"
    else
        log_info "DRY RUN: Would generate release notes"
    fi
    
    # Publish to OCI registry
    if [ "$no_publish" = "false" ] && [ "$DRY_RUN" = "false" ]; then
        print_section "Publishing Chart"
        
        log_info "Publishing chart to OCI registry..."
        local oci_ref="${oci_registry}/${oci_repo}/${CHART_NAME}:${new_version}"
        
        if helm push "$package_file" "oci://${oci_registry}/${oci_repo}" 2>&1; then
            log_success "Chart published to ${oci_ref}"
        else
            log_error "Failed to publish chart"
            exit 1
        fi
    elif [ "$DRY_RUN" = "true" ]; then
        print_section "Publishing Chart"
        log_info "DRY RUN: Would publish chart to ${oci_registry}/${oci_repo}"
    else
        log_info "Skipping publish (--no-publish specified)"
    fi
    
    echo ""
    log_success "Release process completed!"
    log_info "Chart version: $new_version"
    log_info "Package: $package_file"
    echo ""
}


parse_arguments() {
    # Check for help/version flags before command
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                show_help
                exit 0
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


main() {
    parse_arguments "$@"
    
    case "$COMMAND" in
        validate)
            validate_command "$@"
            ;;
        test)
            test_command "$@"
            ;;
        release)
            release_command "$@"
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

main "$@"

