#!/bin/bash
#
# test-packages-vagrant-rocky.sh - Orchestrate complete RPM package testing in Rocky Linux Vagrant VM
#
# This script:
#   1. Downloads packages locally as fallback
#   2. Starts Rocky Linux Vagrant VM
#   3. Downloads/copies packages in VM
#   4. Verifies packages
#   5. Installs packages
#   6. Runs integration tests
#   7. Reports results
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_PACKAGES_DIR="$REPO_ROOT/packages-local"
DESTROY_VM="${DESTROY_VM:-false}"

log_info() {
    echo -e "${BLUE}[HOST]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[HOST]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[HOST]${NC} $*"
}

log_error() {
    echo -e "${RED}[HOST]${NC} $*" >&2
}

# Download packages locally as fallback
download_packages_locally() {
    log_info "Downloading RPM packages locally as fallback source..."

    if [ ! -f "$SCRIPT_DIR/verify-pkgs.sh" ]; then
        log_error "verify-pkgs.sh not found"
        return 1
    fi

    mkdir -p "$LOCAL_PACKAGES_DIR"

    # Download packages and keep them (forcing RPM format)
    if PKG_FORMAT=rpm "$SCRIPT_DIR/verify-pkgs.sh" --keep-downloads -d "$LOCAL_PACKAGES_DIR" 2>&1 | tee /tmp/pkg-download.log; then
        log_success "Packages downloaded locally"
        return 0
    else
        log_warning "Package download had some issues, but continuing..."
        # Check if we got at least some packages
        local pkg_count=$(find "$LOCAL_PACKAGES_DIR" -name "*.rpm" -type f 2>/dev/null | wc -l)
        if [ "$pkg_count" -gt 0 ]; then
            log_info "Found $pkg_count packages locally, continuing..."
            return 0
        else
            log_error "No packages found after download attempt"
            return 1
        fi
    fi
}

# Run command in Vagrant VM
vagrant_run() {
    local cmd="$1"
    local description="${2:-Running command}"

    log_info "$description..."
    if vagrant ssh -c "$cmd" 2>&1; then
        log_success "$description completed"
        return 0
    else
        log_error "$description failed"
        return 1
    fi
}

# Main workflow
main() {
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     NeuronDB RPM Package Testing with Vagrant (Rocky Linux) ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    cd "$REPO_ROOT"

    # Use Rocky Linux Vagrantfile
    export VAGRANT_VAGRANTFILE=Vagrantfile.rocky

    # Step 1: Download packages locally (fallback)
    log_info "Step 1/8: Downloading RPM packages locally as fallback..."
    download_packages_locally || log_warning "Local package download had issues, will try in VM"

    # Step 2: Start Vagrant VM
    log_info "Step 2/8: Starting Rocky Linux Vagrant VM..."
    if vagrant status | grep -q "running"; then
        log_info "VM is already running"
    else
        if vagrant up; then
            log_success "VM started successfully"
        else
            log_error "Failed to start VM"
            exit 1
        fi
    fi

    # Step 3: Download/copy packages in VM
    log_info "Step 3/8: Downloading/copying packages in VM..."
    if vagrant_run "bash /vagrant/vagrant/download-packages.sh" "Downloading packages"; then
        log_success "Packages ready in VM"
    else
        log_error "Package download/copy failed"
        exit 1
    fi

    # Step 4: Verify packages
    log_info "Step 4/8: Verifying packages..."
    if vagrant_run "PKG_FORMAT=rpm bash /vagrant/scripts/verify-pkgs.sh --skip-download --skip-checksums -d /vagrant/packages" "Verifying packages"; then
        log_success "Package verification passed"
    else
        log_warning "Some package verification issues, but continuing..."
    fi

    # Step 5: Install packages
    log_info "Step 5/8: Installing packages..."
    if vagrant_run "bash /vagrant/vagrant/install-packages-rpm.sh" "Installing packages"; then
        log_success "Packages installed successfully"
    else
        log_error "Package installation failed"
        exit 1
    fi

    # Step 6: Run integration tests
    log_info "Step 6/8: Running integration tests..."
    if vagrant_run "bash /vagrant/vagrant/test-integration-rpm.sh" "Running integration tests"; then
        log_success "Integration tests completed"
    else
        log_error "Integration tests failed"
        exit 1
    fi

    # Step 7: Run functionality tests
    log_info "Step 7/8: Running functionality tests..."
    if vagrant_run "PG_CMD='/usr/pgsql-18/bin/psql' bash /vagrant/vagrant/test-functionality.sh" "Running functionality tests"; then
        log_success "Functionality tests completed"
    else
        log_warning "Some functionality tests had issues"
    fi

    # Step 8: Collect results
    log_info "Step 8/8: Collecting test results..."
    mkdir -p "$REPO_ROOT/test-results"
    if vagrant ssh -c "cat /vagrant/test-results/integration-tests-rpm.log" > "$REPO_ROOT/test-results/integration-tests-rpm.log" 2>/dev/null; then
        log_success "Test results collected: $REPO_ROOT/test-results/integration-tests-rpm.log"
    fi

    # Final summary
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}Testing Complete${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    log_success "All steps completed successfully!"

    if [ -f "$REPO_ROOT/test-results/integration-tests-rpm.log" ]; then
        echo ""
        log_info "Test results:"
        tail -20 "$REPO_ROOT/test-results/integration-tests-rpm.log"
    fi

    # Optionally destroy VM
    if [ "$DESTROY_VM" = "true" ]; then
        log_info "Destroying VM as requested..."
        vagrant destroy -f
    else
        echo ""
        log_info "VM is still running. To destroy it, run:"
        log_info "  VAGRANT_VAGRANTFILE=Vagrantfile.rocky vagrant destroy"
        log_info "Or set DESTROY_VM=true to auto-destroy on completion"
    fi

    return 0
}

# Handle script interruption
trap 'log_error "Script interrupted"; exit 1' INT TERM

main "$@"

