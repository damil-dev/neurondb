#!/bin/bash
#
# install-vagrant-deps.sh - Install VirtualBox and Vagrant if missing
#
# This script checks for VirtualBox and Vagrant installation and installs
# them if they're not present on the system.
#
# Usage:
#   ./scripts/install-vagrant-deps.sh
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log_info() {
    echo -e "${BLUE}ℹ${NC} $*"
}

log_success() {
    echo -e "${GREEN}✓${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $*"
}

log_error() {
    echo -e "${RED}✗${NC} $*" >&2
}

# Detect OS
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS_ID="${ID:-}"
        OS_VERSION="${VERSION_ID:-}"
    else
        log_error "Cannot detect OS. /etc/os-release not found."
        exit 1
    fi

    log_info "Detected OS: $OS_ID $OS_VERSION"
}

# Check if running as root or with sudo
check_sudo() {
    if [ "$EUID" -eq 0 ]; then
        SUDO=""
    elif command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
        log_info "Will use sudo for installation commands"
    else
        log_error "This script requires root privileges or sudo access"
        exit 1
    fi
}

# Install VirtualBox
install_virtualbox() {
    if command -v vboxmanage >/dev/null 2>&1; then
        local version=$(vboxmanage --version)
        log_success "VirtualBox already installed: $version"
        return 0
    fi

    log_info "Installing VirtualBox..."

    case "$OS_ID" in
        ubuntu|debian)
            # Add Oracle repository
            log_info "Adding VirtualBox repository..."

            # Download and add GPG key
            wget -O- https://www.virtualbox.org/download/oracle_vbox_2016.asc | \
                $SUDO gpg --yes --output /usr/share/keyrings/oracle-virtualbox-2016.gpg --dearmor

            # Determine codename
            local codename
            case "$OS_VERSION" in
                22.04|"22.04"*)
                    codename="jammy"
                    ;;
                24.04|"24.04"*)
                    codename="noble"
                    ;;
                *)
                    codename=$(lsb_release -cs 2>/dev/null || echo "jammy")
                    log_warning "Unknown Ubuntu version, defaulting to jammy"
                    ;;
            esac

            # Add repository
            echo "deb [arch=amd64 signed-by=/usr/share/keyrings/oracle-virtualbox-2016.gpg] https://download.virtualbox.org/virtualbox/debian $codename contrib" | \
                $SUDO tee /etc/apt/sources.list.d/virtualbox.list >/dev/null

            # Update and install
            $SUDO apt-get update
            $SUDO apt-get install -y virtualbox-7.1

            log_success "VirtualBox installed successfully"
            ;;
        *)
            log_error "Automatic VirtualBox installation not supported for $OS_ID"
            log_info "Please install VirtualBox manually from: https://www.virtualbox.org/wiki/Linux_Downloads"
            return 1
            ;;
    esac
}

# Install Vagrant
install_vagrant() {
    if command -v vagrant >/dev/null 2>&1; then
        local version=$(vagrant version --machine-readable 2>/dev/null | grep "installed_version" | cut -d',' -f4 || echo "unknown")
        log_success "Vagrant already installed: $version"
        return 0
    fi

    log_info "Installing Vagrant..."

    case "$OS_ID" in
        ubuntu|debian)
            # Add HashiCorp repository
            log_info "Adding HashiCorp repository..."

            # Download and add GPG key
            curl -fsSL https://apt.releases.hashicorp.com/gpg | $SUDO apt-key add -

            # Add repository
            local codename=$(lsb_release -cs)
            $SUDO apt-add-repository "deb [arch=$(dpkg --print-architecture)] https://apt.releases.hashicorp.com $codename main"

            # Update and install
            $SUDO apt-get update
            $SUDO apt-get install -y vagrant

            log_success "Vagrant installed successfully"
            ;;
        *)
            log_error "Automatic Vagrant installation not supported for $OS_ID"
            log_info "Please install Vagrant manually from: https://www.vagrantup.com/downloads"
            return 1
            ;;
    esac
}

# Verify installations
verify_installations() {
    local all_ok=true

    if command -v vboxmanage >/dev/null 2>&1; then
        local vbox_version=$(vboxmanage --version)
        log_success "VirtualBox: $vbox_version"
    else
        log_error "VirtualBox not found"
        all_ok=false
    fi

    if command -v vagrant >/dev/null 2>&1; then
        local vagrant_version=$(vagrant version --machine-readable 2>/dev/null | grep "installed_version" | cut -d',' -f4 || echo "unknown")
        log_success "Vagrant: $vagrant_version"
    else
        log_error "Vagrant not found"
        all_ok=false
    fi

    if [ "$all_ok" = true ]; then
        log_success "All dependencies installed and verified!"
        return 0
    else
        log_error "Some dependencies are missing"
        return 1
    fi
}

# Main
main() {
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║     VirtualBox & Vagrant Installation Script                ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    detect_os
    check_sudo

    install_virtualbox || exit 1
    install_vagrant || exit 1

    echo ""
    verify_installations
}

main "$@"

