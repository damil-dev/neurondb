#!/bin/bash
#
# One-Command Installer for NeuronDB Ecosystem
# Detects OS, installs dependencies, and sets up NeuronDB
#
# Usage: ./scripts/install.sh [--skip-deps] [--skip-setup]
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SKIP_DEPS=false
SKIP_SETUP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--skip-deps] [--skip-setup]"
            echo "  --skip-deps   Skip dependency installation"
            echo "  --skip-setup  Skip database setup"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}NeuronDB Ecosystem Installer${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            if [ -f /etc/lsb-release ]; then
                . /etc/lsb-release
                OS="ubuntu"
                OS_VERSION="$DISTRIB_RELEASE"
            else
                OS="debian"
                OS_VERSION=$(cat /etc/debian_version | cut -d. -f1)
            fi
        elif [ -f /etc/redhat-release ]; then
            OS="rhel"
            OS_VERSION=$(cat /etc/redhat-release | grep -oE '[0-9]+' | head -1)
        else
            OS="linux"
            OS_VERSION="unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        OS_VERSION=$(sw_vers -productVersion | cut -d. -f1,2)
    else
        OS="unknown"
        OS_VERSION="unknown"
    fi
    
    echo -e "${GREEN}Detected OS: $OS $OS_VERSION${NC}"
}

# Install dependencies
install_dependencies() {
    if [ "$SKIP_DEPS" = true ]; then
        echo -e "${YELLOW}Skipping dependency installation${NC}"
        return
    fi
    
    echo -e "${BLUE}Installing dependencies...${NC}"
    
    case $OS in
        ubuntu|debian)
            echo "Installing packages for Ubuntu/Debian..."
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                make \
                gcc \
                postgresql-server-dev-all \
                git \
                curl
            ;;
        rhel)
            echo "Installing packages for RHEL/CentOS/Rocky..."
            sudo yum install -y \
                gcc \
                make \
                postgresql-devel \
                git \
                curl
            ;;
        macos)
            echo "Installing packages for macOS..."
            if ! command -v brew &> /dev/null; then
                echo -e "${RED}Homebrew not found. Please install Homebrew first.${NC}"
                echo "Visit: https://brew.sh"
                exit 1
            fi
            brew install postgresql@17
            ;;
        *)
            echo -e "${RED}Unsupported OS: $OS${NC}"
            echo "Please install dependencies manually:"
            echo "  - C compiler (GCC or Clang)"
            echo "  - Make"
            echo "  - PostgreSQL development headers"
            exit 1
            ;;
    esac
    
    echo -e "${GREEN}Dependencies installed${NC}"
}

# Detect PostgreSQL version
detect_postgresql() {
    echo -e "${BLUE}Detecting PostgreSQL installation...${NC}"
    
    if command -v psql &> /dev/null; then
        PG_VERSION=$(psql --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
        PG_MAJOR=$(echo $PG_VERSION | cut -d. -f1)
        echo -e "${GREEN}PostgreSQL $PG_VERSION detected${NC}"
        
        if [ "$PG_MAJOR" -lt 16 ] || [ "$PG_MAJOR" -gt 18 ]; then
            echo -e "${RED}PostgreSQL $PG_VERSION is not supported. NeuronDB requires PostgreSQL 16, 17, or 18.${NC}"
            exit 1
        fi
    else
        echo -e "${RED}PostgreSQL not found. Please install PostgreSQL 16, 17, or 18.${NC}"
        exit 1
    fi
    
    # Find pg_config
    if command -v pg_config &> /dev/null; then
        PG_CONFIG=$(which pg_config)
        echo -e "${GREEN}Found pg_config: $PG_CONFIG${NC}"
    else
        echo -e "${RED}pg_config not found. Please install PostgreSQL development headers.${NC}"
        exit 1
    fi
}

# Build and install NeuronDB extension
install_neurondb() {
    echo -e "${BLUE}Building NeuronDB extension...${NC}"
    
    cd "$(dirname "$0")/../NeuronDB"
    
    # Build
    make clean || true
    PG_CONFIG="$PG_CONFIG" make
    
    # Install
    echo -e "${BLUE}Installing NeuronDB extension...${NC}"
    sudo PG_CONFIG="$PG_CONFIG" make install
    
    echo -e "${GREEN}NeuronDB extension installed${NC}"
    cd - > /dev/null
}

# Setup database
setup_database() {
    if [ "$SKIP_SETUP" = true ]; then
        echo -e "${YELLOW}Skipping database setup${NC}"
        return
    fi
    
    echo -e "${BLUE}Setting up database...${NC}"
    
    # Check if setup script exists
    if [ -f "$(dirname "$0")/setup_neurondb_ecosystem.sh" ]; then
        echo "Running unified setup script..."
        "$(dirname "$0")/setup_neurondb_ecosystem.sh"
    else
        echo -e "${YELLOW}Setup script not found. Please run database setup manually.${NC}"
        echo "See README.md for manual setup instructions."
    fi
}

# Verify installation
verify_installation() {
    echo -e "${BLUE}Verifying installation...${NC}"
    
    if [ -f "$(dirname "$0")/verify_neurondb_integration.sh" ]; then
        echo "Running Tier 0 verification tests..."
        "$(dirname "$0")/verify_neurondb_integration.sh" --tier 0 || {
            echo -e "${YELLOW}Verification tests had issues. Check output above.${NC}"
        }
    else
        echo -e "${YELLOW}Verification script not found. Skipping verification.${NC}"
    fi
}

# Main installation flow
main() {
    detect_os
    install_dependencies
    detect_postgresql
    install_neurondb
    setup_database
    verify_installation
    
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Installation complete!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Verify installation: ./scripts/verify_neurondb_integration.sh"
    echo "  2. Read the documentation: README.md"
    echo "  3. Check compatibility: COMPATIBILITY.md"
    echo ""
}

main "$@"

