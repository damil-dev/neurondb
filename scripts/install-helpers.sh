#!/bin/bash
# ====================================================================
# Installation Helper Functions
# ====================================================================
# Shared functions for component installation scripts
# ====================================================================
# Note: This file is sourced by other scripts, so we don't use 'set -e'
# to allow calling scripts to control error handling

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Check if command exists
check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Check Go version (requires 1.23+)
check_go() {
    if ! check_command go; then
        print_error "Go compiler not found. Please install Go 1.23 or later."
        return 1
    fi
    
    go_version=$(go version 2>/dev/null | awk '{print $3}' | sed 's/go//')
    major=$(echo "$go_version" | cut -d. -f1)
    minor=$(echo "$go_version" | cut -d. -f2)
    
    if [ "$major" -lt 1 ] || ([ "$major" -eq 1 ] && [ "$minor" -lt 23 ]); then
        print_error "Go version too old (have $go_version, need 1.23+)"
        return 1
    fi
    
    print_success "Go $go_version found"
    return 0
}

# Check PostgreSQL
check_postgres() {
    if ! check_command psql; then
        print_error "PostgreSQL client (psql) not found. Please install PostgreSQL 16+."
        return 1
    fi
    
    if ! check_command pg_config; then
        print_warning "pg_config not found. PostgreSQL development headers may be needed."
    fi
    
    print_success "PostgreSQL client found"
    return 0
}

# Check Node.js (for NeuronDesktop)
check_nodejs() {
    if ! check_command node; then
        print_error "Node.js not found. Please install Node.js 18+."
        return 1
    fi
    
    node_version=$(node --version 2>/dev/null | sed 's/v//')
    major=$(echo "$node_version" | cut -d. -f1)
    
    if [ "$major" -lt 18 ]; then
        print_error "Node.js version too old (have $node_version, need 18+)"
        return 1
    fi
    
    print_success "Node.js $node_version found"
    return 0
}

# Detect platform
detect_platform() {
    case "$(uname -s)" in
        Linux*)     echo "linux";;
        Darwin*)    echo "macos";;
        *)          echo "unknown";;
    esac
}

# Detect init system
detect_init_system() {
    if [ -d /run/systemd/system ]; then
        echo "systemd"
    elif [ "$(detect_platform)" = "macos" ]; then
        echo "launchd"
    else
        echo "unknown"
    fi
}

