#!/bin/bash
#
# NeuronDB Ecosystem Setup Script
# Professional, detailed one-command setup for NeuronDB ecosystem components
#
# Usage:
#   ./neurondb-setup.sh --mode docker --all
#   ./neurondb-setup.sh --mode deb --components NeuronDB NeuronAgent
#   ./neurondb-setup.sh --mode rpm --components NeuronDB NeuronMCP NeuronDesktop
#
# See --help for complete documentation

set -euo pipefail

# ============================================================================
# Configuration and Constants
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSION="2.0.0"
LOG_FILE="${LOG_FILE:-/tmp/neurondb-setup-$(date +%Y%m%d-%H%M%S).log}"

# Component definitions
readonly COMPONENT_NEURONDB="NeuronDB"
readonly COMPONENT_NEURONAGENT="NeuronAgent"
readonly COMPONENT_NEURONMCP="NeuronMCP"
readonly COMPONENT_NEURONDESKTOP="NeuronDesktop"
readonly ALL_COMPONENTS=("$COMPONENT_NEURONDB" "$COMPONENT_NEURONAGENT" "$COMPONENT_NEURONMCP" "$COMPONENT_NEURONDESKTOP")

# Component dependencies (what each component needs)
declare -A COMPONENT_DEPS
COMPONENT_DEPS["$COMPONENT_NEURONDB"]=""
COMPONENT_DEPS["$COMPONENT_NEURONAGENT"]="$COMPONENT_NEURONDB"
COMPONENT_DEPS["$COMPONENT_NEURONMCP"]="$COMPONENT_NEURONDB"
COMPONENT_DEPS["$COMPONENT_NEURONDESKTOP"]="$COMPONENT_NEURONDB"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly MAGENTA='\033[0;35m'
readonly BOLD='\033[1m'
readonly DIM='\033[2m'
readonly NC='\033[0m' # No Color

# Default configuration
MODE=""
SELECTED_COMPONENTS=()
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="neurondb"
DB_USER="postgres"
DB_PASSWORD="neurondb"
SKIP_SETUP=false
SKIP_SERVICES=false
VERBOSE=false
DRY_RUN=false
UNINSTALL=false
REMOVE_DATA=false

# Component status tracking
declare -A COMPONENT_INSTALLED
declare -A COMPONENT_STARTED
declare -A COMPONENT_VERIFIED
declare -A COMPONENT_UNINSTALLED

# ============================================================================
# Logging Functions
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

log_info() {
    log "INFO: $*"
}

log_error() {
    log "ERROR: $*"
}

log_warning() {
    log "WARNING: $*"
}

log_success() {
    log "SUCCESS: $*"
}

# ============================================================================
# Output Functions
# ============================================================================

print_header() {
    echo ""
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  ${BOLD}${MAGENTA}NeuronDB Ecosystem Setup${NC}                                          ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  ${BOLD}Professional Setup for PostgreSQL AI Ecosystem${NC}                       ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  ${DIM}Version: $VERSION${NC}                                                ${CYAN}║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    if [ "$VERBOSE" = true ]; then
        echo -e "${DIM}Log file: $LOG_FILE${NC}"
        echo ""
    fi
}

print_section() {
    echo ""
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  ${BOLD}${1}${NC}"
    printf "${BLUE}║${NC}  %-${#1}s" ""
    echo -e "${BLUE}║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${BLUE}▶${NC} ${BOLD}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    if [ "$VERBOSE" = true ]; then
        echo -e "${DIM}$(date '+%Y-%m-%d %H:%M:%S')${NC}"
    fi
    log_info "Step: $1"
}

print_substep() {
    echo -e "${CYAN}  └─${NC} ${1}"
    log_info "  Substep: $1"
}

print_success() {
    echo -e "${GREEN}  ✓${NC} ${1}"
    log_success "$1"
}

print_error() {
    echo -e "${RED}  ✗${NC} ${1}" >&2
    log_error "$1"
}

print_warning() {
    echo -e "${YELLOW}  ⚠${NC} ${1}"
    log_warning "$1"
}

print_info() {
    echo -e "${CYAN}  ℹ${NC} ${1}"
    log_info "$1"
}

print_progress() {
    echo -e "${BLUE}  →${NC} ${1}"
    log_info "Progress: $1"
}

print_detail() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${DIM}    ${1}${NC}"
        log_info "Detail: $1"
    fi
}

print_table_header() {
    printf "${BOLD}%-25s %-15s %-15s %-15s${NC}\n" "Component" "Status" "Version" "Details"
    echo -e "${BLUE}────────────────────────────────────────────────────────────────────────${NC}"
}

print_table_row() {
    local component=$1
    local status=$2
    local version=$3
    local details=$4
    local color="${NC}"
    
    case "$status" in
        "Installed"|"Running"|"Ready")
            color="${GREEN}"
            ;;
        "Failed"|"Error")
            color="${RED}"
            ;;
        "Warning"|"Pending")
            color="${YELLOW}"
            ;;
        *)
            color="${CYAN}"
            ;;
    esac
    
    printf "${color}%-25s${NC} ${color}%-15s${NC} %-15s %-15s\n" "$component" "$status" "$version" "$details"
}

# ============================================================================
# Utility Functions
# ============================================================================

contains_element() {
    local element
    for element in "${@:2}"; do
        [[ "$element" == "$1" ]] && return 0
    done
    return 1
}

resolve_dependencies() {
    local components=("$@")
    local resolved=()
    local to_process=("${components[@]}")
    local processed=()
    
    while [ ${#to_process[@]} -gt 0 ]; do
        local current="${to_process[0]}"
        to_process=("${to_process[@]:1}")
        
        if contains_element "$current" "${processed[@]}"; then
            continue
        fi
        
        # Add dependencies
        local deps="${COMPONENT_DEPS[$current]:-}"
        if [ -n "$deps" ]; then
            for dep in $deps; do
                if ! contains_element "$dep" "${resolved[@]}" "${to_process[@]}" "${processed[@]}"; then
                    to_process+=("$dep")
                    print_detail "Added dependency: $dep (required by $current)"
                fi
            done
        fi
        
        resolved+=("$current")
        processed+=("$current")
    done
    
    echo "${resolved[@]}"
}

validate_component() {
    local component=$1
    if ! contains_element "$component" "${ALL_COMPONENTS[@]}"; then
        print_error "Invalid component: $component"
        print_info "Valid components: ${ALL_COMPONENTS[*]}"
        return 1
    fi
    return 0
}

# ============================================================================
# Prerequisites Checking
# ============================================================================

check_command() {
    local cmd=$1
    local required=${2:-true}
    
    if command -v "$cmd" &> /dev/null; then
        if [ "$VERBOSE" = true ]; then
            local version
            version=$($cmd --version 2>/dev/null | head -1 | cut -d' ' -f3- || echo "unknown")
            print_detail "$cmd found: $version"
        fi
        return 0
    else
        if [ "$required" = true ]; then
            print_error "$cmd is not installed"
            return 1
        else
            print_warning "$cmd is not installed (optional)"
            return 1
        fi
    fi
}

check_prerequisites_docker() {
    print_step "Checking Docker Prerequisites"
    
    local missing=0
    
    print_substep "Checking Docker installation..."
    if ! check_command "docker"; then
        print_info "Install Docker: https://docs.docker.com/get-docker/"
        missing=$((missing + 1))
    else
        local docker_version
        docker_version=$(docker --version 2>/dev/null | cut -d' ' -f3 | tr -d ',' || echo "unknown")
        print_success "Docker found: $docker_version"
    fi
    
    print_substep "Checking Docker Compose..."
    if docker compose version &> /dev/null 2>&1; then
        local compose_version
        compose_version=$(docker compose version 2>/dev/null | head -1 | cut -d' ' -f4 || echo "unknown")
        print_success "Docker Compose found: $compose_version"
    elif command -v docker-compose &> /dev/null && docker-compose version &> /dev/null 2>&1; then
        local compose_version
        compose_version=$(docker-compose version 2>/dev/null | cut -d' ' -f4 | tr -d ',' || echo "unknown")
        print_success "Docker Compose found: $compose_version"
    else
        print_error "Docker Compose not found"
        print_info "Install Docker Compose: https://docs.docker.com/compose/install/"
        missing=$((missing + 1))
    fi
    
    print_substep "Checking Docker daemon..."
    if docker info &> /dev/null 2>&1; then
        print_success "Docker daemon is running"
    else
        print_error "Docker daemon is not running"
        print_info "Start Docker: sudo systemctl start docker (Linux) or start Docker Desktop (macOS/Windows)"
        missing=$((missing + 1))
    fi
    
    if [ $missing -gt 0 ]; then
        print_error "Missing $missing prerequisite(s). Please install them and try again."
        return 1
    fi
    
    print_success "All Docker prerequisites satisfied"
    return 0
}

check_prerequisites_packages() {
    print_step "Checking Package Installation Prerequisites"
    
    local missing=0
    
    print_substep "Checking PostgreSQL installation..."
    if ! check_command "psql"; then
        print_error "PostgreSQL client (psql) not found"
        print_info "Install PostgreSQL 16, 17, or 18: https://www.postgresql.org/download/"
        print_info "  Ubuntu/Debian: sudo apt-get install postgresql-client-17"
        print_info "  RHEL/CentOS: sudo yum install postgresql17"
        print_info "  macOS: brew install postgresql@17"
        missing=$((missing + 1))
    else
        local pg_version pg_major
        pg_version=$(psql --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "unknown")
        pg_major=$(echo "$pg_version" | cut -d. -f1)
        if [ -n "$pg_major" ] && [ "$pg_major" -ge 16 ] && [ "$pg_major" -le 18 ]; then
            print_success "PostgreSQL $pg_version found (compatible version)"
        else
            print_error "PostgreSQL $pg_version found, but version 16-18 is required"
            print_info "Please install PostgreSQL 16, 17, or 18"
            missing=$((missing + 1))
        fi
    fi
    
    print_substep "Checking package manager..."
    if [ "$MODE" = "deb" ]; then
        if ! check_command "dpkg"; then
            print_error "dpkg not found (required for DEB packages)"
            missing=$((missing + 1))
        else
            print_success "dpkg found"
            check_command "apt-get" false || true
        fi
        if ! check_command "sudo"; then
            print_warning "sudo not found (root privileges may be required)"
        fi
    elif [ "$MODE" = "rpm" ]; then
        if ! check_command "rpm"; then
            print_error "rpm not found (required for RPM packages)"
            missing=$((missing + 1))
        else
            print_success "rpm found"
            check_command "yum" false || check_command "dnf" false || true
        fi
        if ! check_command "sudo"; then
            print_warning "sudo not found (root privileges may be required)"
        fi
    elif [ "$MODE" = "mac" ]; then
        if [[ "$OSTYPE" != "darwin"* ]]; then
            print_error "macOS packages can only be installed on macOS"
            missing=$((missing + 1))
        else
            print_success "Running on macOS"
            if ! check_command "pkgutil"; then
                print_error "pkgutil not found (required for macOS packages)"
                missing=$((missing + 1))
            fi
        fi
    fi
    
    if [ $missing -gt 0 ]; then
        print_error "Missing $missing prerequisite(s). Please install them and try again."
        return 1
    fi
    
    print_success "All package prerequisites satisfied"
    return 0
}

check_database_connection() {
    print_step "Checking Database Connection"
    
    print_substep "Testing connection to $DB_HOST:$DB_PORT..."
    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "SELECT 1;" &> /dev/null; then
        print_success "Database connection successful"
        print_detail "Connected as user: $DB_USER"
        print_detail "Host: $DB_HOST, Port: $DB_PORT"
        
        # Get PostgreSQL version
        local pg_version
        pg_version=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -t -c "SELECT version();" 2>/dev/null | head -1 | xargs || echo "unknown")
        print_detail "PostgreSQL version: $pg_version"
        
        return 0
    else
        print_error "Cannot connect to database at $DB_HOST:$DB_PORT"
        print_info "Please ensure:"
        print_info "  - PostgreSQL is running"
        print_info "  - Connection parameters are correct"
        print_info "  - User '$DB_USER' has necessary privileges"
        print_info "  - Network connectivity is available"
        return 1
    fi
}

check_prerequisites_component() {
    local component=$1
    print_substep "Checking prerequisites for $component..."
    
    case "$component" in
        "$COMPONENT_NEURONDB")
            # NeuronDB needs PostgreSQL development headers for compilation
            # but for packages, they're included, so we skip this check
            print_success "Prerequisites for $component satisfied"
            ;;
        "$COMPONENT_NEURONAGENT")
            # NeuronAgent needs Go for source builds, but packages include binaries
            print_success "Prerequisites for $component satisfied"
            ;;
        "$COMPONENT_NEURONMCP")
            # NeuronMCP needs Go for source builds, but packages include binaries
            print_success "Prerequisites for $component satisfied"
            ;;
        "$COMPONENT_NEURONDESKTOP")
            # NeuronDesktop needs Node.js for frontend, but packages may include built assets
            check_command "node" false || check_command "npm" false || true
            print_success "Prerequisites for $component satisfied"
            ;;
    esac
}

# ============================================================================
# Package Installation Functions
# ============================================================================

find_packages() {
    local component=$1
    local mode=$2
    local pkg_dir=""
    local pattern=""
    
    case "$mode" in
        deb)
            pkg_dir="$REPO_ROOT/packaging/deb"
            case "$component" in
                "$COMPONENT_NEURONDB")
                    pattern="$pkg_dir/neurondb/*.deb"
                    ;;
                "$COMPONENT_NEURONAGENT")
                    pattern="$pkg_dir/neuronagent/*.deb"
                    ;;
                "$COMPONENT_NEURONMCP")
                    pattern="$pkg_dir/neuronmcp/*.deb"
                    ;;
            esac
            ;;
        rpm)
            pkg_dir="$REPO_ROOT/packaging/rpm"
            case "$component" in
                "$COMPONENT_NEURONDB")
                    pattern="$pkg_dir/neurondb/*.rpm"
                    ;;
                "$COMPONENT_NEURONAGENT")
                    pattern="$pkg_dir/neuronagent/*.rpm"
                    ;;
                "$COMPONENT_NEURONMCP")
                    pattern="$pkg_dir/neuronmcp/*.rpm"
                    ;;
            esac
            ;;
        mac)
            pkg_dir="$REPO_ROOT/packaging/pkg"
            case "$component" in
                "$COMPONENT_NEURONDB")
                    pattern="$pkg_dir/neurondb/*.pkg"
                    ;;
                "$COMPONENT_NEURONAGENT")
                    pattern="$pkg_dir/neuronagent/*.pkg"
                    ;;
                "$COMPONENT_NEURONMCP")
                    pattern="$pkg_dir/neuronmcp/*.pkg"
                    ;;
            esac
            ;;
    esac
    
    # Find matching packages
    local packages=()
    for pkg in $pattern; do
        if [ -f "$pkg" ]; then
            packages+=("$pkg")
        fi
    done
    
    echo "${packages[@]}"
}

install_component_package() {
    local component=$1
    local mode=$2
    
    print_substep "Installing $component ($mode package)..."
    
    local packages
    read -ra packages <<< "$(find_packages "$component" "$mode")"
    
    if [ ${#packages[@]} -eq 0 ]; then
        print_error "No $mode packages found for $component"
        print_info "Build packages first: cd packaging/$mode && ./build-all-$mode.sh"
        return 1
    fi
    
    for pkg in "${packages[@]}"; do
        local pkg_name
        pkg_name=$(basename "$pkg")
        print_detail "Installing package: $pkg_name"
        
        if [ "$DRY_RUN" = true ]; then
            print_info "[DRY RUN] Would install: $pkg_name"
            continue
        fi
        
        case "$mode" in
            deb)
                if sudo dpkg -i "$pkg" 2>&1 | grep -q "error"; then
                    print_detail "Fixing dependencies..."
                    sudo apt-get install -f -y > /dev/null 2>&1 || true
                    if ! sudo dpkg -i "$pkg" > /dev/null 2>&1; then
                        print_error "Failed to install $pkg_name"
                        return 1
                    fi
                fi
                ;;
            rpm)
                if ! sudo rpm -ivh "$pkg" > /dev/null 2>&1; then
                    print_error "Failed to install $pkg_name"
                    return 1
                fi
                ;;
            mac)
                if ! sudo installer -pkg "$pkg" -target / > /dev/null 2>&1; then
                    print_error "Failed to install $pkg_name"
                    return 1
                fi
                ;;
        esac
        print_success "Installed: $pkg_name"
    done
    
    COMPONENT_INSTALLED[$component]=true
    return 0
}

# ============================================================================
# Database Setup Functions
# ============================================================================

setup_neurondb_schema() {
    print_step "Setting Up NeuronDB Database Schema"
    
    print_substep "Creating database if it doesn't exist..."
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would create database: $DB_NAME"
    else
        if ! PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME';" 2>/dev/null | grep -q 1; then
            if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "CREATE DATABASE $DB_NAME;" > /dev/null 2>&1; then
                print_success "Created database: $DB_NAME"
            else
                print_error "Failed to create database: $DB_NAME"
                return 1
            fi
        else
            print_success "Database already exists: $DB_NAME"
        fi
    fi
    
    print_substep "Installing NeuronDB extension..."
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would install NeuronDB extension"
    else
        if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS neurondb;" > /dev/null 2>&1; then
            local version
            version=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT neurondb.version();" 2>/dev/null | xargs || echo "unknown")
            print_success "NeuronDB extension installed (version: $version)"
        else
            print_error "Failed to install NeuronDB extension"
            print_info "Ensure NeuronDB package is installed and PostgreSQL user has CREATE EXTENSION privilege"
            return 1
        fi
    fi
    
    return 0
}

setup_neuronomcp_schema() {
    print_step "Setting Up NeuronMCP Database Schema"
    
    local setup_script="$REPO_ROOT/NeuronMCP/scripts/setup_neurondb_mcp.sh"
    
    if [ ! -f "$setup_script" ]; then
        print_error "NeuronMCP setup script not found: $setup_script"
        return 1
    fi
    
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would run NeuronMCP setup script"
        return 0
    fi
    
    print_substep "Running NeuronMCP setup script..."
    export NEURONDB_HOST="$DB_HOST"
    export NEURONDB_PORT="$DB_PORT"
    export NEURONDB_DATABASE="$DB_NAME"
    export NEURONDB_USER="$DB_USER"
    export NEURONDB_PASSWORD="$DB_PASSWORD"
    
    if bash "$setup_script" > /dev/null 2>&1; then
        print_success "NeuronMCP schema configured"
    else
        print_warning "NeuronMCP setup script had issues (may already be configured)"
        # Don't fail - schema might already exist
    fi
    
    return 0
}

setup_neuronagent_schema() {
    print_step "Setting Up NeuronAgent Database Schema"
    
    local setup_script="$REPO_ROOT/NeuronAgent/scripts/setup_neurondb_agent.sh"
    
    if [ ! -f "$setup_script" ]; then
        print_error "NeuronAgent setup script not found: $setup_script"
        return 1
    fi
    
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would run NeuronAgent setup script"
        return 0
    fi
    
    print_substep "Running NeuronAgent setup script..."
    export DB_HOST DB_PORT DB_NAME DB_USER DB_PASSWORD
    
    if bash "$setup_script" > /dev/null 2>&1; then
        print_success "NeuronAgent schema configured"
    else
        print_warning "NeuronAgent setup script had issues (may already be configured)"
        # Don't fail - schema might already exist
    fi
    
    return 0
}

setup_neurondesktop_schema() {
    print_step "Setting Up NeuronDesktop Database Schema"
    
    local setup_script="$REPO_ROOT/NeuronDesktop/scripts/setup_neurondesktop.sh"
    local desktop_db_name="${NEURONDESKTOP_DB_NAME:-neurondesk}"
    
    if [ ! -f "$setup_script" ]; then
        print_error "NeuronDesktop setup script not found: $setup_script"
        return 1
    fi
    
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would run NeuronDesktop setup script"
        return 0
    fi
    
    print_substep "Running NeuronDesktop setup script..."
    export DB_HOST DB_PORT
    export DB_NAME="$desktop_db_name"
    export DB_USER="${NEURONDESKTOP_DB_USER:-$DB_USER}"
    export DB_PASSWORD="${NEURONDESKTOP_DB_PASSWORD:-$DB_PASSWORD}"
    
    if bash "$setup_script" > /dev/null 2>&1; then
        print_success "NeuronDesktop schema configured"
    else
        print_warning "NeuronDesktop setup script had issues (may already be configured)"
        # Don't fail - schema might already exist
    fi
    
    return 0
}

# ============================================================================
# Uninstall Functions
# ============================================================================

get_package_name() {
    local component=$1
    case "$component" in
        "$COMPONENT_NEURONDB")
            echo "neurondb"
            ;;
        "$COMPONENT_NEURONAGENT")
            echo "neuronagent"
            ;;
        "$COMPONENT_NEURONMCP")
            echo "neuronmcp"
            ;;
        *)
            echo ""
            ;;
    esac
}

uninstall_component_package() {
    local component=$1
    local mode=$2
    
    print_substep "Uninstalling $component ($mode package)..."
    
    local pkg_name
    pkg_name=$(get_package_name "$component")
    
    if [ -z "$pkg_name" ]; then
        print_warning "Unknown package name for $component"
        return 1
    fi
    
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would uninstall: $pkg_name"
        return 0
    fi
    
    case "$mode" in
        deb)
            if dpkg -l | grep -q "^ii.*$pkg_name "; then
                print_detail "Removing package: $pkg_name"
                if sudo dpkg -r "$pkg_name" > /dev/null 2>&1; then
                    print_success "Removed package: $pkg_name"
                    # Also purge to remove configuration files
                    if sudo dpkg --purge "$pkg_name" > /dev/null 2>&1; then
                        print_detail "Purged configuration files"
                    fi
                    COMPONENT_UNINSTALLED[$component]=true
                    return 0
                else
                    print_error "Failed to remove package: $pkg_name"
                    return 1
                fi
            else
                print_info "Package $pkg_name is not installed"
                return 0
            fi
            ;;
        rpm)
            if rpm -q "$pkg_name" &> /dev/null; then
                print_detail "Removing package: $pkg_name"
                if sudo rpm -e "$pkg_name" > /dev/null 2>&1; then
                    print_success "Removed package: $pkg_name"
                    COMPONENT_UNINSTALLED[$component]=true
                    return 0
                else
                    print_error "Failed to remove package: $pkg_name"
                    return 1
                fi
            else
                print_info "Package $pkg_name is not installed"
                return 0
            fi
            ;;
        mac)
            # macOS packages typically require pkgutil or manual removal
            print_info "macOS package removal may require manual steps"
            if pkgutil --pkgs | grep -q "$pkg_name"; then
                print_detail "Package $pkg_name is registered"
                print_info "Use: sudo pkgutil --forget $pkg_name"
            fi
            COMPONENT_UNINSTALLED[$component]=true
            return 0
            ;;
    esac
}

stop_component_service() {
    local component=$1
    
    print_substep "Stopping $component service..."
    
    if [[ "$OSTYPE" != "linux-gnu"* ]] || ! command -v systemctl &> /dev/null; then
        print_info "Systemd not available. Services may need manual stopping."
        return 0
    fi
    
    case "$component" in
        "$COMPONENT_NEURONAGENT")
            if systemctl is-active --quiet neuronagent 2>/dev/null; then
                if [ "$DRY_RUN" = true ]; then
                    print_info "[DRY RUN] Would stop: sudo systemctl stop neuronagent"
                else
                    if sudo systemctl stop neuronagent &> /dev/null 2>&1; then
                        print_success "Stopped NeuronAgent service"
                        sudo systemctl disable neuronagent &> /dev/null 2>&1 || true
                    else
                        print_warning "Failed to stop NeuronAgent service"
                    fi
                fi
            else
                print_info "NeuronAgent service is not running"
            fi
            ;;
        "$COMPONENT_NEURONMCP")
            print_info "NeuronMCP runs as stdio process (no service to stop)"
            ;;
        "$COMPONENT_NEURONDESKTOP")
            print_info "NeuronDesktop service management depends on deployment method"
            ;;
    esac
}

remove_docker_services() {
    print_step "Removing Docker Services"
    
    cd "$REPO_ROOT"
    
    if [ ! -f "docker-compose.yml" ]; then
        print_error "docker-compose.yml not found in $REPO_ROOT"
        return 1
    fi
    
    # Determine docker compose command
    local compose_cmd="docker compose"
    if ! docker compose version &> /dev/null 2>&1; then
        if command -v docker-compose &> /dev/null; then
            compose_cmd="docker-compose"
        else
            print_error "Neither 'docker compose' nor 'docker-compose' found"
            return 1
        fi
    fi
    
    print_substep "Stopping and removing containers..."
    
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would run: $compose_cmd down"
        if [ "$REMOVE_DATA" = true ]; then
            print_info "[DRY RUN] Would also remove volumes: $compose_cmd down -v"
        fi
        # Mark components as would-be uninstalled for summary
        for component in "${SELECTED_COMPONENTS[@]}"; do
            COMPONENT_UNINSTALLED[$component]=true
        done
        return 0
    fi
    
    # Determine which services to remove based on selected components
    local services_to_remove=()
    if contains_element "$COMPONENT_NEURONDB" "${SELECTED_COMPONENTS[@]}"; then
        services_to_remove+=("neurondb" "neurondb-cuda" "neurondb-rocm" "neurondb-metal")
    fi
    if contains_element "$COMPONENT_NEURONAGENT" "${SELECTED_COMPONENTS[@]}"; then
        services_to_remove+=("neuronagent" "neuronagent-cuda" "neuronagent-rocm" "neuronagent-metal")
    fi
    if contains_element "$COMPONENT_NEURONMCP" "${SELECTED_COMPONENTS[@]}"; then
        services_to_remove+=("neuronmcp" "neuronmcp-cuda" "neuronmcp-rocm" "neuronmcp-metal")
    fi
    
    # Stop and remove services
    if [ "$REMOVE_DATA" = true ]; then
        print_detail "Removing containers and volumes..."
        if $compose_cmd down -v > /dev/null 2>&1; then
            print_success "Removed Docker containers and volumes"
        else
            print_warning "Some containers or volumes may not have been removed"
        fi
    else
        print_detail "Removing containers (keeping volumes)..."
        if $compose_cmd down > /dev/null 2>&1; then
                    print_success "Removed Docker containers"
            print_info "Volumes preserved. Use --remove-data to remove volumes."
        else
            print_warning "Some containers may not have been removed"
        fi
    fi
    
    # Mark components as uninstalled
    for component in "${SELECTED_COMPONENTS[@]}"; do
        COMPONENT_UNINSTALLED[$component]=true
    done
    
    # Optionally remove images
    print_info "Docker images are preserved. Remove manually if needed:"
    print_info "  docker rmi neurondb:cpu-pg17 neuronagent:latest neurondb-mcp:latest"
    
    return 0
}

remove_database_schemas() {
    if [ "$REMOVE_DATA" != true ]; then
        print_info "Skipping database schema removal (use --remove-data to enable)"
        return 0
    fi
    
    print_step "Removing Database Schemas"
    print_warning "This will remove all data for selected components!"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would remove database schemas"
        return 0
    fi
    
    # Remove NeuronAgent schema
    if contains_element "$COMPONENT_NEURONAGENT" "${SELECTED_COMPONENTS[@]}"; then
        print_substep "Removing NeuronAgent schema..."
        # Note: Actual schema removal would require dropping tables/schemas
        # This is destructive and should be done carefully
        print_warning "Manual schema cleanup may be required"
    fi
    
    # Remove NeuronMCP schema
    if contains_element "$COMPONENT_NEURONMCP" "${SELECTED_COMPONENTS[@]}"; then
        print_substep "Removing NeuronMCP schema..."
        print_warning "Manual schema cleanup may be required"
    fi
    
    # Remove NeuronDB extension
    if contains_element "$COMPONENT_NEURONDB" "${SELECTED_COMPONENTS[@]}"; then
        print_substep "Removing NeuronDB extension..."
        if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "DROP EXTENSION IF EXISTS neurondb CASCADE;" > /dev/null 2>&1; then
            print_success "Removed NeuronDB extension"
        else
            print_warning "Failed to remove NeuronDB extension (may not exist)"
        fi
    fi
    
    return 0
}

uninstall_packages_mode() {
    print_header
    
    print_section "Component Uninstallation"
    print_info "Components to uninstall: ${SELECTED_COMPONENTS[*]}"
    
    if [ "$REMOVE_DATA" = true ]; then
        print_warning "Data removal is enabled - all data will be deleted!"
    fi
    
    # Check prerequisites (only need package manager for package modes)
    if [ "$MODE" != "docker" ]; then
        if ! check_prerequisites_packages; then
            print_error "Cannot proceed without package manager"
            exit 1
        fi
    fi
    
    # Stop services first
    print_section "Stopping Services"
    for component in "${SELECTED_COMPONENTS[@]}"; do
        stop_component_service "$component"
    done
    
    # Remove packages
    if [ "$MODE" != "docker" ]; then
        print_section "Removing Packages"
        for component in "${SELECTED_COMPONENTS[@]}"; do
            if [ "$component" != "$COMPONENT_NEURONDESKTOP" ]; then
                uninstall_component_package "$component" "$MODE" || true
            else
                print_warning "NeuronDesktop package removal not yet implemented"
            fi
        done
    fi
    
    # Remove Docker services
    if [ "$MODE" = "docker" ]; then
        remove_docker_services || exit 1
    fi
    
    # Remove database schemas (if requested)
    if [ "$REMOVE_DATA" = true ]; then
        if check_database_connection &> /dev/null; then
            remove_database_schemas || true
        else
            print_warning "Cannot connect to database - skipping schema removal"
        fi
    fi
    
    # Summary
    print_section "Uninstallation Summary"
    print_table_header
    
    for component in "${SELECTED_COMPONENTS[@]}"; do
        local status="Unknown"
        if [ "$DRY_RUN" = true ]; then
            status="Would Uninstall"
        elif [ -n "${COMPONENT_UNINSTALLED[$component]:-}" ]; then
            status="Uninstalled"
        else
            status="Pending"
        fi
        print_table_row "$component" "$status" "N/A" "-"
    done
    
    echo ""
    print_success "Uninstallation process completed"
    
    if [ "$REMOVE_DATA" = false ] && [ "$MODE" = "docker" ]; then
        echo ""
        print_info "Note: Docker volumes were preserved"
        print_info "Remove volumes: docker volume rm neurondb-data neurondb-cuda-data etc."
    fi
}

# ============================================================================
# Service Management Functions
# ============================================================================

start_component_service() {
    local component=$1
    
    if [ "$SKIP_SERVICES" = true ]; then
        print_info "Skipping service startup (--skip-services flag)"
        return 0
    fi
    
    if [[ "$OSTYPE" != "linux-gnu"* ]] || ! command -v systemctl &> /dev/null; then
        print_info "Systemd not available. Please start services manually."
        return 0
    fi
    
    case "$component" in
        "$COMPONENT_NEURONAGENT")
            print_substep "Starting NeuronAgent service..."
            if systemctl is-active --quiet neuronagent 2>/dev/null; then
                print_success "NeuronAgent service is already running"
                COMPONENT_STARTED[$component]=true
            else
                if [ "$DRY_RUN" = true ]; then
                    print_info "[DRY RUN] Would start: sudo systemctl start neuronagent"
                else
                    if sudo systemctl enable --now neuronagent &> /dev/null 2>&1; then
                        print_success "NeuronAgent service started"
                        COMPONENT_STARTED[$component]=true
                    else
                        print_warning "NeuronAgent service not found or failed to start"
                        print_info "Start manually: sudo systemctl start neuronagent"
                    fi
                fi
            fi
            ;;
        "$COMPONENT_NEURONMCP")
            print_info "NeuronMCP runs as a stdio process (no service to manage)"
            ;;
        "$COMPONENT_NEURONDESKTOP")
            print_info "NeuronDesktop typically runs via Docker Compose or manually"
            ;;
    esac
}

# ============================================================================
# Verification Functions
# ============================================================================

verify_component() {
    local component=$1
    
    print_substep "Verifying $component installation..."
    
    case "$component" in
        "$COMPONENT_NEURONDB")
            if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT neurondb.version();" &> /dev/null 2>&1; then
                local version
                version=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT neurondb.version();" 2>/dev/null | xargs || echo "unknown")
                print_success "NeuronDB is working (version: $version)"
                COMPONENT_VERIFIED[$component]=true
                return 0
            else
                print_error "NeuronDB verification failed"
                return 1
            fi
            ;;
        "$COMPONENT_NEURONAGENT")
            if curl -s -f http://localhost:8080/health &> /dev/null 2>&1; then
                print_success "NeuronAgent API is responding"
                COMPONENT_VERIFIED[$component]=true
                return 0
            else
                print_warning "NeuronAgent API not responding (may not be running)"
                return 1
            fi
            ;;
        "$COMPONENT_NEURONMCP")
            # NeuronMCP is harder to verify without stdio interaction
            if command -v neurondb-mcp &> /dev/null; then
                print_success "NeuronMCP binary is available"
                COMPONENT_VERIFIED[$component]=true
                return 0
            else
                print_warning "NeuronMCP binary not found in PATH"
                return 1
            fi
            ;;
        "$COMPONENT_NEURONDESKTOP")
            # Check if API is accessible
            if curl -s -f http://localhost:8081/health &> /dev/null 2>&1; then
                print_success "NeuronDesktop API is responding"
                COMPONENT_VERIFIED[$component]=true
                return 0
            else
                print_info "NeuronDesktop API not responding (may not be running)"
                return 0  # Don't fail - service might not be started
            fi
            ;;
    esac
}

# ============================================================================
# Docker Functions
# ============================================================================

setup_docker_services() {
    print_step "Setting Up Docker Services"
    
    cd "$REPO_ROOT"
    
    if [ ! -f "docker-compose.yml" ]; then
        print_error "docker-compose.yml not found in $REPO_ROOT"
        return 1
    fi
    
    # Determine docker compose command
    local compose_cmd="docker compose"
    if ! docker compose version &> /dev/null 2>&1; then
        if command -v docker-compose &> /dev/null; then
            compose_cmd="docker-compose"
        else
            print_error "Neither 'docker compose' nor 'docker-compose' found"
            return 1
        fi
    fi
    
    print_substep "Using Docker Compose command: $compose_cmd"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would run: $compose_cmd up -d --build"
        return 0
    fi
    
    # Determine which services to start based on selected components
    local profiles=""
    if contains_element "$COMPONENT_NEURONDB" "${SELECTED_COMPONENTS[@]}"; then
        profiles="default cpu"
    fi
    
    print_substep "Building and starting Docker services..."
    if [ -n "$profiles" ]; then
        for profile in $profiles; do
            print_detail "Starting services with profile: $profile"
        done
    fi
    
    if $compose_cmd up -d --build > /dev/null 2>&1; then
        print_success "Docker services started"
    else
        print_error "Failed to start Docker services"
        print_info "Check logs: $compose_cmd logs"
        return 1
    fi
    
    # Wait for services to be healthy
    print_substep "Waiting for services to be healthy..."
    local max_wait=120
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if $compose_cmd ps 2>/dev/null | grep -q "neurondb.*healthy\|neurondb.*Up"; then
            break
        fi
        sleep 5
        waited=$((waited + 5))
        if [ $((waited % 15)) -eq 0 ]; then
            print_detail "Still waiting... (${waited}s/${max_wait}s)"
        fi
    done
    echo ""
    
    # Show service status
    print_substep "Service status:"
    $compose_cmd ps 2>/dev/null | head -10
    
    # Verify services
    verify_docker_services "$compose_cmd"
    
    return 0
}

verify_docker_services() {
    local compose_cmd=$1
    
    print_step "Verifying Docker Services"
    
    if contains_element "$COMPONENT_NEURONDB" "${SELECTED_COMPONENTS[@]}"; then
        print_substep "Verifying NeuronDB..."
        if $compose_cmd exec -T neurondb psql -U neurondb -d neurondb -c "SELECT neurondb.version();" &> /dev/null 2>&1; then
            local version
            version=$($compose_cmd exec -T neurondb psql -U neurondb -d neurondb -t -c "SELECT neurondb.version();" 2>/dev/null | xargs || echo "unknown")
            print_success "NeuronDB is working (version: $version)"
            COMPONENT_VERIFIED[$COMPONENT_NEURONDB]=true
        else
            print_warning "NeuronDB verification failed (may still be starting)"
        fi
    fi
    
    if contains_element "$COMPONENT_NEURONAGENT" "${SELECTED_COMPONENTS[@]}"; then
        print_substep "Verifying NeuronAgent..."
        if curl -s -f http://localhost:8080/health &> /dev/null 2>&1; then
            print_success "NeuronAgent API is responding"
            COMPONENT_VERIFIED[$COMPONENT_NEURONAGENT]=true
        else
            print_warning "NeuronAgent may still be starting up"
        fi
    fi
    
    if contains_element "$COMPONENT_NEURONMCP" "${SELECTED_COMPONENTS[@]}"; then
        print_substep "Verifying NeuronMCP..."
        if $compose_cmd ps 2>/dev/null | grep -q "neurondb-mcp.*Up"; then
            print_success "NeuronMCP container is running"
            COMPONENT_VERIFIED[$COMPONENT_NEURONMCP]=true
        else
            print_warning "NeuronMCP container not found"
        fi
    fi
}

# ============================================================================
# Main Setup Functions
# ============================================================================

setup_packages_mode() {
    print_header
    
    print_section "Component Selection"
    print_info "Selected components: ${SELECTED_COMPONENTS[*]}"
    
    # Resolve dependencies
    local resolved_components
    read -ra resolved_components <<< "$(resolve_dependencies "${SELECTED_COMPONENTS[@]}")"
    if [ ${#resolved_components[@]} -gt ${#SELECTED_COMPONENTS[@]} ]; then
        print_info "Including dependencies: ${resolved_components[*]}"
    fi
    
    # Check prerequisites
    if ! check_prerequisites_packages; then
        exit 1
    fi
    
    # Check database connection (required for most components)
    if contains_element "$COMPONENT_NEURONDB" "${resolved_components[@]}" || \
       contains_element "$COMPONENT_NEURONAGENT" "${resolved_components[@]}" || \
       contains_element "$COMPONENT_NEURONMCP" "${resolved_components[@]}"; then
        if ! check_database_connection; then
            print_error "Database connection required but failed"
            exit 1
        fi
    fi
    
    # Install packages
    print_section "Package Installation"
    for component in "${resolved_components[@]}"; do
        if [ "$component" = "$COMPONENT_NEURONDESKTOP" ]; then
            print_warning "NeuronDesktop package installation not yet implemented"
            continue
        fi
        
        if ! install_component_package "$component" "$MODE"; then
            print_error "Failed to install $component"
            exit 1
        fi
        check_prerequisites_component "$component"
    done
    
    # Setup database schemas
    if [ "$SKIP_SETUP" != true ]; then
        print_section "Database Schema Setup"
        
        if contains_element "$COMPONENT_NEURONDB" "${resolved_components[@]}"; then
            setup_neurondb_schema || exit 1
        fi
        
        if contains_element "$COMPONENT_NEURONMCP" "${resolved_components[@]}"; then
            setup_neuronomcp_schema || true  # Don't fail if it already exists
        fi
        
        if contains_element "$COMPONENT_NEURONAGENT" "${resolved_components[@]}"; then
            setup_neuronagent_schema || true  # Don't fail if it already exists
        fi
        
        if contains_element "$COMPONENT_NEURONDESKTOP" "${resolved_components[@]}"; then
            setup_neurondesktop_schema || true  # Don't fail if it already exists
        fi
    else
        print_info "Skipping database setup (--skip-setup flag)"
    fi
    
    # Start services
    print_section "Service Management"
    for component in "${resolved_components[@]}"; do
        start_component_service "$component"
    done
    
    # Verify installation
    print_section "Verification"
    for component in "${SELECTED_COMPONENTS[@]}"; do
        verify_component "$component" || true
    done
    
    # Summary
    print_summary
}

setup_docker_mode() {
    print_header
    
    print_section "Component Selection"
    print_info "Selected components: ${SELECTED_COMPONENTS[*]}"
    
    # Resolve dependencies
    local resolved_components
    read -ra resolved_components <<< "$(resolve_dependencies "${SELECTED_COMPONENTS[@]}")"
    if [ ${#resolved_components[@]} -gt ${#SELECTED_COMPONENTS[@]} ]; then
        print_info "Including dependencies: ${resolved_components[*]}"
    fi
    
    # Check prerequisites
    if ! check_prerequisites_docker; then
        exit 1
    fi
    
    # Adjust defaults for Docker
    if [ "$DB_PORT" = "5432" ]; then
        DB_PORT="5433"
        print_info "Using Docker port 5433 for PostgreSQL"
    fi
    if [ "$DB_USER" = "postgres" ]; then
        DB_USER="neurondb"
        print_info "Using Docker default user: neurondb"
    fi
    
    # Setup Docker services
    setup_docker_services || exit 1
    
    # Summary
    print_summary
}

print_summary() {
    print_section "Setup Summary"
    
    print_table_header
    
    for component in "${SELECTED_COMPONENTS[@]}"; do
        local status="Unknown"
        local version="N/A"
        local details="-"
        
        if [ -n "${COMPONENT_INSTALLED[$component]:-}" ]; then
            status="Installed"
        fi
        if [ -n "${COMPONENT_STARTED[$component]:-}" ]; then
            status="Running"
        fi
        if [ -n "${COMPONENT_VERIFIED[$component]:-}" ]; then
            status="Verified"
        fi
        
        print_table_row "$component" "$status" "$version" "$details"
    done
    
    echo ""
    print_success "Setup process completed"
    
    if [ "$MODE" = "docker" ]; then
        echo ""
        print_info "Docker services are running"
        print_info "  View logs: docker compose logs -f [service-name]"
        print_info "  Stop services: docker compose down"
        print_info "  Service status: docker compose ps"
    else
        echo ""
        print_info "Components installed via packages"
        print_info "  Database: $DB_HOST:$DB_PORT/$DB_NAME"
        if contains_element "$COMPONENT_NEURONAGENT" "${SELECTED_COMPONENTS[@]}"; then
            print_info "  NeuronAgent API: http://localhost:8080"
        fi
        if contains_element "$COMPONENT_NEURONDESKTOP" "${SELECTED_COMPONENTS[@]}"; then
            print_info "  NeuronDesktop API: http://localhost:8081"
            print_info "  NeuronDesktop UI: http://localhost:3000"
        fi
    fi
    
    if [ "$VERBOSE" = true ]; then
        echo ""
        print_info "Detailed log available at: $LOG_FILE"
    fi
}

# ============================================================================
# Help and Usage
# ============================================================================

show_help() {
    cat << EOF
${BOLD}NeuronDB Ecosystem Setup Script${NC}
${DIM}Version: $VERSION${NC}

${BOLD}Usage:${NC}
    ./neurondb-setup.sh --mode MODE [COMPONENT_OPTIONS] [OPTIONS]

${BOLD}Modes:${NC}
    docker          Use Docker Compose (recommended for quick start)
    deb             Install DEB packages (Ubuntu/Debian)
    rpm             Install RPM packages (RHEL/CentOS/Rocky/Fedora)
    mac             Install macOS packages (.pkg)

${BOLD}Component Selection (choose one):${NC}
    --all                           Install all components
    --components COMP1 [COMP2 ...]  Install specific components
                                    Available: NeuronDB, NeuronAgent, NeuronMCP, NeuronDesktop

${BOLD}Examples:${NC}
    # Install all components via Docker
    ./neurondb-setup.sh --mode docker --all

    # Install specific components (DEB packages)
    ./neurondb-setup.sh --mode deb --components NeuronDB NeuronAgent

    # Install with custom database settings
    ./neurondb-setup.sh --mode rpm --components NeuronDB NeuronMCP \\
        --db-host db.example.com --db-password secret

    # Dry run to see what would happen
    ./neurondb-setup.sh --mode docker --all --dry-run

    # Uninstall all components (Docker)
    ./neurondb-setup.sh --mode docker --all --uninstall

    # Uninstall with data removal
    ./neurondb-setup.sh --mode deb --components NeuronDB NeuronAgent --uninstall --remove-data

${BOLD}Database Options:${NC}
    --db-host HOST          Database host (default: localhost)
    --db-port PORT          Database port (default: 5432, 5433 for docker)
    --db-name NAME          Database name (default: neurondb)
    --db-user USER          Database user (default: postgres, neurondb for docker)
    --db-password PASS      Database password (default: neurondb)

${BOLD}Other Options:${NC}
    --skip-setup            Skip database schema setup (packages only)
    --skip-services         Skip starting services (packages only)
    --uninstall             Uninstall selected components
    --remove-data           Remove data/schemas during uninstall (use with caution!)
    --verbose, -v           Enable verbose output
    --dry-run               Show what would be done without making changes
    --help, -h              Show this help message

${BOLD}Components:${NC}
    ${BOLD}NeuronDB${NC}      - PostgreSQL extension for vector search and ML algorithms
    ${BOLD}NeuronAgent${NC}   - REST API server for AI agent runtime
    ${BOLD}NeuronMCP${NC}     - MCP protocol server for desktop clients (Claude Desktop, etc.)
    ${BOLD}NeuronDesktop${NC} - Unified web interface for managing all components

${BOLD}Dependencies:${NC}
    • NeuronAgent requires NeuronDB
    • NeuronMCP requires NeuronDB
    • NeuronDesktop requires NeuronDB
    (Dependencies are automatically resolved)

${BOLD}Documentation:${NC}
    README.md                               - Complete documentation
    QUICKSTART.md                           - Quick start guide
    scripts/verify_neurondb_integration.sh  - Comprehensive verification tests

${BOLD}Exit Codes:${NC}
    0   - Success
    1   - Error (missing prerequisites, installation failure, etc.)
    2   - Invalid arguments

EOF
}

# ============================================================================
# Argument Parsing
# ============================================================================

parse_arguments() {
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    local has_components=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                MODE="$2"
                shift 2
                ;;
            --all)
                SELECTED_COMPONENTS=("${ALL_COMPONENTS[@]}")
                has_components=true
                shift
                ;;
            --components)
                shift
                while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                    if validate_component "$1"; then
                        if ! contains_element "$1" "${SELECTED_COMPONENTS[@]}"; then
                            SELECTED_COMPONENTS+=("$1")
                        fi
                    fi
                    shift
                done
                has_components=true
                ;;
            --db-host)
                DB_HOST="$2"
                shift 2
                ;;
            --db-port)
                DB_PORT="$2"
                shift 2
                ;;
            --db-name)
                DB_NAME="$2"
                shift 2
                ;;
            --db-user)
                DB_USER="$2"
                shift 2
                ;;
            --db-password)
                DB_PASSWORD="$2"
                shift 2
                ;;
            --skip-setup)
                SKIP_SETUP=true
                shift
                ;;
            --skip-services)
                SKIP_SERVICES=true
                shift
                ;;
            --verbose|-v)
                VERBOSE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --uninstall)
                UNINSTALL=true
                shift
                ;;
            --remove-data)
                REMOVE_DATA=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo ""
                show_help
                exit 2
                ;;
        esac
    done
    
    # Validate mode
    if [ -z "$MODE" ]; then
        print_error "Mode is required. Use --mode docker|deb|rpm|mac"
        echo ""
        show_help
        exit 2
    fi
    
    if [[ ! "$MODE" =~ ^(docker|deb|rpm|mac)$ ]]; then
        print_error "Invalid mode: $MODE. Must be one of: docker, deb, rpm, mac"
        exit 2
    fi
    
    # Validate component selection
    if [ "$has_components" = false ]; then
        print_error "Component selection is required. Use --all or --components"
        echo ""
        show_help
        exit 2
    fi
    
    if [ ${#SELECTED_COMPONENTS[@]} -eq 0 ]; then
        print_error "No valid components selected"
        exit 2
    fi
    
    # Log configuration
    log_info "Mode: $MODE"
    log_info "Action: $([ "$UNINSTALL" = true ] && echo "uninstall" || echo "install")"
    log_info "Components: ${SELECTED_COMPONENTS[*]}"
    log_info "Database: $DB_HOST:$DB_PORT/$DB_NAME (user: $DB_USER)"
    if [ "$REMOVE_DATA" = true ]; then
        log_info "Data removal: enabled"
    fi
}

# ============================================================================
# Main Entry Point
# ============================================================================

main() {
    # Initialize log file
    touch "$LOG_FILE"
    log_info "=== NeuronDB Setup Script Started ==="
    log_info "Arguments: $*"
    
    parse_arguments "$@"
    
    if [ "$UNINSTALL" = true ]; then
        case "$MODE" in
            docker|deb|rpm|mac)
                uninstall_packages_mode
                ;;
            *)
                print_error "Invalid mode: $MODE"
                exit 2
                ;;
        esac
    else
        case "$MODE" in
            docker)
                setup_docker_mode
                ;;
            deb|rpm|mac)
                setup_packages_mode
                ;;
            *)
                print_error "Invalid mode: $MODE"
                exit 2
                ;;
        esac
    fi
    
    log_info "=== NeuronDB Setup Script Completed ==="
}

main "$@"
