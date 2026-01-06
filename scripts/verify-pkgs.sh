#!/bin/bash
#
# verify-pkgs.sh - Verify DEB/RPM packages from GitHub Actions
#
# This script:
#   1. Detects OS and package format (DEB or RPM)
#   2. Downloads latest packages from GitHub Actions workflows
#   3. Verifies 100% that all required runtime files are present
#   4. Validates checksums and package integrity
#
# Usage:
#   ./scripts/verify-pkgs.sh [OPTIONS] [MODULE...]
#

set -euo pipefail

# Script version
VERSION="1.0.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/tmp/neurondb-pkg-verify}"
ALL_MODULES=("neurondb" "neuronagent" "neuronmcp" "neurondesktop")
MODULES=()
VERBOSE=false
KEEP_DOWNLOADS=false
SKIP_CHECKSUMS=false
SKIP_DOWNLOAD=false
DRY_RUN=false
JSON_OUTPUT=false
QUIET=false
VAGRANT_MODE=false
VAGRANT_FILE=""
INSTALL_PACKAGES=false
RUN_TESTS=false
FULL_TEST=false
REPO="${GITHUB_REPO:-neurondb/neurondb}"

# Statistics
TOTAL_PACKAGES=0
VERIFIED_PACKAGES=0
FAILED_PACKAGES=0

# Print version
show_version() {
    echo "verify-pkgs.sh version $VERSION"
    exit 0
}

# Print help message
show_help() {
    cat << EOF
${CYAN}verify-pkgs.sh${NC} - Verify DEB/RPM packages from GitHub Actions

${BLUE}USAGE:${NC}
    $0 [OPTIONS] [MODULE...]

${BLUE}DESCRIPTION:${NC}
    Downloads and verifies packages from GitHub Actions workflows.
    Verifies 100% that all required runtime files are present.

${BLUE}MODULES:${NC}
    neurondb       - PostgreSQL extension package
    neuronagent    - AI agent runtime package
    neuronmcp      - Model Context Protocol server package
    neurondesktop  - Web-based desktop interface package

    If no modules specified, all modules are verified.

${BLUE}OPTIONS:${NC}
    -h, --help              Show this help message
    -v, --verbose           Enable verbose output
    -q, --quiet             Suppress non-error output
    -V, --version           Show version information
    -k, --keep-downloads    Keep downloaded packages after verification
    -s, --skip-checksums    Skip checksum verification
    --skip-download         Use existing downloaded packages (skip download)
    --dry-run               Show what would be done without actually doing it
    -d, --download-dir DIR  Directory to download packages (default: /tmp/neurondb-pkg-verify)
    -r, --repo REPO         GitHub repository (default: neurondb/neurondb)
    --json                  Output results in JSON format
    --debian, --deb         Force DEB package format
    --rpm                   Force RPM package format
    --all                   Verify all modules (default)
    --vagrant               Run verification inside Vagrant VM (auto-detects DEB/RPM)
    --vagrant-file FILE     Specify Vagrantfile to use (default: auto-detect)
    --install               Install packages after verification (Vagrant mode only)
    --test                  Run integration and functionality tests after installation
    --full                  Full test: verify + install + test (Vagrant mode only)

${BLUE}EXAMPLES:${NC}
    # Verify all packages
    $0

    # Verify specific modules
    $0 neurondb neuronagent

    # Verbose output with keeping downloads
    $0 -v -k

    # Use existing downloads (skip download)
    $0 --skip-download

    # Custom download directory
    $0 -d /my/packages

    # JSON output for scripting
    $0 --json > results.json

    # Run verification in Vagrant VM
    $0 --vagrant

    # Run in specific Vagrant VM
    $0 --vagrant --vagrant-file Vagrantfile.rocky

    # Full test: verify, install, and test in Vagrant
    $0 --vagrant --full

    # Verify and install (no tests)
    $0 --vagrant --install

${BLUE}ENVIRONMENT VARIABLES:${NC}
    DOWNLOAD_DIR            Directory for downloaded packages
    GITHUB_REPO             GitHub repository (owner/repo)
    KEEP_DOWNLOADS          Set to 1 to keep downloads (same as -k)

${BLUE}EXIT CODES:${NC}
    0   All packages verified successfully
    1   One or more packages failed verification
    2   Invalid arguments or missing prerequisites
    3   Failed to download packages

${BLUE}AUTHOR:${NC}
    NeuronDB Team

${BLUE}VERSION:${NC}
    $VERSION
EOF
    exit 0
}

# Parse command line arguments
parse_args() {
    local modules_specified=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                show_help
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -q|--quiet)
                QUIET=true
                shift
                ;;
            -V|--version)
                show_version
                ;;
            -k|--keep-downloads)
                KEEP_DOWNLOADS=true
                shift
                ;;
            -s|--skip-checksums)
                SKIP_CHECKSUMS=true
                shift
                ;;
            --skip-download)
                SKIP_DOWNLOAD=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -d|--download-dir)
                DOWNLOAD_DIR="$2"
                shift 2
                ;;
            -r|--repo)
                REPO="$2"
                shift 2
                ;;
            --json)
                JSON_OUTPUT=true
                shift
                ;;
            --debian|--deb)
                PKG_FORMAT="deb"
                PKG_EXT="deb"
                shift
                ;;
            --rpm)
                PKG_FORMAT="rpm"
                PKG_EXT="rpm"
                shift
                ;;
            --all)
                MODULES=("${ALL_MODULES[@]}")
                modules_specified=true
                shift
                ;;
            --vagrant)
                VAGRANT_MODE=true
                shift
                ;;
            --vagrant-file)
                VAGRANT_FILE="$2"
                shift 2
                ;;
            --install)
                INSTALL_PACKAGES=true
                shift
                ;;
            --test)
                RUN_TESTS=true
                shift
                ;;
            --full)
                FULL_TEST=true
                INSTALL_PACKAGES=true
                RUN_TESTS=true
                shift
                ;;
            neurondb|neuronagent|neuronmcp|neurondesktop)
                if [[ ! " ${MODULES[*]} " =~ " $1 " ]]; then
                    MODULES+=("$1")
                fi
                modules_specified=true
                shift
                ;;
            *)
                echo -e "${RED}Error: Unknown option or module: $1${NC}" >&2
                echo "Use --help for usage information" >&2
                exit 2
                ;;
        esac
    done

    # If no modules specified, use all
    if [ "$modules_specified" = false ]; then
        MODULES=("${ALL_MODULES[@]}")
    fi

    # Validate modules
    for module in "${MODULES[@]}"; do
        if [[ ! " ${ALL_MODULES[*]} " =~ " $module " ]]; then
            echo -e "${RED}Error: Invalid module: $module${NC}" >&2
            echo "Valid modules: ${ALL_MODULES[*]}" >&2
            exit 2
        fi
    done
}

# Logging functions
log_info() {
    if [ "$QUIET" = false ]; then
        echo -e "${BLUE}ℹ${NC} $*" >&2
    fi
}

log_success() {
    if [ "$QUIET" = false ]; then
        echo -e "${GREEN}✓${NC} $*" >&2
    fi
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $*" >&2
}

log_error() {
    echo -e "${RED}✗${NC} $*" >&2
}

log_verbose() {
    if [ "$VERBOSE" = true ] && [ "$QUIET" = false ]; then
        echo -e "${CYAN}→${NC} $*" >&2
    fi
}

log_debug() {
    if [ "$VERBOSE" = true ] && [ "$QUIET" = false ]; then
        echo -e "${CYAN}[DEBUG]${NC} $*" >&2
    fi
}

# JSON output functions
json_start() {
    if [ "$JSON_OUTPUT" = true ]; then
        echo "{"
        echo "  \"version\": \"$VERSION\","
        echo "  \"os\": {"
        echo "    \"id\": \"$OS_ID\","
        echo "    \"version\": \"$OS_VERSION\","
        echo "    \"package_format\": \"$PKG_FORMAT\""
        echo "  },"
        echo "  \"modules\": ["
        JSON_FIRST=true
    fi
}

json_module_start() {
    local module="$1"
    if [ "$JSON_OUTPUT" = true ]; then
        if [ "$JSON_FIRST" = false ]; then
            echo ","
        fi
        JSON_FIRST=false
        echo "    {"
        echo "      \"name\": \"$module\","
        echo "      \"status\": \"verifying\","
        echo "      \"errors\": [],"
        echo "      \"warnings\": [],"
        JSON_MODULE_ERRORS=()
        JSON_MODULE_WARNINGS=()
    fi
}

json_module_end() {
    local module="$1"
    local status="$2"
    if [ "$JSON_OUTPUT" = true ]; then
        echo "      \"status\": \"$status\","
        echo "      \"error_count\": ${#JSON_MODULE_ERRORS[@]},"
        echo "      \"warning_count\": ${#JSON_MODULE_WARNINGS[@]}"
        echo "    }"
    fi
}

json_end() {
    if [ "$JSON_OUTPUT" = true ]; then
        echo "  ],"
        echo "  \"summary\": {"
        echo "    \"total\": $TOTAL_PACKAGES,"
        echo "    \"verified\": $VERIFIED_PACKAGES,"
        echo "    \"failed\": $FAILED_PACKAGES"
        echo "  }"
        echo "}"
    fi
}

json_add_error() {
    if [ "$JSON_OUTPUT" = true ]; then
        JSON_MODULE_ERRORS+=("$1")
    fi
}

json_add_warning() {
    if [ "$JSON_OUTPUT" = true ]; then
        JSON_MODULE_WARNINGS+=("$1")
    fi
}

# Cleanup on exit
cleanup() {
    # Don't cleanup if running in Vagrant mode (cleanup happens in VM)
    if [ "$VAGRANT_MODE" = true ]; then
        return 0
    fi
    if [ "$KEEP_DOWNLOADS" = false ]; then
        log_verbose "Cleaning up download directory: $DOWNLOAD_DIR"
        rm -rf "$DOWNLOAD_DIR"
    else
        log_info "Keeping downloads in: $DOWNLOAD_DIR"
    fi
}
trap cleanup EXIT

# Detect OS and package format
detect_os() {
    # Skip if already set via command line
    if [ -n "${PKG_FORMAT:-}" ]; then
        log_verbose "Package format forced via command line: $PKG_FORMAT"
        return
    fi

    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS_ID="${ID:-}"
        OS_VERSION="${VERSION_ID:-}"
    else
        log_error "Cannot detect OS. /etc/os-release not found."
        exit 2
    fi

    # Determine package format
    case "$OS_ID" in
        ubuntu|debian)
            PKG_FORMAT="deb"
            PKG_EXT="deb"
            ;;
        rhel|centos|rocky|fedora|ol)
            PKG_FORMAT="rpm"
            PKG_EXT="rpm"
            ;;
        *)
            log_warning "Unknown OS '$OS_ID'. Defaulting to DEB format."
            PKG_FORMAT="deb"
            PKG_EXT="deb"
            ;;
    esac

    log_info "Detected OS: $OS_ID $OS_VERSION"
    log_info "Package Format: $PKG_FORMAT"
}

# Check prerequisites
check_prerequisites() {
    local missing=()

    # Check for gh CLI (only needed if not skipping download)
    if [ "$SKIP_DOWNLOAD" = false ]; then
        if ! command -v gh >/dev/null 2>&1; then
            missing+=("gh (GitHub CLI)")
        else
            # Check gh auth
            if ! gh auth status >/dev/null 2>&1; then
                echo -e "${RED}Error: GitHub CLI not authenticated. Run 'gh auth login'${NC}" >&2
                exit 1
            fi
        fi
    fi

    # Check for package inspection tools
    if [ "$PKG_FORMAT" = "deb" ]; then
        if ! command -v dpkg-deb >/dev/null 2>&1; then
            missing+=("dpkg-deb (install dpkg-dev)")
        fi
        if ! command -v sha256sum >/dev/null 2>&1; then
            missing+=("sha256sum (install coreutils)")
        fi
    else
        if ! command -v rpm >/dev/null 2>&1; then
            missing+=("rpm (install rpm)")
        fi
        if ! command -v rpm2cpio >/dev/null 2>&1; then
            missing+=("rpm2cpio (install rpm)")
        fi
        if ! command -v cpio >/dev/null 2>&1; then
            missing+=("cpio (install cpio)")
        fi
        if ! command -v sha256sum >/dev/null 2>&1; then
            missing+=("sha256sum (install coreutils)")
        fi
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required tools:"
        printf '  - %s\n' "${missing[@]}" >&2
        exit 2
    fi

    log_success "All prerequisites met"
}

# Get latest successful workflow run ID
get_latest_run() {
    local workflow="$1"
    local run_id

    if [ "$DRY_RUN" = true ]; then
        log_debug "[DRY RUN] Would get latest run for workflow: $workflow"
        echo "dry-run-run-id"
        return 0
    fi

    log_verbose "Getting latest successful run for workflow: $workflow"
    run_id=$(gh run list --repo "$REPO" --workflow="$workflow" --limit 10 \
        --json databaseId,conclusion,status \
        --jq '.[] | select(.conclusion=="success") | .databaseId' 2>/dev/null | head -1)

    if [ -z "$run_id" ]; then
        log_error "No successful run found for workflow '$workflow'"
        return 1
    fi

    log_verbose "Found run ID: $run_id"
    echo "$run_id"
}

# Download package artifacts
download_packages() {
    local module="$1"
    local workflow="${module^} - Packages"
    local run_id
    local module_dir="$DOWNLOAD_DIR/$module"

    if [ "$SKIP_DOWNLOAD" = true ]; then
        # Check if packages exist anywhere in the download directory structure
        # Search for packages recursively - match any depth
        # First try exact module name match in path
        local found_pkg=$(find "$DOWNLOAD_DIR" -type f -name "*.${PKG_EXT}" 2>/dev/null | grep -E "/${module}[_-]|/${module}/" | head -1)
        if [ -z "$found_pkg" ]; then
            # Fallback: case-insensitive search for module name
            found_pkg=$(find "$DOWNLOAD_DIR" -type f -name "*.${PKG_EXT}" 2>/dev/null | grep -i "$module" | head -1)
        fi
        if [ -n "$found_pkg" ]; then
            # Ensure module_dir exists
            mkdir -p "$module_dir"
            log_info "Skipping download, found packages in: $(dirname "$found_pkg")"
            log_verbose "Package found at: $found_pkg"
            return 0
        elif [ -d "$module_dir" ]; then
            # Check direct module_dir
            local pkg_count=$(find "$module_dir" -name "*.${PKG_EXT}" -type f 2>/dev/null | wc -l)
            if [ "$pkg_count" -gt 0 ]; then
                log_info "Skipping download, using existing packages in: $module_dir"
                return 0
            fi
        fi
        log_error "Skip download requested but no packages found for $module in $DOWNLOAD_DIR"
        log_verbose "Searched in: $DOWNLOAD_DIR for *${module}*.${PKG_EXT}"
        return 1
    fi

    if [ "$DRY_RUN" = true ]; then
        log_debug "[DRY RUN] Would download packages for module: $module"
        mkdir -p "$module_dir"
        return 0
    fi

    log_info "Downloading $module packages..."

    # Get latest successful run
    run_id=$(get_latest_run "$workflow")
    if [ -z "$run_id" ] || [ "$run_id" = "dry-run-run-id" ]; then
        return 0
    fi

    log_verbose "Run ID: $run_id"

    # Download artifacts
    mkdir -p "$module_dir"

    if gh run download --repo "$REPO" "$run_id" --dir "$module_dir" 2>/dev/null; then
        log_success "Downloaded artifacts"
        return 0
    else
        log_error "Failed to download artifacts"
        return 1
    fi
}

# Verify DEB package
verify_deb_package() {
    local pkg_file="$1"
    local module="$2"
    local pkg_name=$(basename "$pkg_file")
    local errors=0
    local warnings=0

    echo -e "\n${BLUE}Verifying $pkg_name...${NC}"

    # Check package format
    if ! file "$pkg_file" | grep -q "Debian binary package"; then
        echo -e "${RED}  ✗ Invalid DEB package format${NC}" >&2
        return 1
    fi

    # Extract package info
    local pkg_version=$(dpkg-deb -f "$pkg_file" Version 2>/dev/null || echo "unknown")
    local pkg_arch=$(dpkg-deb -f "$pkg_file" Architecture 2>/dev/null || echo "unknown")
    local pkg_size=$(du -h "$pkg_file" | cut -f1)

    echo -e "  Version: $pkg_version"
    echo -e "  Architecture: $pkg_arch"
    echo -e "  Size: $pkg_size"

    # Extract to temp directory for inspection
    local extract_dir=$(mktemp -d)
    dpkg-deb -x "$pkg_file" "$extract_dir" >/dev/null 2>&1

    # Module-specific verification
    case "$module" in
        neurondb)
            verify_neurondb_deb "$extract_dir" "$module" || errors=$((errors + 1))
            ;;
        neuronagent)
            verify_neuronagent_deb "$extract_dir" "$module" || errors=$((errors + 1))
            ;;
        neuronmcp)
            verify_neuronmcp_deb "$extract_dir" "$module" || errors=$((errors + 1))
            ;;
        neurondesktop)
            verify_neurondesktop_deb "$extract_dir" "$module" || errors=$((errors + 1))
            ;;
    esac

    # Cleanup
    rm -rf "$extract_dir"

    if [ $errors -eq 0 ]; then
        echo -e "${GREEN}  ✓ Package verification passed${NC}"
        return 0
    else
        echo -e "${RED}  ✗ Package verification failed ($errors errors)${NC}" >&2
        return 1
    fi
}

# Verify NeuronDB DEB package
verify_neurondb_deb() {
    local extract_dir="$1"
    local module="$2"
    local errors=0

    echo -e "  ${BLUE}Checking NeuronDB files...${NC}"

    # Required files
    local required_files=(
        "usr/lib/postgresql/*/lib/neurondb.so"
        "usr/share/postgresql/*/extension/neurondb--1.0.sql"
        "usr/share/postgresql/*/extension/neurondb.control"
        "usr/share/doc/neurondb/copyright"
    )

    for pattern in "${required_files[@]}"; do
        if ! find "$extract_dir" -path "$extract_dir/$pattern" -type f | grep -q .; then
            echo -e "    ${RED}✗ Missing: $pattern${NC}" >&2
            errors=$((errors + 1))
        else
            echo -e "    ${GREEN}✓ Found: $pattern${NC}"
        fi
    done

    # Check for bitcode files
    local bitcode_count=$(find "$extract_dir" -name "*.bc" -type f | wc -l)
    if [ "$bitcode_count" -eq 0 ]; then
        echo -e "    ${YELLOW}⚠ Warning: No bitcode files found${NC}"
    else
        echo -e "    ${GREEN}✓ Found $bitcode_count bitcode files${NC}"
    fi

    return $errors
}

# Verify NeuronAgent DEB package
verify_neuronagent_deb() {
    local extract_dir="$1"
    local module="$2"
    local errors=0

    echo -e "  ${BLUE}Checking NeuronAgent files...${NC}"

    # Required files
    local required_files=(
        "usr/bin/neuronagent"
        "usr/share/doc/neuronagent/copyright"
    )

    for pattern in "${required_files[@]}"; do
        if [ ! -f "$extract_dir/$pattern" ]; then
            echo -e "    ${RED}✗ Missing: $pattern${NC}" >&2
            errors=$((errors + 1))
        else
            echo -e "    ${GREEN}✓ Found: $pattern${NC}"
        fi
    done

    # Check migrations directory
    local migrations_dir="$extract_dir/usr/share/neuronagent/migrations"
    if [ ! -d "$migrations_dir" ]; then
        echo -e "    ${RED}✗ Missing: migrations directory${NC}" >&2
        errors=$((errors + 1))
    else
        local sql_count=$(find "$migrations_dir" -name "*.sql" -type f | wc -l)
        if [ "$sql_count" -eq 0 ]; then
            echo -e "    ${RED}✗ Missing: SQL migration files (0 found, expected 20)${NC}" >&2
            errors=$((errors + 1))
        else
            echo -e "    ${GREEN}✓ Found $sql_count SQL migration files${NC}"
            if [ "$sql_count" -lt 20 ]; then
                echo -e "    ${YELLOW}⚠ Warning: Expected 20 files, found $sql_count${NC}"
            fi
        fi
    fi

    # Check config example (optional but recommended)
    if [ -f "$extract_dir/etc/neuronagent/config.yaml.example" ]; then
        echo -e "    ${GREEN}✓ Found: config.yaml.example${NC}"
    else
        echo -e "    ${YELLOW}⚠ Missing: config.yaml.example (optional)${NC}"
    fi

    return $errors
}

# Verify NeuronMCP DEB package
verify_neuronmcp_deb() {
    local extract_dir="$1"
    local module="$2"
    local errors=0

    echo -e "  ${BLUE}Checking NeuronMCP files...${NC}"

    # Required files
    local required_files=(
        "usr/bin/neurondb-mcp"
        "usr/share/neuronmcp/sql"
        "usr/share/doc/neuronmcp/copyright"
    )

    for pattern in "${required_files[@]}"; do
        if [ ! -e "$extract_dir/$pattern" ]; then
            echo -e "    ${RED}✗ Missing: $pattern${NC}" >&2
            errors=$((errors + 1))
        else
            echo -e "    ${GREEN}✓ Found: $pattern${NC}"
        fi
    done

    # Check SQL files
    local sql_count=$(find "$extract_dir/usr/share/neuronmcp/sql" -name "*.sql" -type f 2>/dev/null | wc -l)
    if [ "$sql_count" -eq 0 ]; then
        echo -e "    ${RED}✗ Missing: SQL files${NC}" >&2
        errors=$((errors + 1))
    else
        echo -e "    ${GREEN}✓ Found $sql_count SQL files${NC}"
    fi

    # Check config example
    if [ -f "$extract_dir/etc/neuronmcp/mcp-config.json.example" ]; then
        echo -e "    ${GREEN}✓ Found: mcp-config.json.example${NC}"
    else
        echo -e "    ${YELLOW}⚠ Missing: mcp-config.json.example${NC}"
    fi

    return $errors
}

# Verify NeuronDesktop DEB package
verify_neurondesktop_deb() {
    local extract_dir="$1"
    local module="$2"
    local errors=0

    echo -e "  ${BLUE}Checking NeuronDesktop files...${NC}"

    # Required files
    local required_files=(
        "usr/bin/neurondesktop"
        "usr/share/neurondesktop/migrations"
        "usr/share/doc/neurondesktop/copyright"
    )

    for pattern in "${required_files[@]}"; do
        if [ ! -e "$extract_dir/$pattern" ]; then
            echo -e "    ${RED}✗ Missing: $pattern${NC}" >&2
            errors=$((errors + 1))
        else
            echo -e "    ${GREEN}✓ Found: $pattern${NC}"
        fi
    done

    # Check migration files
    local migrations_dir="$extract_dir/usr/share/neurondesktop/migrations"
    local sql_count=$(find "$migrations_dir" -name "*.sql" -type f 2>/dev/null | wc -l)
    if [ "$sql_count" -eq 0 ]; then
        echo -e "    ${RED}✗ Missing: Migration SQL files${NC}" >&2
        errors=$((errors + 1))
    else
        echo -e "    ${GREEN}✓ Found $sql_count migration files${NC}"
    fi

    # Check setup SQL file
    if [ -f "$extract_dir/usr/share/neurondesktop/neurondesktop.sql" ]; then
        echo -e "    ${GREEN}✓ Found: neurondesktop.sql${NC}"
    else
        echo -e "    ${YELLOW}⚠ Missing: neurondesktop.sql${NC}"
    fi

    return $errors
}

# Verify NeuronDB RPM package
verify_neurondb_rpm() {
    local extract_dir="$1"
    local module="$2"
    local errors=0

    echo -e "  ${BLUE}Checking NeuronDB files...${NC}"

    # Required files (RPM paths)
    local required_files=(
        "usr/pgsql-*/lib/neurondb.so"
        "usr/pgsql-*/share/extension/neurondb--1.0.sql"
        "usr/pgsql-*/share/extension/neurondb.control"
        "usr/share/doc/neurondb-*/copyright"
    )

    for pattern in "${required_files[@]}"; do
        if ! find "$extract_dir" -path "$extract_dir/$pattern" -type f 2>/dev/null | grep -q .; then
            echo -e "    ${RED}✗ Missing: $pattern${NC}" >&2
            errors=$((errors + 1))
        else
            echo -e "    ${GREEN}✓ Found: $pattern${NC}"
        fi
    done

    # Check for bitcode files
    local bitcode_count=$(find "$extract_dir" -name "*.bc" -type f 2>/dev/null | wc -l)
    if [ "$bitcode_count" -eq 0 ]; then
        echo -e "    ${YELLOW}⚠ Warning: No bitcode files found${NC}"
    else
        echo -e "    ${GREEN}✓ Found $bitcode_count bitcode files${NC}"
    fi

    return $errors
}

# Verify NeuronAgent RPM package
verify_neuronagent_rpm() {
    local extract_dir="$1"
    local module="$2"
    local errors=0

    echo -e "  ${BLUE}Checking NeuronAgent files...${NC}"

    # Required files
    local required_files=(
        "usr/bin/neuronagent"
        "usr/share/doc/neuronagent-*/copyright"
    )

    for pattern in "${required_files[@]}"; do
        if ! find "$extract_dir" -path "$extract_dir/$pattern" -type f 2>/dev/null | grep -q .; then
            echo -e "    ${RED}✗ Missing: $pattern${NC}" >&2
            errors=$((errors + 1))
        else
            echo -e "    ${GREEN}✓ Found: $pattern${NC}"
        fi
    done

    # Check migrations directory
    local migrations_dir=$(find "$extract_dir" -type d -name "migrations" -path "*/neuronagent/*" 2>/dev/null | head -1)
    if [ -z "$migrations_dir" ]; then
        echo -e "    ${RED}✗ Missing: migrations directory${NC}" >&2
        errors=$((errors + 1))
    else
        local sql_count=$(find "$migrations_dir" -name "*.sql" -type f 2>/dev/null | wc -l)
        if [ "$sql_count" -eq 0 ]; then
            echo -e "    ${RED}✗ Missing: SQL migration files (0 found, expected 20)${NC}" >&2
            errors=$((errors + 1))
        else
            echo -e "    ${GREEN}✓ Found $sql_count SQL migration files${NC}"
            if [ "$sql_count" -lt 20 ]; then
                echo -e "    ${YELLOW}⚠ Warning: Expected 20 files, found $sql_count${NC}"
            fi
        fi
    fi

    # Check config example (optional)
    if find "$extract_dir" -name "config.yaml.example" -path "*/neuronagent/*" 2>/dev/null | grep -q .; then
        echo -e "    ${GREEN}✓ Found: config.yaml.example${NC}"
    else
        echo -e "    ${YELLOW}⚠ Missing: config.yaml.example (optional)${NC}"
    fi

    return $errors
}

# Verify NeuronMCP RPM package
verify_neuronmcp_rpm() {
    local extract_dir="$1"
    local module="$2"
    local errors=0

    echo -e "  ${BLUE}Checking NeuronMCP files...${NC}"

    # Required files
    local required_files=(
        "usr/bin/neurondb-mcp"
        "usr/share/neuronmcp/sql"
        "usr/share/doc/neuronmcp-*/copyright"
    )

    for pattern in "${required_files[@]}"; do
        if ! find "$extract_dir" -path "$extract_dir/$pattern" 2>/dev/null | grep -q .; then
            echo -e "    ${RED}✗ Missing: $pattern${NC}" >&2
            errors=$((errors + 1))
        else
            echo -e "    ${GREEN}✓ Found: $pattern${NC}"
        fi
    done

    # Check SQL files
    local sql_dir=$(find "$extract_dir" -type d -name "sql" -path "*/neuronmcp/*" 2>/dev/null | head -1)
    local sql_count=$(find "$sql_dir" -name "*.sql" -type f 2>/dev/null | wc -l)
    if [ "$sql_count" -eq 0 ]; then
        echo -e "    ${RED}✗ Missing: SQL files${NC}" >&2
        errors=$((errors + 1))
    else
        echo -e "    ${GREEN}✓ Found $sql_count SQL files${NC}"
    fi

    # Check config example
    if find "$extract_dir" -name "mcp-config.json.example" -path "*/neuronmcp/*" 2>/dev/null | grep -q .; then
        echo -e "    ${GREEN}✓ Found: mcp-config.json.example${NC}"
    else
        echo -e "    ${YELLOW}⚠ Missing: mcp-config.json.example${NC}"
    fi

    return $errors
}

# Verify NeuronDesktop RPM package
verify_neurondesktop_rpm() {
    local extract_dir="$1"
    local module="$2"
    local errors=0

    echo -e "  ${BLUE}Checking NeuronDesktop files...${NC}"

    # Required files
    local required_files=(
        "usr/bin/neurondesktop"
        "usr/share/neurondesktop/migrations"
        "usr/share/doc/neurondesktop-*/copyright"
    )

    for pattern in "${required_files[@]}"; do
        if ! find "$extract_dir" -path "$extract_dir/$pattern" 2>/dev/null | grep -q .; then
            echo -e "    ${RED}✗ Missing: $pattern${NC}" >&2
            errors=$((errors + 1))
        else
            echo -e "    ${GREEN}✓ Found: $pattern${NC}"
        fi
    done

    # Check migration files
    local migrations_dir=$(find "$extract_dir" -type d -name "migrations" -path "*/neurondesktop/*" 2>/dev/null | head -1)
    local sql_count=$(find "$migrations_dir" -name "*.sql" -type f 2>/dev/null | wc -l)
    if [ "$sql_count" -eq 0 ]; then
        echo -e "    ${RED}✗ Missing: Migration SQL files${NC}" >&2
        errors=$((errors + 1))
    else
        echo -e "    ${GREEN}✓ Found $sql_count migration files${NC}"
    fi

    # Check setup SQL file
    if find "$extract_dir" -name "neurondesktop.sql" -path "*/neurondesktop/*" 2>/dev/null | grep -q .; then
        echo -e "    ${GREEN}✓ Found: neurondesktop.sql${NC}"
    else
        echo -e "    ${YELLOW}⚠ Missing: neurondesktop.sql${NC}"
    fi

    return $errors
}

# Verify checksums
verify_checksums() {
    if [ "$SKIP_CHECKSUMS" = true ]; then
        log_verbose "Skipping checksum verification"
        return 0
    fi

    local module="$1"
    local module_dir="$DOWNLOAD_DIR/$module"

    # Find checksum file
    local checksum_file=$(find "$module_dir" -name "SHA256SUMS*" -type f | head -1)
    if [ -z "$checksum_file" ]; then
        log_warning "No checksum file found"
        return 0
    fi

    log_verbose "Verifying checksums from: $checksum_file"

    # Get directory containing the packages
    local pkg_dir=$(find "$module_dir" -name "*.${PKG_EXT}" -type f -exec dirname {} \; | head -1)
    if [ -z "$pkg_dir" ]; then
        log_error "No packages found for checksum verification"
        return 1
    fi

    # Verify checksums
    if (cd "$pkg_dir" && sha256sum -c "$checksum_file" >/dev/null 2>&1); then
        log_success "Checksums valid"
        return 0
    else
        log_error "Checksum verification failed"
        log_verbose "Run manually: cd $pkg_dir && sha256sum -c $checksum_file"
        return 1
    fi
}

# Main verification function
verify_module() {
    local module="$1"
    local module_dir="$DOWNLOAD_DIR/$module"
    local total_errors=0

    json_module_start "$module"

    if [ "$JSON_OUTPUT" = false ]; then
        echo ""
        echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}Verifying $module${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    fi

    # Download packages
    if ! download_packages "$module"; then
        log_error "Failed to download $module packages"
        json_add_error "Failed to download packages"
        json_module_end "$module" "failed"
        FAILED_PACKAGES=$((FAILED_PACKAGES + 1))
        return 1
    fi

    # Find package files - search recursively in nested directories
    local packages=$(find "$DOWNLOAD_DIR" -path "*/$module/*" -name "*.${PKG_EXT}" -type f 2>/dev/null)
    if [ -z "$packages" ]; then
        # Try direct module_dir
        packages=$(find "$module_dir" -name "*.${PKG_EXT}" -type f 2>/dev/null)
    fi
    if [ -z "$packages" ]; then
        log_error "No ${PKG_EXT} packages found for $module in $DOWNLOAD_DIR"
        log_verbose "Searched in: $DOWNLOAD_DIR (pattern: */$module/*/*.${PKG_EXT})"
        json_add_error "No packages found"
        json_module_end "$module" "failed"
        FAILED_PACKAGES=$((FAILED_PACKAGES + 1))
        return 1
    fi
    
    log_verbose "Found $(echo "$packages" | wc -l) package(s) for $module"

    TOTAL_PACKAGES=$((TOTAL_PACKAGES + 1))

    # Verify each package
    for pkg in $packages; do
        if [ "$PKG_FORMAT" = "deb" ]; then
            verify_deb_package "$pkg" "$module" || total_errors=$((total_errors + 1))
        elif [ "$PKG_FORMAT" = "rpm" ]; then
            verify_rpm_package "$pkg" "$module" || total_errors=$((total_errors + 1))
        else
            log_warning "Package format $PKG_FORMAT not supported"
            total_errors=$((total_errors + 1))
        fi
    done

    # Verify checksums
    if ! verify_checksums "$module"; then
        total_errors=$((total_errors + 1))
        json_add_error "Checksum verification failed"
    fi

    if [ $total_errors -eq 0 ]; then
        log_success "$module verification PASSED"
        json_module_end "$module" "passed"
        VERIFIED_PACKAGES=$((VERIFIED_PACKAGES + 1))
        return 0
    else
        log_error "$module verification FAILED ($total_errors errors)"
        json_module_end "$module" "failed"
        FAILED_PACKAGES=$((FAILED_PACKAGES + 1))
        return 1
    fi
}

# Main execution
main() {
    local exit_code=0
    local failed_modules=()

    if [ "$JSON_OUTPUT" = false ]; then
        echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${BLUE}║     NeuronDB Package Verification Script v$VERSION           ║${NC}"
        echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
        echo ""
    fi

    json_start

    # Detect OS
    detect_os

    # Check prerequisites (unless dry run)
    if [ "$DRY_RUN" = false ]; then
        check_prerequisites
    fi

    # Create download directory
    mkdir -p "$DOWNLOAD_DIR"
    log_verbose "Download directory: $DOWNLOAD_DIR"

    # Verify all modules
    for module in "${MODULES[@]}"; do
        if ! verify_module "$module"; then
            failed_modules+=("$module")
            exit_code=1
        fi
    done

    json_end

    # Summary
    if [ "$JSON_OUTPUT" = false ]; then
        echo ""
        echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}Verification Summary${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
        echo "Total packages: $TOTAL_PACKAGES"
        echo "Verified: $VERIFIED_PACKAGES"
        echo "Failed: $FAILED_PACKAGES"

        if [ ${#failed_modules[@]} -eq 0 ]; then
            echo ""
            log_success "All packages verified successfully!"
            log_success "All required runtime files are present."
        else
            echo ""
            log_error "Verification failed for: ${failed_modules[*]}"
        fi
    fi

    return $exit_code
}

# Run in Vagrant VM
run_in_vagrant() {
    local vagrant_file_arg=""
    local script_args=()
    local auto_detect_vm=true

    # Determine which Vagrantfile to use
    if [ -n "$VAGRANT_FILE" ]; then
        vagrant_file_arg="VAGRANT_VAGRANTFILE=$VAGRANT_FILE"
        auto_detect_vm=false
    fi

    # Auto-detect VM based on package format if not specified
    if [ "$auto_detect_vm" = true ]; then
        if [ "$PKG_FORMAT" = "deb" ]; then
            VAGRANT_FILE="Vagrantfile"
        elif [ "$PKG_FORMAT" = "rpm" ]; then
            VAGRANT_FILE="Vagrantfile.rocky"
        else
            log_warning "Unknown package format, defaulting to Ubuntu (DEB)"
            VAGRANT_FILE="Vagrantfile"
        fi
        vagrant_file_arg="VAGRANT_VAGRANTFILE=$VAGRANT_FILE"
    fi

    log_info "Running verification in Vagrant VM: $VAGRANT_FILE"

    # Check if VM is running
    local vm_status=""
    if [ -n "$VAGRANT_FILE" ]; then
        vm_status=$(cd "$REPO_ROOT" && $vagrant_file_arg vagrant status 2>/dev/null | grep -E "^default" | awk '{print $2}' || echo "not_found")
    else
        vm_status=$(cd "$REPO_ROOT" && vagrant status 2>/dev/null | grep -E "^default" | awk '{print $2}' || echo "not_found")
    fi

    log_verbose "VM status: $vm_status"

    if [ "$vm_status" != "running" ]; then
        log_info "Starting Vagrant VM..."
        if [ -n "$VAGRANT_FILE" ]; then
            (cd "$REPO_ROOT" && export VAGRANT_VAGRANTFILE="$VAGRANT_FILE" && vagrant up 2>&1 | grep -v "==> default:" || true)
            if ! (cd "$REPO_ROOT" && export VAGRANT_VAGRANTFILE="$VAGRANT_FILE" && vagrant status 2>/dev/null | grep -q "running"); then
                log_error "Failed to start Vagrant VM"
                exit 1
            fi
        else
            (cd "$REPO_ROOT" && vagrant up 2>&1 | grep -v "==> default:" || true)
            if ! (cd "$REPO_ROOT" && vagrant status 2>/dev/null | grep -q "running"); then
                log_error "Failed to start Vagrant VM"
                exit 1
            fi
        fi
    fi

    # Download/copy packages if needed (when using --full or --install)
    if [ "$INSTALL_PACKAGES" = true ] || [ "$FULL_TEST" = true ]; then
        log_info "Ensuring packages are available in VM..."
        local download_script="bash /vagrant/vagrant/download-packages.sh"
        if [ -n "$VAGRANT_FILE" ]; then
            (cd "$REPO_ROOT" && export VAGRANT_VAGRANTFILE="$VAGRANT_FILE" && vagrant ssh -c "$download_script" 2>&1 | tail -5 || true)
        else
            (cd "$REPO_ROOT" && vagrant ssh -c "$download_script" 2>&1 | tail -5 || true)
        fi
        # After download, use skip-download for verification
        SKIP_DOWNLOAD=true
        script_args+=("--skip-download")
    fi

    # Build arguments to pass to script inside VM
    # Note: Use --verbose instead of -v to avoid conflict with vagrant's -v flag
    script_args=()
    if [ "$VERBOSE" = true ]; then
        script_args+=("--verbose")
    fi
    if [ "$QUIET" = true ]; then
        script_args+=("-q")
    fi
    if [ "$SKIP_CHECKSUMS" = true ]; then
        script_args+=("--skip-checksums")
    fi
    if [ "$SKIP_DOWNLOAD" = true ]; then
        script_args+=("--skip-download")
    fi
    if [ "$KEEP_DOWNLOADS" = true ]; then
        script_args+=("-k")
    fi
    if [ "$JSON_OUTPUT" = true ]; then
        script_args+=("--json")
    fi
    if [ -n "$DOWNLOAD_DIR" ] && [ "$DOWNLOAD_DIR" != "/tmp/neurondb-pkg-verify" ]; then
        script_args+=("-d" "$DOWNLOAD_DIR")
    else
        # Default to /vagrant/packages in VM
        script_args+=("-d" "/vagrant/packages")
        SKIP_DOWNLOAD=true  # Use packages from shared folder
    fi

    # Add modules if specified
    if [ ${#MODULES[@]} -gt 0 ] && [ ${#MODULES[@]} -lt ${#ALL_MODULES[@]} ]; then
        script_args+=("${MODULES[@]}")
    fi

    log_info "Executing verification in VM..."
    
    # Build command with proper quoting using printf %q
    local cmd_parts=("bash" "/vagrant/scripts/verify-pkgs.sh")
    cmd_parts+=("${script_args[@]}")
    
    # Quote each argument properly
    local quoted_cmd=""
    for part in "${cmd_parts[@]}"; do
        quoted_part=$(printf '%q' "$part")
        if [ -z "$quoted_cmd" ]; then
            quoted_cmd="$quoted_part"
        else
            quoted_cmd="$quoted_cmd $quoted_part"
        fi
    done
    
    log_verbose "Command: $quoted_cmd"

    # Run script inside VM
    if [ -n "$VAGRANT_FILE" ]; then
        (cd "$REPO_ROOT" && export VAGRANT_VAGRANTFILE="$VAGRANT_FILE" && vagrant ssh -c "$quoted_cmd" 2>&1)
        local exit_code=$?
    else
        (cd "$REPO_ROOT" && vagrant ssh -c "$quoted_cmd" 2>&1)
        local exit_code=$?
    fi

    log_verbose "VM command exited with code: $exit_code"
    
    # For full test, continue even if verification had warnings (exit_code != 0)
    # Only stop early if verification completely failed and we're not doing full test
    local verification_ok=true
    if [ $exit_code -ne 0 ] && [ "$FULL_TEST" = false ] && [ "$INSTALL_PACKAGES" = false ]; then
        return $exit_code
    fi
    
    # For full test, we continue even with warnings
    if [ $exit_code -ne 0 ] && [ "$FULL_TEST" = true ]; then
        log_warning "Verification had issues, but continuing with full test..."
        verification_ok=false
    fi
    
    # Install packages if requested (allow installation even if verification had warnings)
    if [ "$INSTALL_PACKAGES" = true ]; then
        log_info "Installing packages in VM..."
        local install_script="bash /vagrant/vagrant/install-packages.sh"
        if [ "$PKG_FORMAT" = "rpm" ]; then
            install_script="bash /vagrant/vagrant/install-packages-rpm.sh"
        fi
        
        if [ -n "$VAGRANT_FILE" ]; then
            (cd "$REPO_ROOT" && export VAGRANT_VAGRANTFILE="$VAGRANT_FILE" && vagrant ssh -c "$install_script" 2>&1)
            local install_exit=$?
        else
            (cd "$REPO_ROOT" && vagrant ssh -c "$install_script" 2>&1)
            local install_exit=$?
        fi
        
        if [ $install_exit -ne 0 ]; then
            log_error "Package installation failed"
            return $install_exit
        fi
        log_success "Packages installed successfully"
    fi
    
    # Run tests if requested (allow tests even if installation had issues)
    if [ "$RUN_TESTS" = true ]; then
        log_info "Running integration tests..."
        local test_script="bash /vagrant/vagrant/test-integration.sh"
        local func_test_script="bash /vagrant/vagrant/test-functionality.sh"
        local pg_cmd="psql"
        
        # Set PostgreSQL command based on package format
        if [ "$PKG_FORMAT" = "rpm" ]; then
            pg_cmd="/usr/pgsql-18/bin/psql"
        fi
        
        # Run integration tests
        if [ -n "$VAGRANT_FILE" ]; then
            (cd "$REPO_ROOT" && export VAGRANT_VAGRANTFILE="$VAGRANT_FILE" && vagrant ssh -c "PG_CMD='$pg_cmd' $test_script" 2>&1)
            local test_exit=$?
            (cd "$REPO_ROOT" && export VAGRANT_VAGRANTFILE="$VAGRANT_FILE" && vagrant ssh -c "PG_CMD='$pg_cmd' $func_test_script" 2>&1)
            local func_test_exit=$?
        else
            (cd "$REPO_ROOT" && vagrant ssh -c "PG_CMD='$pg_cmd' $test_script" 2>&1)
            local test_exit=$?
            (cd "$REPO_ROOT" && vagrant ssh -c "PG_CMD='$pg_cmd' $func_test_script" 2>&1)
            local func_test_exit=$?
        fi
        
        if [ $test_exit -ne 0 ] || [ $func_test_exit -ne 0 ]; then
            log_error "Some tests failed"
            return 1
        fi
        log_success "All tests completed"
    fi
    
    return $exit_code
}

# Initialize - parse arguments first
parse_args "$@"

# Run in Vagrant if requested
if [ "$VAGRANT_MODE" = true ]; then
    # Detect OS first to determine which VM to use
    detect_os
    run_in_vagrant
    exit $?
fi

# Run main function
main

