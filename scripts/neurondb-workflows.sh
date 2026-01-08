#!/bin/bash
#
# NeuronDB Workflows Script
# Self-sufficient script for all workflow operations: release, sync, git operations
#
# Usage:
#   ./neurondb-workflows.sh COMMAND [OPTIONS]
#
# Commands:
#   release        Create a new release
#   sync           Sync version branches
#   update-refs    Update markdown file references
#   pull           Safe git pull with rebase
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
    echo -e "${BLUE}║${NC}  ${BOLD}NeuronDB Workflows${NC}                                  ${BLUE}║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

#=========================================================================
# HELP FUNCTION
#=========================================================================

show_help() {
    cat << EOF
${BOLD}NeuronDB Workflows${NC}

${BOLD}Usage:${NC}
    ${SCRIPT_NAME} COMMAND [OPTIONS]

${BOLD}Commands:${NC}
    release        Create a new release
    sync           Sync version branches (REL1_STABLE and main)
    update-refs    Update markdown file references after renames
    pull           Safe git pull with rebase

${BOLD}Release Options:${NC}
    --version VERSION    Version number (required, e.g., 1.0.0)
    --dry-run           Preview changes without applying

${BOLD}Sync Options:${NC}
    --from-branch BRANCH    Source branch (default: main)
    --to-branch BRANCH      Target branch (default: REL1_STABLE)
    --from-version VER      Source version (default: 2.0.0)
    --to-version VER        Target version (default: 1.0.0)

${BOLD}Global Options:${NC}
    --dry-run           Preview changes without applying
    -h, --help          Show this help message
    -v, --verbose       Enable verbose output
    -V, --version       Show version information

${BOLD}Examples:${NC}
    # Create a release
    ${SCRIPT_NAME} release --version 2.0.0

    # Sync branches
    ${SCRIPT_NAME} sync

    # Update markdown references
    ${SCRIPT_NAME} update-refs

    # Safe git pull
    ${SCRIPT_NAME} pull

EOF
}

#=========================================================================
# RELEASE COMMAND
#=========================================================================

release_command() {
    shift
    
    local version=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --version)
                version="$2"
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
    
    if [[ -z "$version" ]]; then
        log_error "Version is required (--version)"
        show_help
        exit 1
    fi
    
    if ! [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?$ ]]; then
        log_error "Invalid version format. Use semver (e.g., 1.0.0)"
        exit 1
    fi
    
    print_header
    log_info "Creating release: $version"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would update version to $version"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    
    # Update version in key files
    local files=(
        "env.example"
        "package.json"
        "NeuronDB/neurondb.control"
        "NeuronAgent/go.mod"
        "NeuronDesktop/frontend/package.json"
        "NeuronMCP/go.mod"
    )
    
    for file in "${files[@]}"; do
        if [[ -f "$file" ]]; then
            [[ "$OSTYPE" == "darwin"* ]] && sed -i '' "s/version.*=.*\".*\"/version = \"$version\"/g" "$file" 2>/dev/null || true
            [[ "$OSTYPE" != "darwin"* ]] && sed -i "s/version.*=.*\".*\"/version = \"$version\"/g" "$file" 2>/dev/null || true
            log_info "Updated: $file"
        fi
    done
    
    log_success "Release $version created"
}

#=========================================================================
# SYNC COMMAND
#=========================================================================

sync_command() {
    shift
    
    local from_branch="main"
    local to_branch="REL1_STABLE"
    local from_version="2.0.0"
    local to_version="1.0.0"
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --from-branch)
                from_branch="$2"
                shift 2
                ;;
            --to-branch)
                to_branch="$2"
                shift 2
                ;;
            --from-version)
                from_version="$2"
                shift 2
                ;;
            --to-version)
                to_version="$2"
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
    log_info "Syncing $from_branch -> $to_branch (version: $from_version -> $to_version)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would sync branches"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    
    # Check if we're on the right branch
    local current_branch=$(git branch --show-current)
    if [[ "$current_branch" != "$from_branch" ]]; then
        log_warning "Not on $from_branch, switching..."
        git checkout "$from_branch"
    fi
    
    # Fetch latest
    git fetch origin
    
    # Check if target branch exists
    if git show-ref --verify --quiet "refs/heads/$to_branch"; then
        git checkout "$to_branch"
        git merge "$from_branch" --no-edit || {
            log_error "Merge conflict detected. Please resolve manually."
            exit 1
        }
    else
        log_info "Creating $to_branch from $from_branch..."
        git checkout -b "$to_branch" "origin/$from_branch" || git checkout -b "$to_branch" "$from_branch"
    fi
    
    # Replace versions
    log_info "Replacing $from_version with $to_version..."
    find . -type f \( -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "*.md" -o -name "*.sh" \) \
        ! -path "./.git/*" \
        -exec sed -i '' "s/$from_version/$to_version/g" {} \; 2>/dev/null || \
        find . -type f \( -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "*.md" -o -name "*.sh" \) \
        ! -path "./.git/*" \
        -exec sed -i "s/$from_version/$to_version/g" {} \;
    
    log_success "Branches synced"
}

#=========================================================================
# UPDATE-REFS COMMAND
#=========================================================================

update_refs_command() {
    shift
    
    print_header
    log_info "Updating markdown file references..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would update references"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    
    # Get list of renamed files from git
    git status --short | grep -E "^R" | while read -r status oldfile newfile; do
        local oldname=$(basename "$oldfile")
        local newname=$(basename "$newfile")
        
        if [[ "$oldname" != "$newname" ]]; then
            log_info "Updating references: $oldname -> $newname"
            
            find . -type f \( -name "*.md" -o -name "*.txt" -o -name "*.sh" -o -name "*.go" -o -name "*.py" -o -name "*.ts" -o -name "*.tsx" -o -name "*.js" -o -name "*.json" \) \
                ! -path "./.git/*" \
                -exec sed -i '' "s|$oldname|$newname|g" {} \; 2>/dev/null || \
                find . -type f \( -name "*.md" -o -name "*.txt" -o -name "*.sh" -o -name "*.go" -o -name "*.py" -o -name "*.ts" -o -name "*.tsx" -o -name "*.js" -o -name "*.json" \) \
                ! -path "./.git/*" \
                -exec sed -i "s|$oldname|$newname|g" {} \;
        fi
    done
    
    log_success "References updated"
}

#=========================================================================
# PULL COMMAND
#=========================================================================

pull_command() {
    shift
    
    print_header
    log_info "Performing safe git pull..."
    
    cd "$PROJECT_ROOT"
    
    # Check for uncommitted changes
    local stashed=false
    if ! git diff-index --quiet HEAD --; then
        log_warning "Uncommitted changes detected. Stashing..."
        git stash push -m "Auto-stash before pull $(date +%Y-%m-%d_%H:%M:%S)"
        stashed=true
    fi
    
    # Fetch and rebase
    git fetch origin
    local current_branch=$(git branch --show-current)
    
    log_info "Rebasing on origin/$current_branch..."
    if git rebase "origin/$current_branch"; then
        log_success "Rebase completed"
    else
        log_error "Rebase had conflicts. Resolve them and run: git rebase --continue"
        if [[ "$stashed" == "true" ]]; then
            log_warning "Stashed changes available with: git stash pop"
        fi
        exit 1
    fi
    
    # Restore stashed changes
    if [[ "$stashed" == "true" ]]; then
        log_info "Restoring stashed changes..."
        git stash pop || log_warning "Could not automatically restore stashed changes. Run: git stash pop"
    fi
    
    log_success "Pull completed"
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
            --version)
                # Handled by release command
                shift 2
                ;;
            --from-branch|--to-branch|--from-version|--to-version)
                # Handled by sync command
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
        release)
            release_command "$@"
            ;;
        sync)
            sync_command "$@"
            ;;
        update-refs)
            update_refs_command "$@"
            ;;
        pull)
            pull_command "$@"
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

main "$@"

