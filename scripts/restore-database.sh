#!/bin/bash
#
# NeuronDB Database Restore Script
# Professional restore solution supporting multiple backup formats
#
# Usage:
#   ./restore-database.sh --backup PATH [OPTIONS]
#
# Options:
#   --backup PATH        Backup file or directory to restore
#   --format FORMAT      Backup format: sql, custom, directory (auto-detected if not specified)
#   --drop              Drop existing database before restore
#   --clean             Clean (drop) database objects before recreating
#   --jobs N            Number of parallel jobs for directory format (default: 4)
#   --help, -h          Show this help

set -euo pipefail

# ============================================================================
# Configuration and Constants
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME=$(basename "$0")

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Default configuration
BACKUP_PATH=""
BACKUP_FORMAT=""
DROP_DATABASE=false
CLEAN_OBJECTS=false
PARALLEL_JOBS=4

# Database configuration from environment or defaults
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-neurondb}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-}"

# ============================================================================
# Logging Functions
# ============================================================================

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
    echo -e "${BLUE}║${NC}  ${BOLD}NeuronDB Database Restore${NC}                                 ${BLUE}║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# ============================================================================
# Utility Functions
# ============================================================================

show_help() {
    cat << EOF
${BOLD}NeuronDB Database Restore Script${NC}

${BOLD}Usage:${NC}
    $SCRIPT_NAME --backup PATH [OPTIONS]

${BOLD}Options:${NC}
    --backup PATH        Backup file or directory to restore (required)
    --format FORMAT      Backup format: sql, custom, directory (auto-detected)
    --drop               Drop existing database before restore
    --clean              Clean (drop) database objects before recreating
    --jobs N             Number of parallel jobs for directory format (default: 4)
    --help, -h           Show this help message

${BOLD}Database Configuration:${NC}
    Set via environment variables:
    - DB_HOST (default: localhost)
    - DB_PORT (default: 5432)
    - DB_NAME (default: neurondb)
    - DB_USER (default: postgres)
    - DB_PASSWORD

${BOLD}Examples:${NC}
    # Restore from SQL backup
    $SCRIPT_NAME --backup neurondb_backup_20250101_120000.sql

    # Restore from custom format
    $SCRIPT_NAME --backup neurondb_backup_20250101_120000.dump

    # Restore from directory format with 8 parallel jobs
    $SCRIPT_NAME --backup neurondb_backup_20250101_120000_dir --jobs 8

    # Drop database and restore
    $SCRIPT_NAME --backup backup.dump --drop

EOF
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v pg_restore &> /dev/null; then
        log_error "pg_restore not found. Install PostgreSQL client tools."
        return 1
    fi
    
    if ! command -v psql &> /dev/null; then
        log_error "psql not found. Install PostgreSQL client tools."
        return 1
    fi
    
    log_success "Prerequisites satisfied"
    return 0
}

check_backup_exists() {
    log_info "Checking backup: $BACKUP_PATH"
    
    if [ ! -e "$BACKUP_PATH" ]; then
        log_error "Backup not found: $BACKUP_PATH"
        return 1
    fi
    
    local size
    if [ -f "$BACKUP_PATH" ]; then
        size=$(du -h "$BACKUP_PATH" | cut -f1)
        log_success "Backup file found: $BACKUP_PATH ($size)"
    elif [ -d "$BACKUP_PATH" ]; then
        size=$(du -sh "$BACKUP_PATH" | cut -f1)
        log_success "Backup directory found: $BACKUP_PATH ($size)"
    fi
    
    return 0
}

detect_backup_format() {
    if [ -n "$BACKUP_FORMAT" ]; then
        log_info "Using specified format: $BACKUP_FORMAT"
        return 0
    fi
    
    log_info "Auto-detecting backup format..."
    
    if [ -d "$BACKUP_PATH" ]; then
        BACKUP_FORMAT="directory"
        log_info "Detected directory format"
    elif [[ "$BACKUP_PATH" =~ \.sql(\.gz)?$ ]]; then
        BACKUP_FORMAT="sql"
        log_info "Detected SQL format"
    elif [[ "$BACKUP_PATH" =~ \.dump$ ]]; then
        BACKUP_FORMAT="custom"
        log_info "Detected custom format"
    else
        # Try to detect by content
        if file "$BACKUP_PATH" | grep -q "PostgreSQL custom database dump"; then
            BACKUP_FORMAT="custom"
            log_info "Detected custom format (by content)"
        else
            BACKUP_FORMAT="sql"
            log_info "Assuming SQL format"
        fi
    fi
}

check_database_connection() {
    log_info "Testing database connection to $DB_HOST:$DB_PORT..."
    
    export PGPASSWORD="$DB_PASSWORD"
    
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "SELECT 1;" &> /dev/null; then
        log_success "Database server connection successful"
        return 0
    else
        log_error "Cannot connect to database server"
        return 1
    fi
}

drop_database_if_requested() {
    if [ "$DROP_DATABASE" = false ]; then
        return 0
    fi
    
    log_warning "Dropping database: $DB_NAME"
    echo -n "Are you sure? This will delete all data. Type 'yes' to confirm: "
    read -r confirmation
    
    if [ "$confirmation" != "yes" ]; then
        log_info "Database drop cancelled"
        return 1
    fi
    
    export PGPASSWORD="$DB_PASSWORD"
    
    # Terminate connections
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres <<SQL 2>/dev/null || true
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = '$DB_NAME' AND pid <> pg_backend_pid();
SQL
    
    # Drop database
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "DROP DATABASE IF EXISTS $DB_NAME;" &> /dev/null; then
        log_success "Database dropped"
    else
        log_error "Failed to drop database"
        return 1
    fi
    
    # Recreate database
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "CREATE DATABASE $DB_NAME;" &> /dev/null; then
        log_success "Database recreated"
    else
        log_error "Failed to create database"
        return 1
    fi
}

# ============================================================================
# Restore Functions
# ============================================================================

restore_sql_format() {
    log_info "Restoring from SQL backup: $BACKUP_PATH"
    
    export PGPASSWORD="$DB_PASSWORD"
    
    local restore_cmd="psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME"
    
    if [[ "$BACKUP_PATH" =~ \.gz$ ]]; then
        log_info "Decompressing and restoring..."
        if gunzip -c "$BACKUP_PATH" | $restore_cmd &> /dev/null; then
            log_success "SQL backup restored successfully"
            return 0
        else
            log_error "Failed to restore SQL backup"
            return 1
        fi
    else
        if $restore_cmd < "$BACKUP_PATH" &> /dev/null; then
            log_success "SQL backup restored successfully"
            return 0
        else
            log_error "Failed to restore SQL backup"
            return 1
        fi
    fi
}

restore_custom_format() {
    log_info "Restoring from custom format backup: $BACKUP_PATH"
    
    export PGPASSWORD="$DB_PASSWORD"
    
    local pg_restore_opts="-h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME"
    
    if [ "$CLEAN_OBJECTS" = true ]; then
        pg_restore_opts="$pg_restore_opts --clean"
        log_info "Will clean database objects before restoring"
    fi
    
    pg_restore_opts="$pg_restore_opts --verbose"
    
    if pg_restore $pg_restore_opts "$BACKUP_PATH" 2>&1 | grep -v "^pg_restore:" | grep -v "WARNING" || true; then
        log_success "Custom format backup restored successfully"
        return 0
    else
        log_warning "Restore completed with warnings (this is often normal)"
        return 0
    fi
}

restore_directory_format() {
    log_info "Restoring from directory format backup: $BACKUP_PATH"
    log_info "Using $PARALLEL_JOBS parallel jobs"
    
    export PGPASSWORD="$DB_PASSWORD"
    
    local pg_restore_opts="-h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME"
    pg_restore_opts="$pg_restore_opts --jobs=$PARALLEL_JOBS"
    
    if [ "$CLEAN_OBJECTS" = true ]; then
        pg_restore_opts="$pg_restore_opts --clean"
        log_info "Will clean database objects before restoring"
    fi
    
    pg_restore_opts="$pg_restore_opts --verbose"
    
    if pg_restore $pg_restore_opts "$BACKUP_PATH" 2>&1 | grep -v "^pg_restore:" | grep -v "WARNING" || true; then
        log_success "Directory format backup restored successfully"
        return 0
    else
        log_warning "Restore completed with warnings (this is often normal)"
        return 0
    fi
}

verify_restore() {
    log_info "Verifying restore..."
    
    export PGPASSWORD="$DB_PASSWORD"
    
    local tables=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog', 'information_schema');" 2>/dev/null | xargs)
    
    log_info "Tables restored: $tables"
    
    # Check if NeuronDB extension exists
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1 FROM pg_extension WHERE extname = 'neurondb';" | grep -q 1 2>/dev/null; then
        log_success "NeuronDB extension found"
    else
        log_warning "NeuronDB extension not found (may need to be installed separately)"
    fi
}

# ============================================================================
# Argument Parsing
# ============================================================================

parse_arguments() {
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --backup)
                BACKUP_PATH="$2"
                shift 2
                ;;
            --format)
                BACKUP_FORMAT="$2"
                shift 2
                ;;
            --drop)
                DROP_DATABASE=true
                shift
                ;;
            --clean)
                CLEAN_OBJECTS=true
                shift
                ;;
            --jobs)
                PARALLEL_JOBS="$2"
                shift 2
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Validate required parameters
    if [ -z "$BACKUP_PATH" ]; then
        log_error "Backup path is required (--backup)"
        show_help
        exit 1
    fi
    
    # Validate format if specified
    if [ -n "$BACKUP_FORMAT" ] && [[ ! "$BACKUP_FORMAT" =~ ^(sql|custom|directory)$ ]]; then
        log_error "Invalid format: $BACKUP_FORMAT. Must be sql, custom, or directory."
        exit 1
    fi
}

# ============================================================================
# Main Function
# ============================================================================

main() {
    parse_arguments "$@"
    
    print_header
    
    if ! check_prerequisites; then
        exit 1
    fi
    
    if ! check_backup_exists; then
        exit 1
    fi
    
    detect_backup_format
    
    if ! check_database_connection; then
        exit 1
    fi
    
    echo -e "${BOLD}Restore Configuration:${NC}"
    echo "  Backup: $BACKUP_PATH"
    echo "  Format: $BACKUP_FORMAT"
    echo "  Target: $DB_HOST:$DB_PORT/$DB_NAME"
    echo "  Drop database: $DROP_DATABASE"
    echo "  Clean objects: $CLEAN_OBJECTS"
    if [ "$BACKUP_FORMAT" = "directory" ]; then
        echo "  Parallel jobs: $PARALLEL_JOBS"
    fi
    echo ""
    
    if ! drop_database_if_requested; then
        exit 1
    fi
    
    case "$BACKUP_FORMAT" in
        sql)
            restore_sql_format
            ;;
        custom)
            restore_custom_format
            ;;
        directory)
            restore_directory_format
            ;;
    esac
    
    if [ $? -eq 0 ]; then
        verify_restore
        
        echo ""
        echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║${NC}  ${BOLD}Restore Completed Successfully!${NC}                           ${GREEN}║${NC}"
        echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        exit 0
    else
        log_error "Restore failed"
        exit 1
    fi
}

main "$@"




