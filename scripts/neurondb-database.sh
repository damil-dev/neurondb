#!/bin/bash
#
# NeuronDB Database Management Script
# Self-sufficient script for all database operations: backup, restore, setup, maintenance
#
# Usage:
#   ./neurondb-database.sh COMMAND [OPTIONS]
#
# Commands:
#   backup          Create database backup
#   restore         Restore database from backup
#   setup           Setup database and extensions
#   status          Check database status and info
#   vacuum          Run VACUUM and ANALYZE
#   list-backups    List available backups
#   verify          Verify database integrity

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
OUTPUT_DIR="${PROJECT_ROOT}/backups"
RETENTION_DAYS=30
BACKUP_FORMAT="custom"
COMPRESS=false
DROP_DATABASE=false
CLEAN_OBJECTS=false
PARALLEL_JOBS=4

# Database configuration from environment or defaults
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-neurondb}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-}"

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
    echo -e "${BLUE}║${NC}  ${BOLD}NeuronDB Database Management${NC}                            ${BLUE}║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

#=========================================================================
# HELP FUNCTION
#=========================================================================

show_help() {
    cat << EOF
${BOLD}NeuronDB Database Management${NC}

${BOLD}Usage:${NC}
    ${SCRIPT_NAME} COMMAND [OPTIONS]

${BOLD}Commands:${NC}
    backup          Create database backup (sql, custom, or directory format)
    restore         Restore database from backup
    setup           Setup database and install extensions
    status          Check database status and information
    vacuum          Run VACUUM and ANALYZE for maintenance
    list-backups    List available backups
    verify          Verify database integrity and extensions

${BOLD}Backup Options:${NC}
    --format FORMAT       Backup format: sql, custom, directory (default: custom)
    --output DIR          Output directory (default: ./backups)
    --compress            Compress SQL backups
    --retention DAYS      Keep backups for N days (default: 30)

${BOLD}Restore Options:${NC}
    --backup PATH         Backup file or directory to restore (required)
    --format FORMAT       Backup format (auto-detected if not specified)
    --drop                Drop existing database before restore
    --clean               Clean database objects before recreating
    --jobs N              Parallel jobs for directory format (default: 4)

${BOLD}Global Options:${NC}
    --host HOST           Database host (default: localhost)
    --port PORT           Database port (default: 5432)
    --database NAME       Database name (default: neurondb)
    --user USER           Database user (default: postgres)
    --password PASSWORD   Database password (or use DB_PASSWORD env var)
    --dry-run             Preview changes without applying
    -h, --help            Show this help message
    -v, --verbose         Enable verbose output
    -V, --version         Show version information

${BOLD}Database Configuration:${NC}
    Can be set via environment variables:
    - DB_HOST (default: localhost)
    - DB_PORT (default: 5432)
    - DB_NAME (default: neurondb)
    - DB_USER (default: postgres)
    - DB_PASSWORD

${BOLD}Examples:${NC}
    # Create backup
    ${SCRIPT_NAME} backup --format custom

    # Restore from backup
    ${SCRIPT_NAME} restore --backup backup.dump

    # Setup database
    ${SCRIPT_NAME} setup

    # Check status
    ${SCRIPT_NAME} status

    # List backups
    ${SCRIPT_NAME} list-backups

    # Run maintenance
    ${SCRIPT_NAME} vacuum

EOF
}

#=========================================================================
# UTILITY FUNCTIONS
#=========================================================================

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v pg_dump &> /dev/null; then
        log_error "pg_dump not found. Install PostgreSQL client tools."
        return 1
    fi
    
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

check_database_connection() {
    local dbname="${1:-$DB_NAME}"
    log_info "Testing database connection to $DB_HOST:$DB_PORT/$dbname..."
    
    export PGPASSWORD="$DB_PASSWORD"
    
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$dbname" -c "SELECT 1;" &> /dev/null; then
        log_success "Database connection successful"
        return 0
    else
        log_error "Cannot connect to database"
        return 1
    fi
}

get_database_info() {
    export PGPASSWORD="$DB_PASSWORD"
    
    local db_size=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -A -c "SELECT pg_size_pretty(pg_database_size('$DB_NAME'));" 2>/dev/null | xargs || echo "unknown")
    local tables=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -A -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog', 'information_schema');" 2>/dev/null | xargs || echo "0")
    local extensions=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -A -c "SELECT COUNT(*) FROM pg_extension;" 2>/dev/null | xargs || echo "0")
    local version=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -A -c "SELECT version();" 2>/dev/null | head -1 || echo "unknown")
    
    echo -e "${BOLD}Database Information:${NC}"
    echo "  Name: $DB_NAME"
    echo "  Host: $DB_HOST:$DB_PORT"
    echo "  Size: $db_size"
    echo "  Tables: $tables"
    echo "  Extensions: $extensions"
    echo "  Version: $version"
    echo ""
}

#=========================================================================
# BACKUP COMMAND
#=========================================================================

backup_command() {
    shift
    
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    print_header
    
    if ! check_prerequisites; then
        exit 1
    fi
    
    if ! check_database_connection; then
        exit 1
    fi
    
    get_database_info
    
    mkdir -p "$OUTPUT_DIR"
    
    echo -e "${BOLD}Backup Configuration:${NC}"
    echo "  Format: $BACKUP_FORMAT"
    echo "  Output: $OUTPUT_DIR"
    echo "  Compress: $COMPRESS"
    echo "  Retention: $RETENTION_DAYS days"
    echo ""
    
    export PGPASSWORD="$DB_PASSWORD"
    local backup_result=""
    
    case "$BACKUP_FORMAT" in
        sql)
            local backup_file="$OUTPUT_DIR/neurondb_backup_${timestamp}.sql"
            log_info "Creating SQL backup: $backup_file"
            
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_info "[DRY RUN] Would create: $backup_file"
                backup_result="$backup_file"
            else
                pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" \
                        --format=plain \
                        --verbose \
                        --file="$backup_file" \
                        "$DB_NAME" 2>&1 | grep -v "^pg_dump:" || true
                
                if [[ -f "$backup_file" ]]; then
                    if [[ "$COMPRESS" == "true" ]]; then
                        log_info "Compressing backup..."
                        gzip -f "$backup_file"
                        backup_file="${backup_file}.gz"
                    fi
                    local size=$(du -h "$backup_file" | cut -f1)
                    log_success "SQL backup created: $backup_file ($size)"
                    backup_result="$backup_file"
                else
                    log_error "SQL backup failed"
                    exit 1
                fi
            fi
            ;;
        custom)
            local backup_file="$OUTPUT_DIR/neurondb_backup_${timestamp}.dump"
            log_info "Creating custom format backup: $backup_file"
            
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_info "[DRY RUN] Would create: $backup_file"
                backup_result="$backup_file"
            else
                pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" \
                        --format=custom \
                        --compress=9 \
                        --verbose \
                        --file="$backup_file" \
                        "$DB_NAME" 2>&1 | grep -v "^pg_dump:" || true
                
                if [[ -f "$backup_file" ]]; then
                    local size=$(du -h "$backup_file" | cut -f1)
                    log_success "Custom format backup created: $backup_file ($size)"
                    backup_result="$backup_file"
                else
                    log_error "Custom format backup failed"
                    exit 1
                fi
            fi
            ;;
        directory)
            local backup_dir="$OUTPUT_DIR/neurondb_backup_${timestamp}_dir"
            log_info "Creating directory format backup: $backup_dir"
            
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_info "[DRY RUN] Would create: $backup_dir"
                backup_result="$backup_dir"
            else
                pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" \
                        --format=directory \
                        --jobs=4 \
                        --compress=9 \
                        --verbose \
                        --file="$backup_dir" \
                        "$DB_NAME" 2>&1 | grep -v "^pg_dump:" || true
                
                if [[ -d "$backup_dir" ]]; then
                    local size=$(du -sh "$backup_dir" | cut -f1)
                    log_success "Directory format backup created: $backup_dir ($size)"
                    backup_result="$backup_dir"
                else
                    log_error "Directory format backup failed"
                    exit 1
                fi
            fi
            ;;
        *)
            log_error "Invalid format: $BACKUP_FORMAT"
            exit 1
            ;;
    esac
    
    if [[ "${DRY_RUN}" != "true" ]]; then
        cleanup_old_backups
    fi
    
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC}  ${BOLD}Backup Completed Successfully!${NC}                            ${GREEN}║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BOLD}Backup Details:${NC}"
    echo "  Location: $backup_result"
    if [[ -f "$backup_result" ]]; then
        echo "  Size: $(du -h "$backup_result" | cut -f1)"
    elif [[ -d "$backup_result" ]]; then
        echo "  Size: $(du -sh "$backup_result" | cut -f1)"
    fi
    echo "  Timestamp: $timestamp"
    echo ""
}

cleanup_old_backups() {
    log_info "Cleaning up backups older than $RETENTION_DAYS days..."
    
    local deleted_count=0
    
    while IFS= read -r -d '' file; do
        log_info "Deleting old backup: $(basename "$file")"
        rm -rf "$file"
        ((deleted_count++))
    done < <(find "$OUTPUT_DIR" -type f \( -name "neurondb_backup_*.sql*" -o -name "neurondb_backup_*.dump" \) -mtime +"$RETENTION_DAYS" -print0 2>/dev/null)
    
    while IFS= read -r -d '' dir; do
        log_info "Deleting old backup directory: $(basename "$dir")"
        rm -rf "$dir"
        ((deleted_count++))
    done < <(find "$OUTPUT_DIR" -type d -name "neurondb_backup_*_dir" -mtime +"$RETENTION_DAYS" -print0 2>/dev/null)
    
    if [[ $deleted_count -gt 0 ]]; then
        log_success "Deleted $deleted_count old backup(s)"
    else
        log_info "No old backups to delete"
    fi
}

#=========================================================================
# RESTORE COMMAND
#=========================================================================

restore_command() {
    shift
    
    local backup_path=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --backup)
                backup_path="$2"
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done
    
    if [[ -z "$backup_path" ]]; then
        log_error "Backup path required (--backup)"
        exit 1
    fi
    
    print_header
    
    if ! check_prerequisites; then
        exit 1
    fi
    
    if [[ ! -e "$backup_path" ]]; then
        log_error "Backup not found: $backup_path"
        exit 1
    fi
    
    local backup_format="$BACKUP_FORMAT"
    if [[ -z "$backup_format" ]]; then
        if [[ -d "$backup_path" ]]; then
            backup_format="directory"
        elif [[ "$backup_path" =~ \.sql(\.gz)?$ ]]; then
            backup_format="sql"
        elif [[ "$backup_path" =~ \.dump$ ]]; then
            backup_format="custom"
        else
            if file "$backup_path" 2>/dev/null | grep -q "PostgreSQL custom database dump"; then
                backup_format="custom"
            else
                backup_format="sql"
            fi
        fi
    fi
    
    log_info "Detected backup format: $backup_format"
    
    if ! check_database_connection "postgres"; then
        exit 1
    fi
    
    echo -e "${BOLD}Restore Configuration:${NC}"
    echo "  Backup: $backup_path"
    echo "  Format: $backup_format"
    echo "  Target: $DB_HOST:$DB_PORT/$DB_NAME"
    echo "  Drop database: $DROP_DATABASE"
    echo "  Clean objects: $CLEAN_OBJECTS"
    if [[ "$backup_format" == "directory" ]]; then
        echo "  Parallel jobs: $PARALLEL_JOBS"
    fi
    echo ""
    
    if [[ "$DROP_DATABASE" == "true" ]]; then
        log_warning "Dropping database: $DB_NAME"
        if [[ "${DRY_RUN}" != "true" ]]; then
            echo -n "Are you sure? Type 'yes' to confirm: "
            read -r confirmation
            if [[ "$confirmation" != "yes" ]]; then
                log_info "Database drop cancelled"
                exit 1
            fi
            
            export PGPASSWORD="$DB_PASSWORD"
            psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres <<SQL 2>/dev/null || true
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = '$DB_NAME' AND pid <> pg_backend_pid();
SQL
            psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "DROP DATABASE IF EXISTS $DB_NAME;" &> /dev/null
            psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "CREATE DATABASE $DB_NAME;" &> /dev/null
            log_success "Database recreated"
        fi
    fi
    
    export PGPASSWORD="$DB_PASSWORD"
    
    case "$backup_format" in
        sql)
            log_info "Restoring from SQL backup..."
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_info "[DRY RUN] Would restore from: $backup_path"
            else
                if [[ "$backup_path" =~ \.gz$ ]]; then
                    gunzip -c "$backup_path" | psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" &> /dev/null
                else
                    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" < "$backup_path" &> /dev/null
                fi
                log_success "SQL backup restored"
            fi
            ;;
        custom)
            log_info "Restoring from custom format backup..."
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_info "[DRY RUN] Would restore from: $backup_path"
            else
                local opts="-h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME"
                [[ "$CLEAN_OBJECTS" == "true" ]] && opts="$opts --clean"
                pg_restore $opts "$backup_path" 2>&1 | grep -v "^pg_restore:" | grep -v "WARNING" || true
                log_success "Custom format backup restored"
            fi
            ;;
        directory)
            log_info "Restoring from directory format backup..."
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_info "[DRY RUN] Would restore from: $backup_path"
            else
                local opts="-h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME --jobs=$PARALLEL_JOBS"
                [[ "$CLEAN_OBJECTS" == "true" ]] && opts="$opts --clean"
                pg_restore $opts "$backup_path" 2>&1 | grep -v "^pg_restore:" | grep -v "WARNING" || true
                log_success "Directory format backup restored"
            fi
            ;;
    esac
    
    if [[ "${DRY_RUN}" != "true" ]]; then
        verify_restore
    fi
    
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC}  ${BOLD}Restore Completed Successfully!${NC}                           ${GREEN}║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

verify_restore() {
    log_info "Verifying restore..."
    
    export PGPASSWORD="$DB_PASSWORD"
    
    local tables=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -A -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog', 'information_schema');" 2>/dev/null | xargs || echo "0")
    log_info "Tables restored: $tables"
    
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1 FROM pg_extension WHERE extname = 'neurondb';" 2>/dev/null | grep -q 1; then
        log_success "NeuronDB extension found"
    else
        log_warning "NeuronDB extension not found (may need to be installed separately)"
    fi
}

#=========================================================================
# SETUP COMMAND
#=========================================================================

setup_command() {
    shift
    
    print_header
    
    if ! check_prerequisites; then
        exit 1
    fi
    
    if ! check_database_connection "postgres"; then
        exit 1
    fi
    
    export PGPASSWORD="$DB_PASSWORD"
    
    log_info "Setting up database: $DB_NAME"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would create database and install extensions"
        return 0
    fi
    
    # Create database if it doesn't exist
    if ! psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME';" | grep -q 1; then
        log_info "Creating database: $DB_NAME"
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "CREATE DATABASE $DB_NAME;" &> /dev/null
        log_success "Database created"
    else
        log_info "Database already exists"
    fi
    
    # Install extensions
    log_info "Installing extensions..."
    
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" <<SQL 2>/dev/null || true
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS neurondb;
SQL
    
    log_success "Extensions installed"
    
    get_database_info
    
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC}  ${BOLD}Database Setup Completed!${NC}                                ${GREEN}║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

#=========================================================================
# STATUS COMMAND
#=========================================================================

status_command() {
    shift
    
    print_header
    
    if ! check_prerequisites; then
        exit 1
    fi
    
    if ! check_database_connection; then
        exit 1
    fi
    
    get_database_info
    
    export PGPASSWORD="$DB_PASSWORD"
    
    echo -e "${BOLD}Extensions:${NC}"
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "\dx" 2>/dev/null || true
    echo ""
    
    echo -e "${BOLD}Connection Info:${NC}"
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT current_database(), current_user, inet_server_addr(), inet_server_port();" 2>/dev/null || true
    echo ""
}

#=========================================================================
# VACUUM COMMAND
#=========================================================================

vacuum_command() {
    shift
    
    print_header
    
    if ! check_prerequisites; then
        exit 1
    fi
    
    if ! check_database_connection; then
        exit 1
    fi
    
    export PGPASSWORD="$DB_PASSWORD"
    
    log_info "Running VACUUM and ANALYZE..."
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would run VACUUM ANALYZE"
        return 0
    fi
    
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "VACUUM ANALYZE;" &> /dev/null
    
    log_success "VACUUM and ANALYZE completed"
    echo ""
}

#=========================================================================
# LIST-BACKUPS COMMAND
#=========================================================================

list_backups_command() {
    shift
    
    print_header
    
    if [[ ! -d "$OUTPUT_DIR" ]]; then
        log_warning "Backup directory does not exist: $OUTPUT_DIR"
        exit 0
    fi
    
    log_info "Available backups in: $OUTPUT_DIR"
    echo ""
    
    echo -e "${BOLD}SQL Backups:${NC}"
    find "$OUTPUT_DIR" -name "neurondb_backup_*.sql*" -type f -exec ls -lh {} \; 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  None"
    echo ""
    
    echo -e "${BOLD}Custom Format Backups:${NC}"
    find "$OUTPUT_DIR" -name "neurondb_backup_*.dump" -type f -exec ls -lh {} \; 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  None"
    echo ""
    
    echo -e "${BOLD}Directory Format Backups:${NC}"
    find "$OUTPUT_DIR" -name "neurondb_backup_*_dir" -type d -exec du -sh {} \; 2>/dev/null | awk '{print "  " $2 " (" $1 ")"}' || echo "  None"
    echo ""
}

#=========================================================================
# VERIFY COMMAND
#=========================================================================

verify_command() {
    shift
    
    print_header
    
    if ! check_prerequisites; then
        exit 1
    fi
    
    if ! check_database_connection; then
        exit 1
    fi
    
    export PGPASSWORD="$DB_PASSWORD"
    
    log_info "Verifying database integrity..."
    
    # Check database size
    local db_size=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -A -c "SELECT pg_size_pretty(pg_database_size('$DB_NAME'));" 2>/dev/null | xargs)
    log_info "Database size: $db_size"
    
    # Check for NeuronDB extension
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1 FROM pg_extension WHERE extname = 'neurondb';" 2>/dev/null | grep -q 1; then
        local version=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -A -c "SELECT extversion FROM pg_extension WHERE extname = 'neurondb';" 2>/dev/null | xargs)
        log_success "NeuronDB extension installed (version: $version)"
    else
        log_warning "NeuronDB extension not found"
    fi
    
    # Check for vector extension
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1 FROM pg_extension WHERE extname = 'vector';" 2>/dev/null | grep -q 1; then
        log_success "Vector extension installed"
    else
        log_warning "Vector extension not found"
    fi
    
    # Check table count
    local tables=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -A -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog', 'information_schema');" 2>/dev/null | xargs)
    log_info "User tables: $tables"
    
    echo ""
    log_success "Verification completed"
    echo ""
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
            --format)
                BACKUP_FORMAT="$2"
                shift 2
                ;;
            --output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --compress)
                COMPRESS=true
                shift
                ;;
            --retention)
                RETENTION_DAYS="$2"
                shift 2
                ;;
            --backup)
                # Handled by restore command
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
            --host)
                DB_HOST="$2"
                shift 2
                ;;
            --port)
                DB_PORT="$2"
                shift 2
                ;;
            --database)
                DB_NAME="$2"
                shift 2
                ;;
            --user)
                DB_USER="$2"
                shift 2
                ;;
            --password)
                DB_PASSWORD="$2"
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
                # Remaining arguments passed to command
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
    
    case "${COMMAND}" in
        backup)
            backup_command "$@"
            ;;
        restore)
            restore_command "$@"
            ;;
        setup)
            setup_command "$@"
            ;;
        status)
            status_command "$@"
            ;;
        vacuum)
            vacuum_command "$@"
            ;;
        list-backups)
            list_backups_command "$@"
            ;;
        verify)
            verify_command "$@"
            ;;
        *)
            log_error "Unknown command: ${COMMAND}"
            show_help
            exit 1
            ;;
    esac
}

main "$@"

