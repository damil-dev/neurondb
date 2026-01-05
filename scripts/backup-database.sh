#!/bin/bash
#
# NeuronDB Database Backup Script
# Professional backup solution with multiple formats and retention policies
#
# Usage:
#   ./backup-database.sh [OPTIONS]
#
# Options:
#   --format [sql|custom|directory]  Backup format (default: custom)
#   --output PATH                     Output directory (default: ./backups)
#   --compress                        Compress backup
#   --retention DAYS                  Keep backups for N days (default: 30)
#   --help                           Show this help

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
BACKUP_FORMAT="custom"
OUTPUT_DIR="./backups"
COMPRESS=false
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

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
    echo -e "${BLUE}║${NC}  ${BOLD}NeuronDB Database Backup${NC}                                  ${BLUE}║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# ============================================================================
# Utility Functions
# ============================================================================

show_help() {
    cat << EOF
${BOLD}NeuronDB Database Backup Script${NC}

${BOLD}Usage:${NC}
    $SCRIPT_NAME [OPTIONS]

${BOLD}Options:${NC}
    --format FORMAT       Backup format: sql, custom, or directory (default: custom)
    --output PATH         Output directory (default: ./backups)
    --compress            Compress backup (applies to SQL format)
    --retention DAYS      Keep backups for N days (default: 30)
    --help, -h            Show this help message

${BOLD}Database Configuration:${NC}
    Set via environment variables:
    - DB_HOST (default: localhost)
    - DB_PORT (default: 5432)
    - DB_NAME (default: neurondb)
    - DB_USER (default: postgres)
    - DB_PASSWORD

${BOLD}Formats:${NC}
    sql        Plain SQL dump (text format)
    custom     PostgreSQL custom format (compressed, supports parallel restore)
    directory  Directory format (parallel dump, best for large databases)

${BOLD}Examples:${NC}
    # Basic backup with custom format
    $SCRIPT_NAME

    # SQL backup with compression
    $SCRIPT_NAME --format sql --compress

    # Directory format for large database
    $SCRIPT_NAME --format directory --output /backups/neurondb

    # Custom retention policy
    $SCRIPT_NAME --retention 7

EOF
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v pg_dump &> /dev/null; then
        log_error "pg_dump not found. Install PostgreSQL client tools."
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
    log_info "Testing database connection to $DB_HOST:$DB_PORT/$DB_NAME..."
    
    export PGPASSWORD="$DB_PASSWORD"
    
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" &> /dev/null; then
        log_success "Database connection successful"
        return 0
    else
        log_error "Cannot connect to database"
        return 1
    fi
}

create_output_directory() {
    log_info "Creating output directory: $OUTPUT_DIR"
    
    mkdir -p "$OUTPUT_DIR"
    
    if [ ! -w "$OUTPUT_DIR" ]; then
        log_error "Output directory not writable: $OUTPUT_DIR"
        return 1
    fi
    
    log_success "Output directory ready"
    return 0
}

# ============================================================================
# Backup Functions
# ============================================================================

backup_sql_format() {
    local backup_file="$OUTPUT_DIR/neurondb_backup_${TIMESTAMP}.sql"
    
    log_info "Creating SQL backup: $backup_file"
    
    export PGPASSWORD="$DB_PASSWORD"
    
    pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" \
            --format=plain \
            --verbose \
            --file="$backup_file" \
            "$DB_NAME" 2>&1 | grep -v "^pg_dump:" || true
    
    if [ $? -eq 0 ] && [ -f "$backup_file" ]; then
        local size=$(du -h "$backup_file" | cut -f1)
        log_success "SQL backup created: $backup_file ($size)"
        
        if [ "$COMPRESS" = true ]; then
            log_info "Compressing backup..."
            gzip -f "$backup_file"
            backup_file="${backup_file}.gz"
            size=$(du -h "$backup_file" | cut -f1)
            log_success "Compressed backup: $backup_file ($size)"
        fi
        
        echo "$backup_file"
        return 0
    else
        log_error "SQL backup failed"
        return 1
    fi
}

backup_custom_format() {
    local backup_file="$OUTPUT_DIR/neurondb_backup_${TIMESTAMP}.dump"
    
    log_info "Creating custom format backup: $backup_file"
    
    export PGPASSWORD="$DB_PASSWORD"
    
    pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" \
            --format=custom \
            --compress=9 \
            --verbose \
            --file="$backup_file" \
            "$DB_NAME" 2>&1 | grep -v "^pg_dump:" || true
    
    if [ $? -eq 0 ] && [ -f "$backup_file" ]; then
        local size=$(du -h "$backup_file" | cut -f1)
        log_success "Custom format backup created: $backup_file ($size)"
        echo "$backup_file"
        return 0
    else
        log_error "Custom format backup failed"
        return 1
    fi
}

backup_directory_format() {
    local backup_dir="$OUTPUT_DIR/neurondb_backup_${TIMESTAMP}_dir"
    
    log_info "Creating directory format backup: $backup_dir"
    
    export PGPASSWORD="$DB_PASSWORD"
    
    pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" \
            --format=directory \
            --jobs=4 \
            --compress=9 \
            --verbose \
            --file="$backup_dir" \
            "$DB_NAME" 2>&1 | grep -v "^pg_dump:" || true
    
    if [ $? -eq 0 ] && [ -d "$backup_dir" ]; then
        local size=$(du -sh "$backup_dir" | cut -f1)
        log_success "Directory format backup created: $backup_dir ($size)"
        echo "$backup_dir"
        return 0
    else
        log_error "Directory format backup failed"
        return 1
    fi
}

# ============================================================================
# Maintenance Functions
# ============================================================================

cleanup_old_backups() {
    log_info "Cleaning up backups older than $RETENTION_DAYS days..."
    
    local deleted_count=0
    
    # Find and delete old backups
    while IFS= read -r -d '' file; do
        log_info "Deleting old backup: $(basename "$file")"
        rm -rf "$file"
        ((deleted_count++))
    done < <(find "$OUTPUT_DIR" -type f \( -name "neurondb_backup_*.sql*" -o -name "neurondb_backup_*.dump" \) -mtime +"$RETENTION_DAYS" -print0 2>/dev/null)
    
    # Delete old directory backups
    while IFS= read -r -d '' dir; do
        log_info "Deleting old backup directory: $(basename "$dir")"
        rm -rf "$dir"
        ((deleted_count++))
    done < <(find "$OUTPUT_DIR" -type d -name "neurondb_backup_*_dir" -mtime +"$RETENTION_DAYS" -print0 2>/dev/null)
    
    if [ $deleted_count -gt 0 ]; then
        log_success "Deleted $deleted_count old backup(s)"
    else
        log_info "No old backups to delete"
    fi
}

get_database_info() {
    log_info "Gathering database information..."
    
    export PGPASSWORD="$DB_PASSWORD"
    
    local db_size=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT pg_size_pretty(pg_database_size('$DB_NAME'));" 2>/dev/null | xargs)
    local tables=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog', 'information_schema');" 2>/dev/null | xargs)
    
    echo -e "${BOLD}Database Information:${NC}"
    echo "  Name: $DB_NAME"
    echo "  Host: $DB_HOST:$DB_PORT"
    echo "  Size: $db_size"
    echo "  Tables: $tables"
    echo ""
}

# ============================================================================
# Argument Parsing
# ============================================================================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
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
    
    # Validate format
    if [[ ! "$BACKUP_FORMAT" =~ ^(sql|custom|directory)$ ]]; then
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
    
    if ! check_database_connection; then
        exit 1
    fi
    
    get_database_info
    
    if ! create_output_directory; then
        exit 1
    fi
    
    echo -e "${BOLD}Backup Configuration:${NC}"
    echo "  Format: $BACKUP_FORMAT"
    echo "  Output: $OUTPUT_DIR"
    echo "  Compress: $COMPRESS"
    echo "  Retention: $RETENTION_DAYS days"
    echo ""
    
    local backup_result=""
    
    case "$BACKUP_FORMAT" in
        sql)
            backup_result=$(backup_sql_format)
            ;;
        custom)
            backup_result=$(backup_custom_format)
            ;;
        directory)
            backup_result=$(backup_directory_format)
            ;;
    esac
    
    if [ $? -eq 0 ] && [ -n "$backup_result" ]; then
        cleanup_old_backups
        
        echo ""
        echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║${NC}  ${BOLD}Backup Completed Successfully!${NC}                            ${GREEN}║${NC}"
        echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${BOLD}Backup Details:${NC}"
        echo "  Location: $backup_result"
        if [ -f "$backup_result" ]; then
            echo "  Size: $(du -h "$backup_result" | cut -f1)"
        elif [ -d "$backup_result" ]; then
            echo "  Size: $(du -sh "$backup_result" | cut -f1)"
        fi
        echo "  Timestamp: $TIMESTAMP"
        echo ""
        echo -e "${BOLD}Restore Instructions:${NC}"
        case "$BACKUP_FORMAT" in
            sql)
                echo "  psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME < $backup_result"
                ;;
            custom)
                echo "  pg_restore -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME $backup_result"
                ;;
            directory)
                echo "  pg_restore -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -j 4 $backup_result"
                ;;
        esac
        echo ""
        exit 0
    else
        log_error "Backup failed"
        exit 1
    fi
}

main "$@"


