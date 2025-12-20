#!/bin/bash
# Unified Setup Script for NeuronDB Ecosystem
#
# Sets up the complete NeuronDB ecosystem including:
# 1. Database creation (if needed)
# 2. NeuronDB extension installation
# 3. NeuronMCP schema and functions
# 4. NeuronAgent migrations
#
# Usage:
#   ./scripts/setup_neurondb_ecosystem.sh
#   DB_HOST=localhost DB_PORT=5432 DB_NAME=neurondb DB_USER=postgres ./scripts/setup_neurondb_ecosystem.sh

set -e

# Default values (can be overridden by environment variables)
DB_HOST="${NEURONDB_HOST:-${DB_HOST:-localhost}}"
DB_PORT="${NEURONDB_PORT:-${DB_PORT:-5432}}"
DB_NAME="${NEURONDB_DATABASE:-${DB_NAME:-neurondb}}"
DB_USER="${NEURONDB_USER:-${DB_USER:-postgres}}"
DB_PASSWORD="${NEURONDB_PASSWORD:-${DB_PASSWORD:-}}"

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
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
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}\n"
}

# Function to check if PostgreSQL is accessible
check_postgres() {
    print_info "Checking PostgreSQL connection..."
    
    if [ -n "$DB_PASSWORD" ]; then
        export PGPASSWORD="$DB_PASSWORD"
    fi
    
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "SELECT 1" > /dev/null 2>&1; then
        print_success "Connected to PostgreSQL"
        return 0
    else
        print_error "Cannot connect to PostgreSQL"
        print_info "Connection details:"
        print_info "  Host: $DB_HOST"
        print_info "  Port: $DB_PORT"
        print_info "  User: $DB_USER"
        return 1
    fi
}

# Function to create database if it doesn't exist
create_database() {
    print_info "Checking if database '$DB_NAME' exists..."
    
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -tAc \
        "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" | grep -q 1; then
        print_success "Database '$DB_NAME' already exists"
        return 0
    else
        print_info "Creating database '$DB_NAME'..."
        if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c \
            "CREATE DATABASE $DB_NAME"; then
            print_success "Database '$DB_NAME' created"
            return 0
        else
            print_error "Failed to create database '$DB_NAME'"
            return 1
        fi
    fi
}

# Function to ensure NeuronDB extension exists
ensure_neurondb_extension() {
    print_info "Ensuring NeuronDB extension is installed..."
    
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT 1 FROM pg_extension WHERE extname = 'neurondb'" | grep -q 1; then
        print_success "NeuronDB extension is already installed"
        return 0
    else
        print_info "Installing NeuronDB extension..."
        if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c \
            "CREATE EXTENSION IF NOT EXISTS neurondb;" 2>/dev/null; then
            print_success "NeuronDB extension installed"
            return 0
        else
            print_error "Failed to install NeuronDB extension"
            print_warning "Please ensure NeuronDB is properly installed on the PostgreSQL server"
            print_info "You may need to install NeuronDB packages or build from source"
            return 1
        fi
    fi
}

# Function to run SQL file
run_sql_file() {
    local sql_file="$1"
    local description="$2"
    
    if [ ! -f "$sql_file" ]; then
        print_error "SQL file not found: $sql_file"
        return 1
    fi
    
    print_info "Running $description..."
    
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$sql_file" > /dev/null 2>&1; then
        print_success "$description completed"
        return 0
    else
        print_error "Failed to run $description"
        print_info "Attempting to show error details..."
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$sql_file" 2>&1 | tail -20
        return 1
    fi
}

# Function to setup NeuronMCP
setup_neurondb_mcp() {
    print_section "Setting up NeuronMCP"
    
    local mcp_sql_dir="$PROJECT_ROOT/NeuronMCP/sql"
    
    if [ ! -d "$mcp_sql_dir" ]; then
        print_error "NeuronMCP SQL directory not found: $mcp_sql_dir"
        return 1
    fi
    
    # Run schema setup
    if ! run_sql_file "$mcp_sql_dir/setup_neurondb_mcp_schema.sql" "NeuronMCP schema setup"; then
        return 1
    fi
    
    # Run functions setup
    if ! run_sql_file "$mcp_sql_dir/neurondb_mcp_functions.sql" "NeuronMCP functions setup"; then
        return 1
    fi
    
    print_success "NeuronMCP setup completed"
    return 0
}

# Function to setup NeuronAgent
setup_neurondb_agent() {
    print_section "Setting up NeuronAgent"
    
    local agent_migrations_dir="$PROJECT_ROOT/NeuronAgent/migrations"
    
    if [ ! -d "$agent_migrations_dir" ]; then
        print_error "NeuronAgent migrations directory not found: $agent_migrations_dir"
        return 1
    fi
    
    # Run migrations in order
    local migrations=(
        "001_initial_schema.sql"
        "002_add_indexes.sql"
        "003_add_triggers.sql"
        "004_advanced_features.sql"
    )
    
    for migration in "${migrations[@]}"; do
        local migration_file="$agent_migrations_dir/$migration"
        if [ -f "$migration_file" ]; then
            if ! run_sql_file "$migration_file" "NeuronAgent migration: $migration"; then
                return 1
            fi
        else
            print_warning "Migration file not found: $migration_file"
        fi
    done
    
    print_success "NeuronAgent setup completed"
    return 0
}

# Function to verify installation
verify_installation() {
    print_section "Verifying Installation"
    
    local all_ok=true
    
    # Check NeuronDB extension
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT 1 FROM pg_extension WHERE extname = 'neurondb'" | grep -q 1; then
        print_success "✓ NeuronDB extension installed"
    else
        print_error "✗ NeuronDB extension missing"
        all_ok=false
    fi
    
    # Check NeuronMCP schema
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT 1 FROM information_schema.schemata WHERE schema_name = 'neurondb'" | grep -q 1; then
        print_success "✓ NeuronMCP schema exists"
        
        # Check key tables
        local mcp_tables=("llm_providers" "llm_models" "tool_configs")
        for table in "${mcp_tables[@]}"; do
            if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
                "SELECT 1 FROM information_schema.tables WHERE table_schema = 'neurondb' AND table_name = '$table'" | grep -q 1; then
                print_success "  ✓ Table neurondb.$table exists"
            else
                print_error "  ✗ Table neurondb.$table missing"
                all_ok=false
            fi
        done
    else
        print_error "✗ NeuronMCP schema missing"
        all_ok=false
    fi
    
    # Check NeuronAgent schema
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT 1 FROM information_schema.schemata WHERE schema_name = 'neurondb_agent'" | grep -q 1; then
        print_success "✓ NeuronAgent schema exists"
        
        # Check key tables
        local agent_tables=("agents" "sessions" "messages" "memory_chunks")
        for table in "${agent_tables[@]}"; do
            if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
                "SELECT 1 FROM information_schema.tables WHERE table_schema = 'neurondb_agent' AND table_name = '$table'" | grep -q 1; then
                print_success "  ✓ Table neurondb_agent.$table exists"
            else
                print_error "  ✗ Table neurondb_agent.$table missing"
                all_ok=false
            fi
        done
    else
        print_error "✗ NeuronAgent schema missing"
        all_ok=false
    fi
    
    # Check vector type
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT 1 FROM pg_type WHERE typname = 'neurondb_vector'" | grep -q 1; then
        print_success "✓ neurondb_vector type available"
    else
        print_error "✗ neurondb_vector type missing"
        all_ok=false
    fi
    
    if [ "$all_ok" = true ]; then
        print_success "\nAll components verified successfully!"
        return 0
    else
        print_error "\nSome components failed verification"
        return 1
    fi
}

# Main execution
main() {
    echo ""
    print_section "NeuronDB Ecosystem Setup"
    echo ""
    print_info "This script will set up the complete NeuronDB ecosystem:"
    print_info "  1. Create database (if needed)"
    print_info "  2. Install NeuronDB extension"
    print_info "  3. Setup NeuronMCP schema and functions"
    print_info "  4. Setup NeuronAgent migrations"
    echo ""
    print_info "Connection details:"
    print_info "  Host: $DB_HOST"
    print_info "  Port: $DB_PORT"
    print_info "  Database: $DB_NAME"
    print_info "  User: $DB_USER"
    echo ""
    
    # Check prerequisites
    if ! check_postgres; then
        exit 1
    fi
    
    # Step 1: Create database
    print_section "Step 1: Database Setup"
    if ! create_database; then
        exit 1
    fi
    
    # Step 2: Ensure NeuronDB extension
    print_section "Step 2: NeuronDB Extension"
    if ! ensure_neurondb_extension; then
        exit 1
    fi
    
    # Step 3: Setup NeuronMCP
    if ! setup_neurondb_mcp; then
        print_error "NeuronMCP setup failed"
        exit 1
    fi
    
    # Step 4: Setup NeuronAgent
    if ! setup_neurondb_agent; then
        print_error "NeuronAgent setup failed"
        exit 1
    fi
    
    # Step 5: Verify installation
    if ! verify_installation; then
        print_warning "Installation completed but verification found issues"
        exit 1
    fi
    
    # Success message
    echo ""
    print_section "Setup Complete!"
    print_success "All components have been successfully set up."
    echo ""
    print_info "Next steps:"
    print_info "  1. Configure NeuronMCP: Set API keys using neurondb_set_model_key()"
    print_info "  2. Create agents: INSERT INTO neurondb_agent.agents (...) VALUES (...)"
    print_info "  3. Run verification: ./scripts/verify_neurondb_integration.sh"
    echo ""
}

# Run main function
main "$@"

