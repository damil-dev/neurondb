#!/bin/bash
# Integration Verification Script for NeuronDB Ecosystem
#
# Verifies that all three modules (NeuronDB, NeuronMCP, NeuronAgent) are
# properly integrated and can work together seamlessly.
#
# Usage:
#   ./scripts/verify_neurondb_integration.sh
#   DB_HOST=localhost DB_PORT=5432 DB_NAME=neurondb DB_USER=postgres ./scripts/verify_neurondb_integration.sh

set -e

# Default values (can be overridden by environment variables)
DB_HOST="${NEURONDB_HOST:-${DB_HOST:-localhost}}"
DB_PORT="${NEURONDB_PORT:-${DB_PORT:-5432}}"
DB_NAME="${NEURONDB_DATABASE:-${DB_NAME:-neurondb}}"
DB_USER="${NEURONDB_USER:-${DB_USER:-postgres}}"
DB_PASSWORD="${NEURONDB_PASSWORD:-${DB_PASSWORD:-}}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
    ((TESTS_TOTAL++))
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
    ((TESTS_TOTAL++))
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_section() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}\n"
}

# Function to run SQL query and check result
run_sql_check() {
    local query="$1"
    local description="$2"
    local expected_result="${3:-1}"
    
    if [ -n "$DB_PASSWORD" ]; then
        export PGPASSWORD="$DB_PASSWORD"
    fi
    
    local result=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc "$query" 2>/dev/null || echo "0")
    
    if [ "$result" = "$expected_result" ] || [ "$result" -gt 0 ] 2>/dev/null; then
        print_success "$description"
        return 0
    else
        print_fail "$description (got: $result, expected: $expected_result)"
        return 1
    fi
}

# Function to test NeuronDB core functionality
test_neurondb_core() {
    print_section "Testing NeuronDB Core"
    
    # Check extension exists
    run_sql_check \
        "SELECT 1 FROM pg_extension WHERE extname = 'neurondb'" \
        "NeuronDB extension installed"
    
    # Check vector type exists
    run_sql_check \
        "SELECT 1 FROM pg_type WHERE typname = 'neurondb_vector'" \
        "neurondb_vector type available"
    
    # Test creating a vector
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c \
        "SELECT '[1,2,3]'::neurondb_vector(3) AS test_vector" > /dev/null 2>&1; then
        print_success "Can create neurondb_vector values"
    else
        print_fail "Cannot create neurondb_vector values"
    fi
}

# Function to test NeuronMCP integration
test_neurondb_mcp() {
    print_section "Testing NeuronMCP Integration"
    
    # Check schema exists
    run_sql_check \
        "SELECT 1 FROM information_schema.schemata WHERE schema_name = 'neurondb'" \
        "NeuronMCP schema exists"
    
    # Check key tables
    local mcp_tables=(
        "llm_providers"
        "llm_models"
        "llm_model_keys"
        "llm_model_configs"
        "index_configs"
        "index_templates"
        "worker_configs"
        "tool_configs"
        "system_configs"
    )
    
    for table in "${mcp_tables[@]}"; do
        run_sql_check \
            "SELECT 1 FROM information_schema.tables WHERE table_schema = 'neurondb' AND table_name = '$table'" \
            "Table neurondb.$table exists"
    done
    
    # Check views
    run_sql_check \
        "SELECT 1 FROM information_schema.views WHERE table_schema = 'neurondb' AND table_name = 'v_llm_models_active'" \
        "View neurondb.v_llm_models_active exists"
    
    # Check key functions
    local functions=(
        "neurondb_set_model_key"
        "neurondb_get_model_key"
        "neurondb_list_models"
        "neurondb_get_index_config"
        "neurondb_get_worker_config"
        "neurondb_get_tool_config"
        "neurondb_get_system_config"
    )
    
    for func in "${functions[@]}"; do
        run_sql_check \
            "SELECT 1 FROM pg_proc WHERE proname = '$func' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb')" \
            "Function neurondb.$func exists"
    done
    
    # Test function calls (non-destructive)
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c \
        "SELECT neurondb_list_providers()" > /dev/null 2>&1; then
        print_success "Can call neurondb_list_providers()"
    else
        print_fail "Cannot call neurondb_list_providers()"
    fi
    
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c \
        "SELECT neurondb_get_system_config()" > /dev/null 2>&1; then
        print_success "Can call neurondb_get_system_config()"
    else
        print_fail "Cannot call neurondb_get_system_config()"
    fi
    
    # Check pre-populated data
    local provider_count=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT COUNT(*) FROM neurondb.llm_providers" 2>/dev/null || echo "0")
    
    if [ "$provider_count" -ge 5 ]; then
        print_success "Pre-populated LLM providers exist ($provider_count providers)"
    else
        print_warning "Fewer LLM providers than expected ($provider_count providers, expected >= 5)"
    fi
}

# Function to test NeuronAgent integration
test_neurondb_agent() {
    print_section "Testing NeuronAgent Integration"
    
    # Check schema exists
    run_sql_check \
        "SELECT 1 FROM information_schema.schemata WHERE schema_name = 'neurondb_agent'" \
        "NeuronAgent schema exists"
    
    # Check key tables
    local agent_tables=(
        "agents"
        "sessions"
        "messages"
        "memory_chunks"
        "tools"
        "jobs"
        "api_keys"
    )
    
    for table in "${agent_tables[@]}"; do
        run_sql_check \
            "SELECT 1 FROM information_schema.tables WHERE table_schema = 'neurondb_agent' AND table_name = '$table'" \
            "Table neurondb_agent.$table exists"
    done
    
    # Check that memory_chunks table has vector column
    run_sql_check \
        "SELECT 1 FROM information_schema.columns WHERE table_schema = 'neurondb_agent' AND table_name = 'memory_chunks' AND data_type LIKE '%vector%'" \
        "memory_chunks table has vector column"
    
    # Check HNSW index exists
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT 1 FROM pg_indexes WHERE schemaname = 'neurondb_agent' AND indexname = 'idx_memory_chunks_embedding_hnsw'" | grep -q 1; then
        print_success "HNSW index on memory_chunks.embedding exists"
    else
        print_fail "HNSW index on memory_chunks.embedding missing"
    fi
    
    # Test inserting a vector (cleanup after)
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c \
        "CREATE TEMP TABLE test_vector_table (id INT, vec neurondb_vector(768)); \
         INSERT INTO test_vector_table VALUES (1, '[0.1,0.2,0.3]'::neurondb_vector(768)); \
         SELECT 1 FROM test_vector_table WHERE id = 1;" > /dev/null 2>&1; then
        print_success "Can insert and query neurondb_vector in tables"
    else
        print_fail "Cannot insert neurondb_vector in tables"
    fi
}

# Function to test cross-module integration
test_cross_module() {
    print_section "Testing Cross-Module Integration"
    
    # Test that NeuronAgent can reference NeuronMCP model names
    # (This is a conceptual test - in practice, agents reference model names as strings)
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c \
        "SELECT 1 FROM neurondb.llm_models WHERE model_name = 'gpt-4-turbo' LIMIT 1" > /dev/null 2>&1; then
        print_success "NeuronMCP model catalog accessible"
    else
        print_warning "NeuronMCP model catalog may not be populated"
    fi
    
    # Test that both schemas can coexist
    run_sql_check \
        "SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name IN ('neurondb', 'neurondb_agent')" \
        "Both schemas coexist (neurondb and neurondb_agent)"
    
    # Test vector operations work across modules
    # Create a test vector and verify it works
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c \
        "SELECT 1 FROM neurondb_agent.memory_chunks LIMIT 0" > /dev/null 2>&1; then
        print_success "Can query NeuronAgent vector tables"
    else
        print_fail "Cannot query NeuronAgent vector tables"
    fi
    
    # Test that functions from both modules are accessible
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c \
        "SELECT neurondb_get_all_configs()" > /dev/null 2>&1; then
        print_success "NeuronMCP functions accessible from default search path"
    else
        print_fail "NeuronMCP functions not accessible"
    fi
}

# Function to test vector operations
test_vector_operations() {
    print_section "Testing Vector Operations"
    
    # Test creating vectors of different dimensions
    for dim in 128 256 384 512 768 1024 1536; do
        if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c \
            "SELECT format('[%s]', string_agg('0.1', ','))::neurondb_vector($dim) AS v" > /dev/null 2>&1; then
            print_success "Can create neurondb_vector($dim)"
        else
            print_fail "Cannot create neurondb_vector($dim)"
        fi
    done
    
    # Test vector distance operations (if supported)
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c \
        "SELECT 1 FROM pg_operator WHERE oprname = '<->' AND oprleft = (SELECT oid FROM pg_type WHERE typname = 'neurondb_vector')" > /dev/null 2>&1; then
        print_success "Vector distance operator (<->) available"
    else
        print_warning "Vector distance operator may not be available"
    fi
}

# Function to show summary
show_summary() {
    print_section "Verification Summary"
    
    echo ""
    echo "Total Tests: $TESTS_TOTAL"
    echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
    echo -e "${RED}Failed: $TESTS_FAILED${NC}"
    echo ""
    
    if [ $TESTS_FAILED -eq 0 ]; then
        print_success "All integration tests passed! ✓"
        echo ""
        print_info "The NeuronDB ecosystem is properly integrated and ready to use."
        return 0
    else
        print_fail "Some integration tests failed. Please review the errors above."
        echo ""
        print_info "Common issues:"
        print_info "  1. NeuronDB extension not installed - run: CREATE EXTENSION neurondb;"
        print_info "  2. Migrations not run - run: ./scripts/setup_neurondb_ecosystem.sh"
        print_info "  3. Database connection issues - check connection parameters"
        return 1
    fi
}

# Main execution
main() {
    echo ""
    print_section "NeuronDB Ecosystem Integration Verification"
    echo ""
    print_info "This script verifies that NeuronDB, NeuronMCP, and NeuronAgent"
    print_info "are properly integrated and can work together seamlessly."
    echo ""
    print_info "Connection details:"
    print_info "  Host: $DB_HOST"
    print_info "  Port: $DB_PORT"
    print_info "  Database: $DB_NAME"
    print_info "  User: $DB_USER"
    echo ""
    
    # Check PostgreSQL connection
    if [ -n "$DB_PASSWORD" ]; then
        export PGPASSWORD="$DB_PASSWORD"
    fi
    
    if ! psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1" > /dev/null 2>&1; then
        print_fail "Cannot connect to PostgreSQL database"
        echo ""
        print_info "Please check:"
        print_info "  1. PostgreSQL is running"
        print_info "  2. Connection parameters are correct"
        print_info "  3. Database '$DB_NAME' exists"
        exit 1
    fi
    
    # Run all tests
    test_neurondb_core
    test_neurondb_mcp
    test_neurondb_agent
    test_cross_module
    test_vector_operations
    
    # Show summary
    show_summary
}

# Run main function
main "$@"

