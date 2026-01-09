#!/bin/bash
# Run All NeuronMCP Verification Scripts
#
# This script runs all verification tests in sequence to validate
# NeuronMCP compatibility and integration with NeuronDB.
#
# Usage:
#   ./run_all_verifications.sh
#   NEURONDB_HOST=localhost NEURONDB_PORT=5432 ./run_all_verifications.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo ""
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
    echo ""
}

# Check if Go is available
if ! command -v go &> /dev/null; then
    print_error "Go is not installed or not in PATH"
    exit 1
fi

print_section "NeuronMCP Comprehensive Verification Suite"

# Test 1: Database Connection Verification
print_section "Test 1: Database Connection Verification"
print_info "Testing database connection handling and retry logic..."
if go run "$SCRIPT_DIR/test_connection_verification.go"; then
    print_success "Database connection verification passed"
else
    print_error "Database connection verification failed"
    exit 1
fi

# Test 2: Vector Operations Verification
print_section "Test 2: Vector Operations Verification"
print_info "Testing vector search operations with different distance metrics..."
if go run "$SCRIPT_DIR/test_vector_operations.go"; then
    print_success "Vector operations verification passed"
else
    print_error "Vector operations verification failed"
    exit 1
fi

# Test 3: Schema Setup Validation
print_section "Test 3: Schema Setup Validation"
print_info "Validating NeuronMCP configuration schema setup..."
if go run "$SCRIPT_DIR/validate_schema_setup.go"; then
    print_success "Schema setup validation passed"
else
    print_error "Schema setup validation failed"
    print_info "Note: Some failures may be expected if schema is not yet set up"
    print_info "Run: ./scripts/setup_neurondb_mcp.sh"
fi

# Test 4: Version Compatibility Verification
print_section "Test 4: Version Compatibility Verification"
print_info "Verifying PostgreSQL version compatibility (16, 17, 18)..."
if go run "$SCRIPT_DIR/test_version_compatibility.go"; then
    print_success "Version compatibility verification passed"
else
    print_error "Version compatibility verification failed"
    exit 1
fi

# Test 5: Tool Execution Flow Verification
print_section "Test 5: Tool Execution Flow Verification"
print_info "Testing tool execution flow from MCP client to NeuronDB..."
if go run "$SCRIPT_DIR/test_tool_execution.go"; then
    print_success "Tool execution flow verification passed"
else
    print_error "Tool execution flow verification failed"
    exit 1
fi

# Summary
print_section "Verification Summary"
print_success "All verification tests completed!"
print_info ""
print_info "Next steps:"
print_info "  1. Review any warnings or skipped tests above"
print_info "  2. If schema validation failed, run: ./scripts/setup_neurondb_mcp.sh"
print_info "  3. Set API keys: SELECT neurondb_set_model_key('model_name', 'api_key');"
print_info "  4. View verification summary: cat VERIFICATION_SUMMARY.md"
print_info ""


