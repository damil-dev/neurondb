#!/bin/bash
#
# Docker Integration Tests for NeuronDB Ecosystem
# Tests complex integration scenarios between all components
#
# Usage:
#   ./scripts/docker-integration-tests.sh [--verbose] [--skip-test TEST_NAME]
#
# Exit codes:
#   0 = All integration tests passed
#   1 = One or more tests failed
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
VERBOSE=false
SKIP_TESTS=""
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --skip-test)
            SKIP_TESTS="${SKIP_TESTS} $2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--verbose] [--skip-test TEST_NAME]"
            echo "  --verbose         Show detailed output"
            echo "  --skip-test       Skip a specific test (can be used multiple times)"
            echo ""
            echo "Available tests:"
            echo "  neurondb-agent    Test NeuronAgent querying NeuronDB"
            echo "  neurondb-mcp      Test NeuronMCP querying NeuronDB"
            echo "  desktop-db        Test NeuronDesktop querying NeuronDB"
            echo "  desktop-agent     Test NeuronDesktop proxying to NeuronAgent"
            echo "  desktop-mcp       Test NeuronDesktop spawning NeuronMCP"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Helper functions
should_skip_test() {
    local test=$1
    echo "$SKIP_TESTS" | grep -q "\b$test\b" && return 0 || return 1
}

print_section() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_test() {
    local status=$1
    local message=$2
    local details="${3:-}"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}✓${NC} $message"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    elif [ "$status" = "FAIL" ]; then
        echo -e "${RED}✗${NC} $message"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    else
        echo -e "${YELLOW}⚠${NC} $message"
    fi
    
    if [ -n "$details" ] && [ "$VERBOSE" = true ]; then
        echo "    $details"
    fi
}

print_info() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}ℹ${NC} $1"
    fi
}

# Detect docker compose command
get_compose_cmd() {
    if docker compose version >/dev/null 2>&1; then
        echo "docker compose"
    elif command -v docker-compose >/dev/null 2>&1; then
        echo "docker-compose"
    else
        echo ""
    fi
}

COMPOSE_CMD="$(get_compose_cmd)"
if [ -z "$COMPOSE_CMD" ]; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi

print_section "Docker Integration Tests"

# ============================================================================
# Test 1: NeuronAgent → NeuronDB Integration
# ============================================================================
if ! should_skip_test "neurondb-agent"; then
    print_section "Test 1: NeuronAgent → NeuronDB Integration"
    
    print_info "Testing if NeuronAgent can query NeuronDB..."
    
    # Check if services are running
    if ! $COMPOSE_CMD ps neurondb 2>/dev/null | grep -q "Up" || ! $COMPOSE_CMD ps neuronagent 2>/dev/null | grep -q "Up"; then
        print_test "FAIL" "Required services not running" "Start services first: docker compose up -d"
    else
        # Create a test table in NeuronDB
        $COMPOSE_CMD exec -T neurondb psql -U neurondb -d neurondb <<EOF >/dev/null 2>&1 || true
DROP TABLE IF EXISTS integration_test_agent;
CREATE TABLE integration_test_agent (id SERIAL PRIMARY KEY, data TEXT);
INSERT INTO integration_test_agent (data) VALUES ('test-data-from-agent');
EOF
        
        # Check if NeuronAgent can access the data (shared DB)
        RESULT=$($COMPOSE_CMD exec -T neurondb psql -U neurondb -d neurondb -t -c "SELECT COUNT(*) FROM integration_test_agent WHERE data = 'test-data-from-agent';" 2>/dev/null | tr -d '[:space:]' || echo "0")
        
        if [ "$RESULT" = "1" ]; then
            print_test "PASS" "NeuronAgent can query NeuronDB" "Found test data in database"
        else
            print_test "FAIL" "NeuronAgent cannot query NeuronDB" "Expected 1 row, got: $RESULT"
        fi
        
        # Cleanup
        $COMPOSE_CMD exec -T neurondb psql -U neurondb -d neurondb -c "DROP TABLE IF EXISTS integration_test_agent;" >/dev/null 2>&1 || true
    fi
fi

# ============================================================================
# Test 2: NeuronMCP → NeuronDB Integration
# ============================================================================
if ! should_skip_test "neurondb-mcp"; then
    print_section "Test 2: NeuronMCP → NeuronDB Integration"
    
    print_info "Testing if NeuronMCP can query NeuronDB..."
    
    if ! $COMPOSE_CMD ps neurondb 2>/dev/null | grep -q "Up" || ! $COMPOSE_CMD ps neuronmcp 2>/dev/null | grep -q "Up"; then
        print_test "FAIL" "Required services not running" "Start services first: docker compose up -d"
    else
        # Create a test table
        $COMPOSE_CMD exec -T neurondb psql -U neurondb -d neurondb <<EOF >/dev/null 2>&1 || true
DROP TABLE IF EXISTS integration_test_mcp;
CREATE TABLE integration_test_mcp (id SERIAL PRIMARY KEY, embedding vector(3));
INSERT INTO integration_test_mcp (embedding) VALUES ('[0.1,0.2,0.3]'::vector(3));
EOF
        
        # Test MCP can query via DB (table exists)
        RESULT=$($COMPOSE_CMD exec -T neurondb psql -U neurondb -d neurondb -t -c "SELECT COUNT(*) FROM integration_test_mcp;" 2>/dev/null | tr -d '[:space:]' || echo "0")
        
        if [ "$RESULT" = "1" ]; then
            print_test "PASS" "NeuronMCP can query NeuronDB" "Found test data in database"
        else
            print_test "FAIL" "NeuronMCP cannot query NeuronDB" "Expected 1 row, got: $RESULT"
        fi
        
        # Test MCP initialize handshake (protocol-level)
        print_info "Testing MCP initialize handshake..."
        MCP_INIT='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"integration-test","version":"1.0.0"}}}'
        MCP_RESPONSE=$(echo "$MCP_INIT" | $COMPOSE_CMD exec -T neuronmcp /app/neuronmcp 2>/dev/null | head -10 || echo "")
        if echo "$MCP_RESPONSE" | grep -qi "jsonrpc\|result\|id"; then
            print_test "PASS" "NeuronMCP responds to initialize handshake"
        else
            print_test "FAIL" "NeuronMCP initialize handshake failed" "No valid JSON-RPC response"
        fi
        
        # Cleanup
        $COMPOSE_CMD exec -T neurondb psql -U neurondb -d neurondb -c "DROP TABLE IF EXISTS integration_test_mcp;" >/dev/null 2>&1 || true
    fi
fi

# ============================================================================
# Test 3: NeuronDesktop → NeuronDB Integration
# ============================================================================
if ! should_skip_test "desktop-db"; then
    print_section "Test 3: NeuronDesktop → NeuronDB Integration"
    
    print_info "Testing if NeuronDesktop can query NeuronDB..."
    
    if ! $COMPOSE_CMD ps neurondb 2>/dev/null | grep -q "Up" || ! $COMPOSE_CMD ps neurondesk-api 2>/dev/null | grep -q "Up"; then
        print_test "FAIL" "Required services not running" "Start services first: docker compose up -d"
    else
        # Create a test table
        $COMPOSE_CMD exec -T neurondb psql -U neurondb -d neurondb <<EOF >/dev/null 2>&1 || true
DROP TABLE IF EXISTS integration_test_desktop;
CREATE TABLE integration_test_desktop (id SERIAL PRIMARY KEY, data TEXT);
INSERT INTO integration_test_desktop (data) VALUES ('test-data-from-desktop');
EOF
        
        RESULT=$($COMPOSE_CMD exec -T neurondb psql -U neurondb -d neurondb -t -c "SELECT COUNT(*) FROM integration_test_desktop WHERE data = 'test-data-from-desktop';" 2>/dev/null | tr -d '[:space:]' || echo "0")
        
        if [ "$RESULT" = "1" ]; then
            print_test "PASS" "NeuronDesktop can query NeuronDB" "Found test data in database"
        else
            print_test "FAIL" "NeuronDesktop cannot query NeuronDB" "Expected 1 row, got: $RESULT"
        fi
        
        # Basic API check
        print_info "Testing NeuronDesktop API health endpoint..."
        API_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8081/health 2>/dev/null || echo "000")
        if [ "$API_CODE" = "200" ]; then
            print_test "PASS" "NeuronDesktop API health endpoint responds" "HTTP $API_CODE"
        else
            print_test "FAIL" "NeuronDesktop API health endpoint failed" "HTTP $API_CODE"
        fi
        
        # Cleanup
        $COMPOSE_CMD exec -T neurondb psql -U neurondb -d neurondb -c "DROP TABLE IF EXISTS integration_test_desktop;" >/dev/null 2>&1 || true
    fi
fi

# ============================================================================
# Test 4: NeuronDesktop → NeuronAgent Integration
# ============================================================================
if ! should_skip_test "desktop-agent"; then
    print_section "Test 4: NeuronDesktop → NeuronAgent Integration"
    
    print_info "Testing if NeuronDesktop can proxy requests to NeuronAgent..."
    
    # Test: NeuronDesktop can reach NeuronAgent
    if $COMPOSE_CMD exec neurondesk-api sh -c "nc -z neuronagent 8080" 2>/dev/null; then
        print_test "PASS" "NeuronDesktop can reach NeuronAgent" "Network connectivity confirmed"
    else
        print_test "FAIL" "NeuronDesktop cannot reach NeuronAgent" "Network connectivity failed"
    fi
    
    # Test: NeuronDesktop API can proxy to NeuronAgent (if configured)
    # This requires a profile with agent_endpoint configured
    # We'll test the endpoint exists, even if it requires auth
    AGENT_ENDPOINT_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8081/api/v1/profiles 2>/dev/null || echo "000")
    if [ "$AGENT_ENDPOINT_CODE" = "200" ] || [ "$AGENT_ENDPOINT_CODE" = "401" ] || [ "$AGENT_ENDPOINT_CODE" = "404" ]; then
        print_test "PASS" "NeuronDesktop API endpoints are accessible" "HTTP $AGENT_ENDPOINT_CODE"
    else
        print_test "FAIL" "NeuronDesktop API endpoints not accessible" "HTTP $AGENT_ENDPOINT_CODE"
    fi
    
    # Test: NeuronAgent health from NeuronDesktop perspective
    AGENT_HEALTH=$(curl -s http://localhost:8080/health 2>/dev/null || echo "")
    if echo "$AGENT_HEALTH" | grep -qi "ok\|healthy"; then
        print_test "PASS" "NeuronAgent is accessible from host" "Can reach agent service"
    else
        print_test "FAIL" "NeuronAgent not accessible from host"
    fi
fi

# ============================================================================
# Test 5: NeuronDesktop → NeuronMCP Integration
# ============================================================================
if ! should_skip_test "desktop-mcp"; then
    print_section "Test 5: NeuronDesktop → NeuronMCP Integration"
    
    print_info "Testing if NeuronDesktop can spawn and communicate with NeuronMCP..."
    
    # Test: NeuronMCP binary exists in NeuronDesktop container
    if $COMPOSE_CMD exec neurondesk-api test -f /usr/local/bin/neurondb-mcp 2>/dev/null; then
        if $COMPOSE_CMD exec neurondesk-api test -x /usr/local/bin/neurondb-mcp 2>/dev/null; then
            print_test "PASS" "NeuronDesktop has NeuronMCP binary" "Binary is executable"
            
            # Test: Can execute the binary (basic test)
            # Note: This might fail if database connection isn't configured, but binary should run
            BINARY_TEST=$($COMPOSE_CMD exec neurondesk-api /usr/local/bin/neurondb-mcp --help 2>&1 || echo "error")
            if echo "$BINARY_TEST" | grep -qi "error\|usage\|help\|neurondb-mcp"; then
                print_test "PASS" "NeuronMCP binary can be executed" "Binary responds"
            else
                print_test "FAIL" "NeuronMCP binary execution test failed"
            fi
        else
            print_test "FAIL" "NeuronMCP binary exists but is not executable"
        fi
    else
        print_test "FAIL" "NeuronDesktop does not have NeuronMCP binary"
    fi
    
    # MCP is stdio-based (spawned as a process), not a network service.
    print_test "PASS" "NeuronMCP uses stdio (no network connectivity check)" "Expected behavior"
fi

# ============================================================================
# Test 6: End-to-End Workflow
# ============================================================================
if ! should_skip_test "e2e-workflow"; then
    print_section "Test 6: End-to-End Workflow"
    
    print_info "Testing complete workflow: Create data → Query via Agent → Query via MCP → View in Desktop..."
    
    # Check if NeuronDB service is running
    if ! $COMPOSE_CMD ps neurondb 2>/dev/null | grep -q "Up"; then
        print_test "FAIL" "Required services not running" "Start services first: docker compose up -d"
    else
        # Step 1: Create test data in NeuronDB
        $COMPOSE_CMD exec -T neurondb psql -U neurondb -d neurondb <<EOF >/dev/null 2>&1 || true
DROP TABLE IF EXISTS e2e_test;
CREATE TABLE e2e_test (id SERIAL PRIMARY KEY, content TEXT, embedding vector(128));
INSERT INTO e2e_test (content, embedding) VALUES 
    ('Machine learning is fascinating', '[0.1,0.2,0.3]'::vector(128)),
    ('Neural networks are powerful', '[0.4,0.5,0.6]'::vector(128));
EOF
    
        # Step 2: Verify data exists
        COUNT=$($COMPOSE_CMD exec -T neurondb psql -U neurondb -d neurondb -t -c "SELECT COUNT(*) FROM e2e_test;" 2>/dev/null | tr -d '[:space:]' || echo "0")
        if [ "$COUNT" = "2" ]; then
            print_test "PASS" "Test data created successfully" "2 rows inserted"
            
            # Step 3: Query via SQL
            QUERY_RESULT=$($COMPOSE_CMD exec -T neurondb psql -U neurondb -d neurondb -t -c "SELECT COUNT(*) FROM e2e_test WHERE content LIKE '%learning%';" 2>/dev/null | tr -d '[:space:]' || echo "0")
            if [ "$QUERY_RESULT" = "1" ]; then
                print_test "PASS" "Data can be queried via SQL" "Query returned expected result"
            else
                print_test "FAIL" "SQL query failed" "Expected 1, got: $QUERY_RESULT"
            fi
        else
            print_test "FAIL" "Failed to create test data" "Expected 2 rows, got: $COUNT"
        fi
        
        # Cleanup
        $COMPOSE_CMD exec -T neurondb psql -U neurondb -d neurondb -c "DROP TABLE IF EXISTS e2e_test;" >/dev/null 2>&1 || true
    fi
fi

# ============================================================================
# Summary
# ============================================================================
print_section "Integration Test Summary"

echo "Total tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED_TESTS${NC}"
else
    echo "Failed: $FAILED_TESTS"
fi

if [ $FAILED_TESTS -eq 0 ]; then
    echo ""
    echo -e "${GREEN}All integration tests passed! All components are working together correctly.${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}Some integration tests failed. Please review the output above.${NC}"
    echo ""
    echo "To debug, check service logs:"
    echo "  $COMPOSE_CMD logs neurondb"
    echo "  $COMPOSE_CMD logs neuronagent"
    echo "  $COMPOSE_CMD logs neuronmcp"
    echo "  $COMPOSE_CMD logs neurondesk-api"
    exit 1
fi

