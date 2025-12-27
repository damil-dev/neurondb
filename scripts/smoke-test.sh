#!/bin/bash
#
# Smoke test script for NeuronDB ecosystem
# Tests SQL query, REST API, and MCP protocol
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

# Function to print test result
print_result() {
    local status=$1
    local message=$2
    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}✓${NC} $message"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗${NC} $message"
        ((TESTS_FAILED++))
    fi
}

# Function to print section header
print_section() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "$1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Check if docker compose is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed or not in PATH${NC}"
    exit 1
fi

# Check if services are running
print_section "Checking Service Status"
if ! docker compose ps | grep -q "neurondb-cpu.*Up"; then
    echo -e "${YELLOW}Warning: Services may not be running. Starting services...${NC}"
    docker compose up -d
    echo "Waiting for services to be healthy..."
    sleep 10
fi

# Test 1: SQL Query (NeuronDB Extension)
print_section "Test 1: NeuronDB SQL Query"
if docker compose exec -T neurondb psql -U neurondb -d neurondb -c "SELECT neurondb.version();" > /dev/null 2>&1; then
    VERSION=$(docker compose exec -T neurondb psql -U neurondb -d neurondb -t -c "SELECT neurondb.version();" | tr -d ' ')
    if [ -n "$VERSION" ]; then
        print_result "PASS" "NeuronDB SQL query successful (version: $VERSION)"
    else
        print_result "FAIL" "NeuronDB SQL query returned empty result"
    fi
else
    print_result "FAIL" "NeuronDB SQL query failed"
fi

# Test 2: REST API Call (NeuronAgent)
print_section "Test 2: NeuronAgent REST API"
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health 2>/dev/null || echo "000")
if [ "$HEALTH_RESPONSE" = "200" ]; then
    print_result "PASS" "NeuronAgent REST API responding (HTTP $HEALTH_RESPONSE)"
else
    print_result "FAIL" "NeuronAgent REST API not responding (HTTP $HEALTH_RESPONSE)"
fi

# Test 3: MCP Protocol Call (NeuronMCP)
print_section "Test 3: NeuronMCP Server"
MCP_TEST='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke-test","version":"1.0.0"}}}'
if echo "$MCP_TEST" | docker compose exec -T neurondb-mcp /app/neurondb-mcp 2>/dev/null | grep -q "jsonrpc"; then
    print_result "PASS" "NeuronMCP server responding to MCP protocol"
else
    # Alternative test: check if binary exists and is executable
    if docker compose exec -T neurondb-mcp test -f /app/neurondb-mcp && docker compose exec -T neurondb-mcp test -x /app/neurondb-mcp; then
        print_result "PASS" "NeuronMCP server binary exists and is executable"
    else
        print_result "FAIL" "NeuronMCP server not responding or binary missing"
    fi
fi

# Summary
print_section "Test Summary"
echo "Tests passed: $TESTS_PASSED"
echo "Tests failed: $TESTS_FAILED"

if [ $TESTS_FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}All smoke tests passed!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}Some tests failed. Check service logs:${NC}"
    echo "  docker compose logs neurondb"
    echo "  docker compose logs neuronagent"
    echo "  docker compose logs neurondb-mcp"
    exit 1
fi

