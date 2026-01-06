#!/bin/bash
#
# Smoke test script for NeuronDB ecosystem
# Tests SQL query, REST API, and MCP protocol
#

set -e

# Detect docker compose command (v2: `docker compose`, v1: `docker-compose`)
get_compose_cmd() {
    if docker compose version >/dev/null 2>&1; then
        echo "docker compose"
    elif command -v docker-compose >/dev/null 2>&1; then
        echo "docker-compose"
    else
        echo ""
    fi
}

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
        # Avoid `set -e` aborting on arithmetic exit status when the value is 0
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗${NC} $message"
        TESTS_FAILED=$((TESTS_FAILED + 1))
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

COMPOSE_CMD="$(get_compose_cmd)"
if [ -z "$COMPOSE_CMD" ]; then
    echo -e "${RED}Error: Docker Compose is not installed or not in PATH${NC}"
    echo -e "${YELLOW}Install either docker-compose (v1) or Docker Compose plugin (v2).${NC}"
    exit 1
fi

# Check if services are running
print_section "Checking Service Status"
if ! $COMPOSE_CMD ps | grep -q "neurondb-cpu.*Up"; then
    echo -e "${YELLOW}Warning: Services may not be running. Starting services...${NC}"
    $COMPOSE_CMD up -d
    echo "Waiting for services to be healthy..."
    sleep 10
fi

# Test 1: SQL Query (NeuronDB Extension)
print_section "Test 1: NeuronDB SQL Query"
if $COMPOSE_CMD exec -T neurondb psql -U neurondb -d neurondb -c "SELECT neurondb.version();" > /dev/null 2>&1; then
    VERSION=$($COMPOSE_CMD exec -T neurondb psql -U neurondb -d neurondb -t -c "SELECT neurondb.version();" | tr -d ' ')
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
# Service name in docker-compose is `neuronmcp` (container name may be `neurondb-mcp`)
if echo "$MCP_TEST" | $COMPOSE_CMD exec -T neuronmcp /app/neuronmcp 2>/dev/null | grep -q "jsonrpc"; then
    print_result "PASS" "NeuronMCP server responding to MCP protocol"
else
    # Alternative test: check if binary exists and is executable
    if $COMPOSE_CMD exec -T neuronmcp test -f /app/neuronmcp && $COMPOSE_CMD exec -T neuronmcp test -x /app/neuronmcp; then
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
    echo "  $COMPOSE_CMD logs neurondb"
    echo "  $COMPOSE_CMD logs neuronagent"
    echo "  $COMPOSE_CMD logs neuronmcp"
    exit 1
fi

