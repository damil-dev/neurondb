#!/bin/bash
#
# Comprehensive Docker Ecosystem Verification Script
# Verifies NeuronDB, NeuronAgent, NeuronDesktop, and NeuronMCP are all working together
#
# Usage:
#   ./scripts/verify-docker-ecosystem.sh [--verbose] [--skip-service SERVICE]
#
# Exit codes:
#   0 = All checks passed
#   1 = One or more checks failed
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
SKIP_SERVICES=""
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --skip-service)
            SKIP_SERVICES="${SKIP_SERVICES} $2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--verbose] [--skip-service SERVICE]"
            echo "  --verbose         Show detailed output"
            echo "  --skip-service    Skip checking a specific service (can be used multiple times)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Helper functions
should_skip() {
    local service=$1
    echo "$SKIP_SERVICES" | grep -q "\b$service\b" && return 0 || return 1
}

print_section() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_check() {
    local status=$1
    local message=$2
    local details="${3:-}"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}✓${NC} $message"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    elif [ "$status" = "FAIL" ]; then
        echo -e "${RED}✗${NC} $message"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
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

print_section "Docker Ecosystem Verification"

# Check if services are running
if [ "$($COMPOSE_CMD ps 2>/dev/null | grep -c "Up")" -eq 0 ]; then
    echo -e "${YELLOW}Warning: No services are currently running.${NC}"
    echo "To start all services, run:"
    echo "  docker compose up -d"
    echo ""
    echo "The verification will continue but most checks will fail until services are started."
    echo ""
fi

# Check if Docker is running
print_info "Checking Docker daemon..."
if ! docker info >/dev/null 2>&1; then
    print_check "FAIL" "Docker daemon is not running"
    exit 1
fi
print_check "PASS" "Docker daemon is running"

# ============================================================================
# Service Health Checks
# ============================================================================
print_section "Service Health Checks"

# Check NeuronDB
if ! should_skip "neurondb"; then
    print_info "Checking NeuronDB service..."
    if $COMPOSE_CMD ps neurondb 2>/dev/null | grep -q "Up.*healthy"; then
        print_check "PASS" "NeuronDB container is running and healthy"
    elif $COMPOSE_CMD ps neurondb 2>/dev/null | grep -q "Up"; then
        print_check "FAIL" "NeuronDB container is running but not healthy"
    else
        print_check "FAIL" "NeuronDB service is not running (start with: docker compose up -d neurondb)"
    fi
fi

# Check NeuronAgent
if ! should_skip "neuronagent"; then
    print_info "Checking NeuronAgent service..."
    if $COMPOSE_CMD ps neuronagent 2>/dev/null | grep -q "Up.*healthy"; then
        print_check "PASS" "NeuronAgent container is running and healthy"
    elif $COMPOSE_CMD ps neuronagent 2>/dev/null | grep -q "Up"; then
        print_check "FAIL" "NeuronAgent container is running but not healthy"
    else
        print_check "FAIL" "NeuronAgent container is not running"
    fi
fi

# Check NeuronMCP
if ! should_skip "neuronmcp"; then
    print_info "Checking NeuronMCP service..."
    if $COMPOSE_CMD ps neuronmcp 2>/dev/null | grep -q "Up.*healthy"; then
        print_check "PASS" "NeuronMCP container is running and healthy"
    elif $COMPOSE_CMD ps neuronmcp 2>/dev/null | grep -q "Up"; then
        print_check "FAIL" "NeuronMCP container is running but not healthy"
    else
        print_check "FAIL" "NeuronMCP service is not running (start with: docker compose up -d neuronmcp)"
    fi
fi

# Check NeuronDesktop API
if ! should_skip "neurondesk-api"; then
    print_info "Checking NeuronDesktop API service..."
    if $COMPOSE_CMD ps neurondesk-api 2>/dev/null | grep -q "Up.*healthy"; then
        print_check "PASS" "NeuronDesktop API container is running and healthy"
    elif $COMPOSE_CMD ps neurondesk-api 2>/dev/null | grep -q "Up"; then
        print_check "FAIL" "NeuronDesktop API container is running but not healthy"
    else
        print_check "FAIL" "NeuronDesktop API container is not running"
    fi
fi

# Check NeuronDesktop Frontend
if ! should_skip "neurondesk-frontend"; then
    print_info "Checking NeuronDesktop Frontend service..."
    if $COMPOSE_CMD ps neurondesk-frontend 2>/dev/null | grep -q "Up"; then
        print_check "PASS" "NeuronDesktop Frontend container is running"
    else
        print_check "FAIL" "NeuronDesktop Frontend container is not running"
    fi
fi

# ============================================================================
# NeuronDB Functionality Tests
# ============================================================================
print_section "NeuronDB Functionality"

if ! should_skip "neurondb"; then
    print_info "Testing NeuronDB extension..."
    
    # Check if container is running first
    if ! $COMPOSE_CMD ps neurondb 2>/dev/null | grep -q "Up"; then
        print_check "FAIL" "NeuronDB container is not running" "Skipping functionality tests"
    else
        # Test: Extension is loaded
        if $COMPOSE_CMD exec -T neurondb psql -U neurondb -d neurondb -c "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'neurondb');" 2>/dev/null | grep -q "t"; then
            print_check "PASS" "NeuronDB extension is loaded"
        else
            print_check "FAIL" "NeuronDB extension is not loaded"
        fi
        
        # Test: Version function works
        VERSION=$($COMPOSE_CMD exec -T neurondb psql -U neurondb -d neurondb -t -c "SELECT neurondb.version();" 2>/dev/null | tr -d '[:space:]' || echo "")
        if [ -n "$VERSION" ]; then
            print_check "PASS" "NeuronDB version function works" "Version: $VERSION"
        else
            print_check "FAIL" "NeuronDB version function failed"
        fi
        
        # Test: Can create vector table
        $COMPOSE_CMD exec -T neurondb psql -U neurondb -d neurondb -c "DROP TABLE IF EXISTS verify_test;" >/dev/null 2>&1 || true
        if $COMPOSE_CMD exec -T neurondb psql -U neurondb -d neurondb -c "CREATE TABLE verify_test (id SERIAL PRIMARY KEY, embedding vector(3));" >/dev/null 2>&1; then
            print_check "PASS" "Can create vector table"
            $COMPOSE_CMD exec -T neurondb psql -U neurondb -d neurondb -c "DROP TABLE verify_test;" >/dev/null 2>&1 || true
        else
            print_check "FAIL" "Cannot create vector table"
        fi
    fi
fi

# ============================================================================
# NeuronAgent Functionality Tests
# ============================================================================
print_section "NeuronAgent Functionality"

if ! should_skip "neuronagent"; then
    print_info "Testing NeuronAgent API..."
    
    # Test: Health endpoint
    HEALTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health 2>/dev/null || echo "000")
    if [ "$HEALTH_CODE" = "200" ]; then
        print_check "PASS" "NeuronAgent health endpoint responds" "HTTP $HEALTH_CODE"
    else
        print_check "FAIL" "NeuronAgent health endpoint not responding" "HTTP $HEALTH_CODE"
    fi
    
    # Test: API endpoints are accessible (may require auth)
    API_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/api/v1/agents 2>/dev/null || echo "000")
    if [ "$API_CODE" = "200" ] || [ "$API_CODE" = "401" ] || [ "$API_CODE" = "403" ]; then
        print_check "PASS" "NeuronAgent API endpoints are accessible" "HTTP $API_CODE"
    else
        print_check "FAIL" "NeuronAgent API endpoints not accessible" "HTTP $API_CODE"
    fi
fi

# ============================================================================
# NeuronMCP Functionality Tests
# ============================================================================
print_section "NeuronMCP Functionality"

if ! should_skip "neuronmcp"; then
    print_info "Testing NeuronMCP server..."
    
    # Check if container is running first
    if ! $COMPOSE_CMD ps neuronmcp 2>/dev/null | grep -q "Up"; then
        print_check "FAIL" "NeuronMCP container is not running" "Skipping functionality tests"
    else
        # Test: Binary exists and is executable
        if $COMPOSE_CMD exec -T neuronmcp test -f /app/neuronmcp && $COMPOSE_CMD exec -T neuronmcp test -x /app/neuronmcp 2>/dev/null; then
            print_check "PASS" "NeuronMCP binary exists and is executable"
            
            # Test: MCP initialize handshake
            MCP_INIT='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"verify-test","version":"1.0.0"}}}'
            MCP_RESPONSE=$(echo "$MCP_INIT" | $COMPOSE_CMD exec -T neuronmcp /app/neuronmcp 2>/dev/null | head -5 || echo "")
            if echo "$MCP_RESPONSE" | grep -qi "jsonrpc\|result\|id"; then
                print_check "PASS" "NeuronMCP responds to initialize handshake"
            else
                print_check "FAIL" "NeuronMCP does not respond to initialize"
            fi
        else
            print_check "FAIL" "NeuronMCP binary not found or not executable"
        fi
    fi
fi

# ============================================================================
# NeuronDesktop Functionality Tests
# ============================================================================
print_section "NeuronDesktop Functionality"

if ! should_skip "neurondesk-api"; then
    print_info "Testing NeuronDesktop API..."
    
    # Test: Health endpoint
    HEALTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8081/health 2>/dev/null || echo "000")
    if [ "$HEALTH_CODE" = "200" ]; then
        print_check "PASS" "NeuronDesktop API health endpoint responds" "HTTP $HEALTH_CODE"
    else
        print_check "FAIL" "NeuronDesktop API health endpoint not responding" "HTTP $HEALTH_CODE"
    fi
    
    # Test: API endpoints are accessible
    API_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8081/api/v1/health 2>/dev/null || echo "000")
    if [ "$API_CODE" = "200" ] || [ "$API_CODE" = "401" ]; then
        print_check "PASS" "NeuronDesktop API endpoints are accessible" "HTTP $API_CODE"
    else
        print_check "FAIL" "NeuronDesktop API endpoints not accessible" "HTTP $API_CODE"
    fi
fi

if ! should_skip "neurondesk-frontend"; then
    print_info "Testing NeuronDesktop Frontend..."
    
    # Test: Frontend is accessible
    FRONTEND_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 2>/dev/null || echo "000")
    if [ "$FRONTEND_CODE" = "200" ] || [ "$FRONTEND_CODE" = "304" ]; then
        print_check "PASS" "NeuronDesktop Frontend is accessible" "HTTP $FRONTEND_CODE"
    else
        print_check "FAIL" "NeuronDesktop Frontend not accessible" "HTTP $FRONTEND_CODE"
    fi
fi

# ============================================================================
# Integration Tests
# ============================================================================
print_section "Integration Tests"

# Test: NeuronAgent can connect to NeuronDB
if ! should_skip "neuronagent" && ! should_skip "neurondb"; then
    print_info "Testing NeuronAgent → NeuronDB connection..."
    # Check if containers are running
    if $COMPOSE_CMD ps neuronagent 2>/dev/null | grep -q "Up" && $COMPOSE_CMD ps neurondb 2>/dev/null | grep -q "Up"; then
        # Check if NeuronAgent can query NeuronDB (via its database connection)
        if $COMPOSE_CMD exec neuronagent sh -c "nc -z neurondb 5432" 2>/dev/null; then
            print_check "PASS" "NeuronAgent can reach NeuronDB"
        else
            print_check "FAIL" "NeuronAgent cannot reach NeuronDB"
        fi
    else
        print_check "FAIL" "Required containers not running" "Start services first"
    fi
fi

# Test: NeuronMCP can connect to NeuronDB
if ! should_skip "neuronmcp" && ! should_skip "neurondb"; then
    print_info "Testing NeuronMCP → NeuronDB connection..."
    # Check if containers are running
    if $COMPOSE_CMD ps neuronmcp 2>/dev/null | grep -q "Up" && $COMPOSE_CMD ps neurondb 2>/dev/null | grep -q "Up"; then
        # Check if NeuronMCP can query NeuronDB (via environment variables)
        if $COMPOSE_CMD exec neuronmcp sh -c "nc -z neurondb 5432" 2>/dev/null; then
            print_check "PASS" "NeuronMCP can reach NeuronDB"
        else
            print_check "FAIL" "NeuronMCP cannot reach NeuronDB"
        fi
    else
        print_check "FAIL" "Required containers not running" "Start services first"
    fi
fi

# Test: NeuronDesktop can connect to NeuronDB
if ! should_skip "neurondesk-api" && ! should_skip "neurondb"; then
    print_info "Testing NeuronDesktop → NeuronDB connection..."
    if $COMPOSE_CMD ps neurondesk-api 2>/dev/null | grep -q "Up" && $COMPOSE_CMD ps neurondb 2>/dev/null | grep -q "Up"; then
        if $COMPOSE_CMD exec neurondesk-api sh -c "nc -z neurondb 5432" 2>/dev/null; then
            print_check "PASS" "NeuronDesktop can reach NeuronDB"
        else
            print_check "FAIL" "NeuronDesktop cannot reach NeuronDB"
        fi
    else
        print_check "FAIL" "Required containers not running" "Start services first"
    fi
fi

# Test: NeuronDesktop can connect to NeuronAgent
if ! should_skip "neurondesk-api" && ! should_skip "neuronagent"; then
    print_info "Testing NeuronDesktop → NeuronAgent connection..."
    if $COMPOSE_CMD ps neurondesk-api 2>/dev/null | grep -q "Up" && $COMPOSE_CMD ps neuronagent 2>/dev/null | grep -q "Up"; then
        if $COMPOSE_CMD exec neurondesk-api sh -c "nc -z neuronagent 8080" 2>/dev/null; then
            print_check "PASS" "NeuronDesktop can reach NeuronAgent"
        else
            print_check "FAIL" "NeuronDesktop cannot reach NeuronAgent"
        fi
    else
        print_check "FAIL" "Required containers not running" "Start services first"
    fi
fi

# Test: NeuronDesktop has NeuronMCP binary (if needed)
if ! should_skip "neurondesk-api"; then
    print_info "Checking NeuronDesktop for NeuronMCP binary..."
    if $COMPOSE_CMD ps neurondesk-api 2>/dev/null | grep -q "Up"; then
        if $COMPOSE_CMD exec neurondesk-api test -f /usr/local/bin/neurondb-mcp 2>/dev/null; then
            if $COMPOSE_CMD exec neurondesk-api test -x /usr/local/bin/neurondb-mcp 2>/dev/null; then
                print_check "PASS" "NeuronDesktop has NeuronMCP binary"
            else
                print_check "FAIL" "NeuronDesktop has NeuronMCP binary but it's not executable"
            fi
        else
            print_check "FAIL" "NeuronDesktop does not have NeuronMCP binary" "Rebuild neurondesk-api after building neuronmcp"
        fi
    else
        print_check "FAIL" "NeuronDesktop API container is not running" "Start services first"
    fi
fi

# ============================================================================
# Summary
# ============================================================================
print_section "Summary"

echo "Total checks: $TOTAL_CHECKS"
echo -e "${GREEN}Passed: $PASSED_CHECKS${NC}"
if [ $FAILED_CHECKS -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED_CHECKS${NC}"
else
    echo "Failed: $FAILED_CHECKS"
fi

if [ $FAILED_CHECKS -eq 0 ]; then
    echo ""
    echo -e "${GREEN}All checks passed! The Docker ecosystem is working correctly.${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}Some checks failed. Please review the output above and check service logs:${NC}"
    echo "  $COMPOSE_CMD logs neurondb"
    echo "  $COMPOSE_CMD logs neuronagent"
    echo "  $COMPOSE_CMD logs neurondb-mcp"
    echo "  $COMPOSE_CMD logs neurondesk-api"
    echo "  $COMPOSE_CMD logs neurondesk-frontend"
    exit 1
fi

