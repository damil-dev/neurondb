#!/bin/bash
# Smoke test script to verify all services are running correctly

set -euo pipefail

echo "🧪 Running NeuronDB Ecosystem Smoke Tests"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NEURONDB_HOST="${NEURONDB_HOST:-localhost}"
NEURONDB_PORT="${NEURONDB_PORT:-5432}"
NEURONAGENT_URL="${NEURONAGENT_URL:-http://localhost:8080}"
NEURONDESKTOP_URL="${NEURONDESKTOP_URL:-http://localhost:8081}"
NEURONMCP_URL="${NEURONMCP_URL:-http://localhost:8082}"

FAILED=0
PASSED=0

# Test function
test_endpoint() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}
    
    echo -n "Testing $name... "
    
    if response=$(curl -s -w "\n%{http_code}" -o /tmp/response.json "$url" 2>/dev/null); then
        status_code=$(echo "$response" | tail -n1)
        if [ "$status_code" = "$expected_status" ]; then
            echo -e "${GREEN}✓ PASSED${NC}"
            PASSED=$((PASSED + 1))
            return 0
        else
            echo -e "${RED}✗ FAILED (expected $expected_status, got $status_code)${NC}"
            FAILED=$((FAILED + 1))
            return 1
        fi
    else
        echo -e "${RED}✗ FAILED (connection error)${NC}"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# Test PostgreSQL connection
echo ""
echo "📊 Testing PostgreSQL Connection..."
if PGPASSWORD="${POSTGRES_PASSWORD:-postgres}" psql -h "$NEURONDB_HOST" -p "$NEURONDB_PORT" -U postgres -d postgres -c "SELECT 1;" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ PostgreSQL connection successful${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ PostgreSQL connection failed${NC}"
    FAILED=$((FAILED + 1))
fi

# Test NeuronDB extension
echo ""
echo "📊 Testing NeuronDB Extension..."
if PGPASSWORD="${POSTGRES_PASSWORD:-postgres}" psql -h "$NEURONDB_HOST" -p "$NEURONDB_PORT" -U postgres -d postgres -c "SELECT neurondb_version();" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ NeuronDB extension loaded${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${YELLOW}⚠ NeuronDB extension not loaded (may be expected)${NC}"
fi

# Test NeuronAgent
echo ""
echo "📊 Testing NeuronAgent..."
test_endpoint "NeuronAgent Health" "$NEURONAGENT_URL/health" 200
test_endpoint "NeuronAgent API" "$NEURONAGENT_URL/api/v1/agents" 401  # Should require auth

# Test NeuronDesktop API
echo ""
echo "📊 Testing NeuronDesktop API..."
test_endpoint "NeuronDesktop Health" "$NEURONDESKTOP_URL/health" 200
test_endpoint "NeuronDesktop API" "$NEURONDESKTOP_URL/api/v1/profiles" 401  # Should require auth

# Test NeuronMCP
echo ""
echo "📊 Testing NeuronMCP..."
if curl -s "$NEURONMCP_URL/health" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ NeuronMCP health check passed${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${YELLOW}⚠ NeuronMCP health check failed (may be expected)${NC}"
fi

# Test Docker services (if running in Docker)
echo ""
echo "📊 Testing Docker Services..."
if command -v docker >/dev/null 2>&1; then
    if docker ps | grep -q neurondb; then
        echo -e "${GREEN}✓ NeuronDB container running${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${YELLOW}⚠ NeuronDB container not found${NC}"
    fi
    
    if docker ps | grep -q neuronagent; then
        echo -e "${GREEN}✓ NeuronAgent container running${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${YELLOW}⚠ NeuronAgent container not found${NC}"
    fi
    
    if docker ps | grep -q neurondesk; then
        echo -e "${GREEN}✓ NeuronDesktop container running${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${YELLOW}⚠ NeuronDesktop container not found${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Docker not available${NC}"
fi

# Summary
echo ""
echo "=========================================="
echo "📊 Test Summary"
echo "=========================================="
echo -e "${GREEN}Passed: $PASSED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}Failed: $FAILED${NC}"
    exit 0
fi
