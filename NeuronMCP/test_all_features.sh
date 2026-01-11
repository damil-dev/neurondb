#!/bin/bash

# NeuronMCP Feature Test Suite
# Tests all features one by one

set -e

echo "=================================================================================="
echo "NEURONMCP COMPREHENSIVE FEATURE TEST SUITE"
echo "=================================================================================="

cd "$(dirname "$0")"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0

test_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ PASSED${NC}: $2"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗ FAILED${NC}: $2"
        FAILED=$((FAILED + 1))
    fi
}

echo ""
echo "=== PHASE 1: BUILD VERIFICATION ==="
echo ""

echo "Test 1.1: Building all packages..."
if go build ./internal/... ./pkg/... 2>&1; then
    test_result 0 "Build all packages"
else
    test_result 1 "Build all packages"
fi

echo ""
echo "Test 1.2: Running go vet..."
if go vet ./internal/... ./pkg/... 2>&1; then
    test_result 0 "Go vet check"
else
    test_result 1 "Go vet check"
fi

echo ""
echo "=== PHASE 2: TOOL REGISTRATION ==="
echo ""

echo "Test 2.1: Checking tool registration..."
TOOL_COUNT=$(grep -r "registry.Register" internal/tools/register.go | wc -l | tr -d ' ')
if [ "$TOOL_COUNT" -ge 150 ]; then
    test_result 0 "Tool registration count ($TOOL_COUNT tools)"
else
    test_result 1 "Tool registration count ($TOOL_COUNT tools, expected >= 150)"
fi

echo ""
echo "Test 2.2: Checking PostgreSQL tools..."
POSTGRESQL_TOOLS=$(grep "postgresql_" internal/tools/register.go | wc -l | tr -d ' ')
if [ "$POSTGRESQL_TOOLS" -ge 50 ]; then
    test_result 0 "PostgreSQL tools ($POSTGRESQL_TOOLS tools)"
else
    test_result 1 "PostgreSQL tools ($POSTGRESQL_TOOLS tools, expected >= 50)"
fi

echo ""
echo "Test 2.3: Checking vector tools..."
VECTOR_TOOLS=$(grep "vector_" internal/tools/register.go | wc -l | tr -d ' ')
if [ "$VECTOR_TOOLS" -ge 20 ]; then
    test_result 0 "Vector tools ($VECTOR_TOOLS tools)"
else
    test_result 1 "Vector tools ($VECTOR_TOOLS tools, expected >= 20)"
fi

echo ""
echo "Test 2.4: Checking ML tools..."
ML_TOOLS=$(grep -E "(ml_|train_|predict)" internal/tools/register.go | wc -l | tr -d ' ')
if [ "$ML_TOOLS" -ge 15 ]; then
    test_result 0 "ML tools ($ML_TOOLS tools)"
else
    test_result 1 "ML tools ($ML_TOOLS tools, expected >= 15)"
fi

echo ""
echo "Test 2.5: Checking graph tools..."
GRAPH_TOOLS=$(grep "graph" internal/tools/register.go | wc -l | tr -d ' ')
if [ "$GRAPH_TOOLS" -ge 5 ]; then
    test_result 0 "Graph tools ($GRAPH_TOOLS tools)"
else
    test_result 1 "Graph tools ($GRAPH_TOOLS tools, expected >= 5)"
fi

echo ""
echo "Test 2.6: Checking multi-modal tools..."
MULTIMODAL_TOOLS=$(grep -E "(multimodal|image_|audio_)" internal/tools/register.go | wc -l | tr -d ' ')
if [ "$MULTIMODAL_TOOLS" -ge 5 ]; then
    test_result 0 "Multi-modal tools ($MULTIMODAL_TOOLS tools)"
else
    test_result 1 "Multi-modal tools ($MULTIMODAL_TOOLS tools, expected >= 5)"
fi

echo ""
echo "=== PHASE 3: SECURITY FEATURES ==="
echo ""

echo "Test 3.1: Checking RBAC module..."
if [ -f "internal/security/rbac.go" ]; then
    test_result 0 "RBAC module exists"
else
    test_result 1 "RBAC module missing"
fi

echo ""
echo "Test 3.2: Checking API key rotation..."
if [ -f "internal/security/api_key_rotation.go" ]; then
    test_result 0 "API key rotation module exists"
else
    test_result 1 "API key rotation module missing"
fi

echo ""
echo "Test 3.3: Checking MFA support..."
if [ -f "internal/security/mfa.go" ]; then
    test_result 0 "MFA module exists"
else
    test_result 1 "MFA module missing"
fi

echo ""
echo "Test 3.4: Checking data masking..."
if [ -f "internal/security/data_masking.go" ]; then
    test_result 0 "Data masking module exists"
else
    test_result 1 "Data masking module missing"
fi

echo ""
echo "Test 3.5: Checking network security..."
if [ -f "internal/security/network_security.go" ]; then
    test_result 0 "Network security module exists"
else
    test_result 1 "Network security module missing"
fi

echo ""
echo "Test 3.6: Checking compliance framework..."
if [ -f "internal/security/compliance.go" ]; then
    test_result 0 "Compliance module exists"
else
    test_result 1 "Compliance module missing"
fi

echo ""
echo "=== PHASE 4: OBSERVABILITY ==="
echo ""

echo "Test 4.1: Checking metrics collection..."
if [ -f "internal/observability/metrics.go" ]; then
    test_result 0 "Metrics module exists"
else
    test_result 1 "Metrics module missing"
fi

echo ""
echo "Test 4.2: Checking distributed tracing..."
if [ -f "internal/observability/tracing.go" ]; then
    test_result 0 "Tracing module exists"
else
    test_result 1 "Tracing module missing"
fi

echo ""
echo "=== PHASE 5: HIGH AVAILABILITY ==="
echo ""

echo "Test 5.1: Checking health check system..."
if [ -f "internal/ha/health.go" ]; then
    test_result 0 "Health check module exists"
else
    test_result 1 "Health check module missing"
fi

echo ""
echo "Test 5.2: Checking HA features compilation..."
if go build ./internal/ha/... 2>&1; then
    test_result 0 "HA module compiles"
else
    test_result 1 "HA module compilation failed"
fi

echo ""
echo "=== PHASE 6: PLUGIN SYSTEM ==="
echo ""

echo "Test 6.1: Checking plugin framework..."
if [ -f "internal/plugin/plugin.go" ]; then
    test_result 0 "Plugin framework exists"
else
    test_result 1 "Plugin framework missing"
fi

echo ""
echo "Test 6.2: Checking plugin system compilation..."
if go build ./internal/plugin/... 2>&1; then
    test_result 0 "Plugin system compiles"
else
    test_result 1 "Plugin system compilation failed"
fi

echo ""
echo "=== PHASE 7: PERFORMANCE BENCHMARKING ==="
echo ""

echo "Test 7.1: Checking performance benchmarking..."
if [ -f "internal/performance/benchmark.go" ]; then
    test_result 0 "Benchmarking module exists"
else
    test_result 1 "Benchmarking module missing"
fi

echo ""
echo "=== PHASE 8: SDK IMPLEMENTATIONS ==="
echo ""

echo "Test 8.1: Checking Python SDK..."
if [ -f "sdks/python/neurondb_mcp/client.py" ]; then
    test_result 0 "Python SDK exists"
else
    test_result 1 "Python SDK missing"
fi

echo ""
echo "Test 8.2: Checking TypeScript SDK..."
if [ -f "sdks/typescript/src/client.ts" ]; then
    test_result 0 "TypeScript SDK exists"
else
    test_result 1 "TypeScript SDK missing"
fi

echo ""
echo "=== PHASE 9: FILE STRUCTURE VERIFICATION ==="
echo ""

echo "Test 9.1: Checking PostgreSQL tool files..."
POSTGRESQL_FILES=$(find internal/tools -name "postgresql_*.go" | wc -l | tr -d ' ')
if [ "$POSTGRESQL_FILES" -ge 6 ]; then
    test_result 0 "PostgreSQL tool files ($POSTGRESQL_FILES files)"
else
    test_result 1 "PostgreSQL tool files ($POSTGRESQL_FILES files, expected >= 6)"
fi

echo ""
echo "Test 9.2: Checking vector tool files..."
VECTOR_FILES=$(find internal/tools -name "vector_*.go" | wc -l | tr -d ' ')
if [ "$VECTOR_FILES" -ge 5 ]; then
    test_result 0 "Vector tool files ($VECTOR_FILES files)"
else
    test_result 1 "Vector tool files ($VECTOR_FILES files, expected >= 5)"
fi

echo ""
echo "Test 9.3: Checking ML tool files..."
ML_FILES=$(find internal/tools -name "ml_*.go" | wc -l | tr -d ' ')
if [ "$ML_FILES" -ge 2 ]; then
    test_result 0 "ML tool files ($ML_FILES files)"
else
    test_result 1 "ML tool files ($ML_FILES files, expected >= 2)"
fi

echo ""
echo "Test 9.4: Checking security module files..."
SECURITY_FILES=$(find internal/security -name "*.go" | wc -l | tr -d ' ')
if [ "$SECURITY_FILES" -ge 6 ]; then
    test_result 0 "Security module files ($SECURITY_FILES files)"
else
    test_result 1 "Security module files ($SECURITY_FILES files, expected >= 6)"
fi

echo ""
echo "=== PHASE 10: COMPILATION VERIFICATION ==="
echo ""

echo "Test 10.1: Compiling tools package..."
if go build ./internal/tools/... 2>&1; then
    test_result 0 "Tools package compilation"
else
    test_result 1 "Tools package compilation"
fi

echo ""
echo "Test 10.2: Compiling security package..."
if go build ./internal/security/... 2>&1; then
    test_result 0 "Security package compilation"
else
    test_result 1 "Security package compilation"
fi

echo ""
echo "Test 10.3: Compiling observability package..."
if go build ./internal/observability/... 2>&1; then
    test_result 0 "Observability package compilation"
else
    test_result 1 "Observability package compilation"
fi

echo ""
echo "Test 10.4: Compiling plugin package..."
if go build ./internal/plugin/... 2>&1; then
    test_result 0 "Plugin package compilation"
else
    test_result 1 "Plugin package compilation"
fi

echo ""
echo "=================================================================================="
echo "TEST SUMMARY"
echo "=================================================================================="
echo ""
TOTAL=$((PASSED + FAILED))
echo "Total Tests: $TOTAL"
echo -e "${GREEN}Passed: $PASSED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED${NC}"
else
    echo -e "${GREEN}Failed: $FAILED${NC}"
fi
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  $FAILED TEST(S) FAILED${NC}"
    exit 1
fi

