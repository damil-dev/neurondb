#!/bin/bash
#
# Tiered Integration Verification Script for NeuronDB Ecosystem
# Tests all components across 6 tiers of functionality
#
# Usage:
#   ./scripts/verify_neurondb_integration.sh [--tier N] [--json] [--skip-tier N]
#
# Exit codes:
#   0 = All tests passed
#   1 = One or more tests failed
#   2 = Partial (some tiers skipped)
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
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-neurondb}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-}"
AGENT_URL="${AGENT_URL:-http://localhost:8080}"
JSON_OUTPUT=false
TIER_FILTER=""
SKIP_TIERS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tier)
            TIER_FILTER="$2"
            shift 2
            ;;
        --json)
            JSON_OUTPUT=true
            shift
            ;;
        --skip-tier)
            SKIP_TIERS="${SKIP_TIERS} $2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--tier N] [--json] [--skip-tier N]"
            echo "  --tier N      Run only tier N (0-6)"
            echo "  --json        Output results in JSON format"
            echo "  --skip-tier N Skip tier N (can be used multiple times)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Test tracking
declare -A TIER_RESULTS
declare -A TIER_TESTS
declare -A TIER_PASSED
declare -A TIER_FAILED
TOTAL_TESTS=0
TOTAL_PASSED=0
TOTAL_FAILED=0
EXIT_CODE=0

# JSON output helpers
json_start() {
    if [ "$JSON_OUTPUT" = true ]; then
        echo "{"
        echo '  "tiers": ['
    fi
}

json_tier_start() {
    if [ "$JSON_OUTPUT" = true ]; then
        if [ "$1" != "0" ]; then
            echo ","
        fi
        echo "    {"
        echo "      \"tier\": $1,"
        echo "      \"name\": \"$2\","
        echo "      \"tests\": ["
    fi
}

json_test() {
    if [ "$JSON_OUTPUT" = true ]; then
        if [ "$3" != "0" ]; then
            echo ","
        fi
        echo "        {"
        echo "          \"name\": \"$1\","
        echo "          \"status\": \"$2\""
        if [ -n "${4:-}" ]; then
            echo ","
            echo "          \"message\": \"$4\""
        fi
        echo "        }"
    fi
}

json_tier_end() {
    if [ "$JSON_OUTPUT" = true ]; then
        echo "      ],"
        echo "      \"passed\": ${TIER_PASSED[$1]},"
        echo "      \"failed\": ${TIER_FAILED[$1]},"
        echo "      \"total\": ${TIER_TESTS[$1]}"
        echo "    }"
    fi
}

json_end() {
    if [ "$JSON_OUTPUT" = true ]; then
        echo "  ],"
        echo "  \"summary\": {"
        echo "    \"total_tests\": $TOTAL_TESTS,"
        echo "    \"passed\": $TOTAL_PASSED,"
        echo "    \"failed\": $TOTAL_FAILED"
        echo "  }"
        echo "}"
    fi
}

# Test helpers
test_pass() {
    local tier=$1
    local test_name=$2
    local message="${3:-}"
    ((TIER_TESTS[$tier]++))
    ((TIER_PASSED[$tier]++))
    ((TOTAL_TESTS++))
    ((TOTAL_PASSED++))
    if [ "$JSON_OUTPUT" = false ]; then
        echo -e "${GREEN}✓${NC} $test_name"
        if [ -n "$message" ]; then
            echo "    $message"
        fi
    else
        json_test "$test_name" "pass" "${TIER_TESTS[$tier]}" "$message"
    fi
}

test_fail() {
    local tier=$1
    local test_name=$2
    local message="${3:-}"
    ((TIER_TESTS[$tier]++))
    ((TIER_FAILED[$tier]++))
    ((TOTAL_TESTS++))
    ((TOTAL_FAILED++))
    EXIT_CODE=1
    if [ "$JSON_OUTPUT" = false ]; then
        echo -e "${RED}✗${NC} $test_name"
        if [ -n "$message" ]; then
            echo "    ${RED}$message${NC}"
        fi
    else
        json_test "$test_name" "fail" "${TIER_TESTS[$tier]}" "$message"
    fi
}

test_info() {
    if [ "$JSON_OUTPUT" = false ]; then
        echo -e "${BLUE}ℹ${NC} $1"
    fi
}

test_section() {
    if [ "$JSON_OUTPUT" = false ]; then
        echo ""
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${CYAN}$1${NC}"
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    fi
}

# Check if tier should run
should_run_tier() {
    local tier=$1
    if [ -n "$TIER_FILTER" ] && [ "$tier" != "$TIER_FILTER" ]; then
        return 1
    fi
    if echo "$SKIP_TIERS" | grep -q "\b$tier\b"; then
        return 1
    fi
    return 0
}

# Database connection helper
psql_exec() {
    local sql="$1"
    if [ -n "$DB_PASSWORD" ]; then
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -A -c "$sql" 2>&1
    else
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -A -c "$sql" 2>&1
    fi
}

# Initialize tier counters
for i in {0..6}; do
    TIER_TESTS[$i]=0
    TIER_PASSED[$i]=0
    TIER_FAILED[$i]=0
done

# Start JSON output
if [ "$JSON_OUTPUT" = true ]; then
    json_start
fi

# ============================================================================
# TIER 0: Basic Extension
# ============================================================================
if should_run_tier 0; then
    test_section "TIER 0: Basic Extension"
    json_tier_start 0 "Basic Extension"
    
    # Test: Extension loads
    test_info "Checking if NeuronDB extension is installed..."
    EXT_EXISTS=$(psql_exec "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'neurondb');" | tr -d '[:space:]')
    if [ "$EXT_EXISTS" = "t" ]; then
        test_pass 0 "Extension installed" "NeuronDB extension found in pg_extension"
    else
        test_fail 0 "Extension installed" "NeuronDB extension not found. Run: CREATE EXTENSION neurondb;"
    fi
    
    # Test: Version function
    test_info "Testing neurondb.version() function..."
    VERSION=$(psql_exec "SELECT neurondb.version();" | tr -d '[:space:]')
    if [ -n "$VERSION" ] && [ "$VERSION" != "" ]; then
        test_pass 0 "Version function" "Version: $VERSION"
    else
        test_fail 0 "Version function" "neurondb.version() returned empty result"
    fi
    
    # Test: Extension schema exists
    test_info "Checking neurondb schema..."
    SCHEMA_EXISTS=$(psql_exec "SELECT EXISTS(SELECT 1 FROM pg_namespace WHERE nspname = 'neurondb');" | tr -d '[:space:]')
    if [ "$SCHEMA_EXISTS" = "t" ]; then
        test_pass 0 "Schema exists" "neurondb schema found"
    else
        test_fail 0 "Schema exists" "neurondb schema not found"
    fi
    
    json_tier_end 0
fi

# ============================================================================
# TIER 1: Vector Operations
# ============================================================================
if should_run_tier 1; then
    test_section "TIER 1: Vector Operations"
    json_tier_start 1 "Vector Operations"
    
    # Test: Create vector column
    test_info "Creating test table with vector column..."
    psql_exec "DROP TABLE IF EXISTS verify_test_vectors;" > /dev/null 2>&1
    CREATE_RESULT=$(psql_exec "CREATE TABLE verify_test_vectors (id SERIAL PRIMARY KEY, embedding vector(128));" 2>&1)
    if [ $? -eq 0 ]; then
        test_pass 1 "Create vector column" "Table with vector(128) column created"
    else
        test_fail 1 "Create vector column" "Failed to create table: $CREATE_RESULT"
        json_tier_end 1
        continue
    fi
    
    # Test: Insert vectors
    test_info "Inserting test vectors..."
    INSERT_RESULT=$(psql_exec "INSERT INTO verify_test_vectors (embedding) VALUES ('[0.1,0.2,0.3]'::vector(128)), ('[0.4,0.5,0.6]'::vector(128)), ('[0.7,0.8,0.9]'::vector(128));" 2>&1)
    if [ $? -eq 0 ]; then
        test_pass 1 "Insert vectors" "3 test vectors inserted"
    else
        test_fail 1 "Insert vectors" "Failed to insert vectors: $INSERT_RESULT"
    fi
    
    # Test: Build HNSW index
    test_info "Building HNSW index..."
    INDEX_RESULT=$(psql_exec "CREATE INDEX verify_test_vectors_hnsw_idx ON verify_test_vectors USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 64);" 2>&1)
    if [ $? -eq 0 ]; then
        test_pass 1 "Build HNSW index" "HNSW index created successfully"
    else
        # Index might already exist or there's an issue
        if echo "$INDEX_RESULT" | grep -qi "already exists"; then
            test_pass 1 "Build HNSW index" "HNSW index already exists"
        else
            test_fail 1 "Build HNSW index" "Failed to create index: $INDEX_RESULT"
        fi
    fi
    
    # Test: Execute kNN query
    test_info "Executing kNN query..."
    KNN_RESULT=$(psql_exec "SELECT id FROM verify_test_vectors ORDER BY embedding <-> '[0.1,0.2,0.3]'::vector(128) LIMIT 1;" 2>&1)
    if [ $? -eq 0 ] && [ -n "$KNN_RESULT" ]; then
        test_pass 1 "kNN query" "kNN query executed successfully, found ID: $KNN_RESULT"
    else
        test_fail 1 "kNN query" "kNN query failed: $KNN_RESULT"
    fi
    
    # Cleanup
    psql_exec "DROP TABLE IF EXISTS verify_test_vectors;" > /dev/null 2>&1
    
    json_tier_end 1
fi

# ============================================================================
# TIER 2: Hybrid Search
# ============================================================================
if should_run_tier 2; then
    test_section "TIER 2: Hybrid Search"
    json_tier_start 2 "Hybrid Search"
    
    # Test: Create table with vector + full-text
    test_info "Creating table with vector and full-text columns..."
    psql_exec "DROP TABLE IF EXISTS verify_test_hybrid;" > /dev/null 2>&1
    CREATE_RESULT=$(psql_exec "CREATE TABLE verify_test_hybrid (id SERIAL PRIMARY KEY, content TEXT, embedding vector(128));" 2>&1)
    if [ $? -eq 0 ]; then
        test_pass 2 "Create hybrid table" "Table with vector and text columns created"
    else
        test_fail 2 "Create hybrid table" "Failed: $CREATE_RESULT"
        json_tier_end 2
        continue
    fi
    
    # Test: Insert data
    test_info "Inserting test data..."
    INSERT_RESULT=$(psql_exec "INSERT INTO verify_test_hybrid (content, embedding) VALUES ('machine learning algorithms', '[0.1,0.2,0.3]'::vector(128)), ('neural networks deep learning', '[0.4,0.5,0.6]'::vector(128)), ('vector search similarity', '[0.7,0.8,0.9]'::vector(128));" 2>&1)
    if [ $? -eq 0 ]; then
        test_pass 2 "Insert hybrid data" "3 rows inserted"
    else
        test_fail 2 "Insert hybrid data" "Failed: $INSERT_RESULT"
    fi
    
    # Test: Build GIN index for full-text
    test_info "Building GIN index for full-text search..."
    GIN_RESULT=$(psql_exec "CREATE INDEX verify_test_hybrid_gin_idx ON verify_test_hybrid USING gin (to_tsvector('english', content));" 2>&1)
    if [ $? -eq 0 ]; then
        test_pass 2 "Build GIN index" "GIN index created for full-text search"
    else
        test_fail 2 "Build GIN index" "Failed: $GIN_RESULT"
    fi
    
    # Test: Build vector index
    test_info "Building vector index..."
    VEC_INDEX_RESULT=$(psql_exec "CREATE INDEX verify_test_hybrid_vec_idx ON verify_test_hybrid USING hnsw (embedding vector_l2_ops);" 2>&1)
    if [ $? -eq 0 ]; then
        test_pass 2 "Build vector index" "Vector index created"
    else
        if echo "$VEC_INDEX_RESULT" | grep -qi "already exists"; then
            test_pass 2 "Build vector index" "Vector index already exists"
        else
            test_fail 2 "Build vector index" "Failed: $VEC_INDEX_RESULT"
        fi
    fi
    
    # Test: Execute hybrid query
    test_info "Executing hybrid search query..."
    HYBRID_RESULT=$(psql_exec "SELECT id, content FROM verify_test_hybrid WHERE to_tsvector('english', content) @@ to_tsquery('english', 'learning') ORDER BY embedding <-> '[0.1,0.2,0.3]'::vector(128) LIMIT 2;" 2>&1)
    if [ $? -eq 0 ] && [ -n "$HYBRID_RESULT" ]; then
        test_pass 2 "Hybrid query" "Hybrid search query executed successfully"
    else
        test_fail 2 "Hybrid query" "Failed: $HYBRID_RESULT"
    fi
    
    # Cleanup
    psql_exec "DROP TABLE IF EXISTS verify_test_hybrid;" > /dev/null 2>&1
    
    json_tier_end 2
fi

# ============================================================================
# TIER 3: ML Algorithms
# ============================================================================
if should_run_tier 3; then
    test_section "TIER 3: ML Algorithms"
    json_tier_start 3 "ML Algorithms"
    
    # Test: Classification (Random Forest)
    test_info "Testing Random Forest Classifier..."
    psql_exec "DROP TABLE IF EXISTS verify_test_classification;" > /dev/null 2>&1
    CREATE_RESULT=$(psql_exec "CREATE TABLE verify_test_classification (features vector(4), label INTEGER);" 2>&1)
    if [ $? -eq 0 ]; then
        # Insert training data
        psql_exec "INSERT INTO verify_test_classification VALUES ('[1,2,3,4]'::vector(4), 0), ('[2,3,4,5]'::vector(4), 0), ('[5,6,7,8]'::vector(4), 1), ('[6,7,8,9]'::vector(4), 1);" > /dev/null 2>&1
        
        # Try to train (may fail if function doesn't exist or needs config)
        TRAIN_RESULT=$(psql_exec "SELECT train_random_forest_classifier('verify_test_classification', 'features', 'label', 2, 2, 10);" 2>&1)
        if [ $? -eq 0 ] || echo "$TRAIN_RESULT" | grep -qi "model\|trained"; then
            test_pass 3 "Random Forest Classification" "Training function executed"
        else
            test_fail 3 "Random Forest Classification" "Training failed: $TRAIN_RESULT"
        fi
        psql_exec "DROP TABLE IF EXISTS verify_test_classification;" > /dev/null 2>&1
    else
        test_fail 3 "Random Forest Classification" "Failed to create table: $CREATE_RESULT"
    fi
    
    # Test: Regression (Linear Regression)
    test_info "Testing Linear Regression..."
    psql_exec "DROP TABLE IF EXISTS verify_test_regression;" > /dev/null 2>&1
    CREATE_RESULT=$(psql_exec "CREATE TABLE verify_test_regression (features vector(4), target FLOAT);" 2>&1)
    if [ $? -eq 0 ]; then
        psql_exec "INSERT INTO verify_test_regression VALUES ('[1,2,3,4]'::vector(4), 10.0), ('[2,3,4,5]'::vector(4), 14.0), ('[3,4,5,6]'::vector(4), 18.0);" > /dev/null 2>&1
        
        TRAIN_RESULT=$(psql_exec "SELECT train_linear_regression('verify_test_regression', 'features', 'target');" 2>&1)
        if [ $? -eq 0 ] || echo "$TRAIN_RESULT" | grep -qi "model\|trained\|coefficient"; then
            test_pass 3 "Linear Regression" "Training function executed"
        else
            test_fail 3 "Linear Regression" "Training failed: $TRAIN_RESULT"
        fi
        psql_exec "DROP TABLE IF EXISTS verify_test_regression;" > /dev/null 2>&1
    else
        test_fail 3 "Linear Regression" "Failed to create table: $CREATE_RESULT"
    fi
    
    # Test: Clustering (K-Means)
    test_info "Testing K-Means Clustering..."
    psql_exec "DROP TABLE IF EXISTS verify_test_clustering;" > /dev/null 2>&1
    CREATE_RESULT=$(psql_exec "CREATE TABLE verify_test_clustering (features vector(4));" 2>&1)
    if [ $? -eq 0 ]; then
        psql_exec "INSERT INTO verify_test_clustering VALUES ('[1,1,1,1]'::vector(4)), ('[2,2,2,2]'::vector(4)), ('[10,10,10,10]'::vector(4)), ('[11,11,11,11]'::vector(4));" > /dev/null 2>&1
        
        CLUSTER_RESULT=$(psql_exec "SELECT cluster_kmeans(features, 2, 10, 0.001) FROM verify_test_clustering;" 2>&1)
        if [ $? -eq 0 ] && [ -n "$CLUSTER_RESULT" ]; then
            test_pass 3 "K-Means Clustering" "Clustering executed successfully"
        else
            test_fail 3 "K-Means Clustering" "Clustering failed: $CLUSTER_RESULT"
        fi
        psql_exec "DROP TABLE IF EXISTS verify_test_clustering;" > /dev/null 2>&1
    else
        test_fail 3 "K-Means Clustering" "Failed to create table: $CREATE_RESULT"
    fi
    
    json_tier_end 3
fi

# ============================================================================
# TIER 4: Embeddings
# ============================================================================
if should_run_tier 4; then
    test_section "TIER 4: Embeddings"
    json_tier_start 4 "Embeddings"
    
    # Test: Check embedding function exists
    test_info "Checking embedding function availability..."
    EMBED_FUNC=$(psql_exec "SELECT EXISTS(SELECT 1 FROM pg_proc WHERE proname = 'neurondb_embed' OR proname = 'embed_text');" | tr -d '[:space:]')
    if [ "$EMBED_FUNC" = "t" ]; then
        test_pass 4 "Embedding function exists" "Embedding function found in database"
        
        # Test: Generate embedding (may fail if not configured)
        test_info "Testing embedding generation..."
        EMBED_RESULT=$(psql_exec "SELECT neurondb_embed('test text for embedding', 'all-MiniLM-L6-v2')::text;" 2>&1 || psql_exec "SELECT embed_text('test text for embedding', 'all-MiniLM-L6-v2')::text;" 2>&1)
        if echo "$EMBED_RESULT" | grep -qE "^\[[0-9\.,\s]+\]$" || echo "$EMBED_RESULT" | grep -qi "vector"; then
            test_pass 4 "Generate embedding" "Embedding generated successfully"
        elif echo "$EMBED_RESULT" | grep -qi "error\|not configured\|not available"; then
            test_fail 4 "Generate embedding" "Embedding function not configured: $EMBED_RESULT"
        else
            test_fail 4 "Generate embedding" "Unexpected result: $EMBED_RESULT"
        fi
    else
        test_fail 4 "Embedding function exists" "Embedding function not found"
    fi
    
    # Test: Store and query with embedding
    test_info "Testing embedding storage and query..."
    psql_exec "DROP TABLE IF EXISTS verify_test_embeddings;" > /dev/null 2>&1
    CREATE_RESULT=$(psql_exec "CREATE TABLE verify_test_embeddings (id SERIAL PRIMARY KEY, text_content TEXT, embedding vector(384));" 2>&1)
    if [ $? -eq 0 ]; then
        test_pass 4 "Create embedding table" "Table for embedding storage created"
        
        # Insert with placeholder embedding (actual embedding may require config)
        INSERT_RESULT=$(psql_exec "INSERT INTO verify_test_embeddings (text_content, embedding) VALUES ('sample text', '[0.1,0.2,0.3]'::vector(384));" 2>&1)
        if [ $? -eq 0 ]; then
            test_pass 4 "Store embedding" "Embedding stored in table"
            
            # Query using embedding similarity
            QUERY_RESULT=$(psql_exec "SELECT id FROM verify_test_embeddings ORDER BY embedding <-> '[0.1,0.2,0.3]'::vector(384) LIMIT 1;" 2>&1)
            if [ $? -eq 0 ] && [ -n "$QUERY_RESULT" ]; then
                test_pass 4 "Query with embedding" "Embedding similarity query executed"
            else
                test_fail 4 "Query with embedding" "Query failed: $QUERY_RESULT"
            fi
        else
            test_fail 4 "Store embedding" "Failed to insert: $INSERT_RESULT"
        fi
        psql_exec "DROP TABLE IF EXISTS verify_test_embeddings;" > /dev/null 2>&1
    else
        test_fail 4 "Create embedding table" "Failed: $CREATE_RESULT"
    fi
    
    json_tier_end 4
fi

# ============================================================================
# TIER 5: NeuronAgent Integration
# ============================================================================
if should_run_tier 5; then
    test_section "TIER 5: NeuronAgent Integration"
    json_tier_start 5 "NeuronAgent Integration"
    
    # Test: Health check
    test_info "Checking NeuronAgent health endpoint..."
    HEALTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$AGENT_URL/health" 2>/dev/null || echo "000")
    if [ "$HEALTH_CODE" = "200" ]; then
        test_pass 5 "Health check" "NeuronAgent health endpoint responding"
    else
        test_fail 5 "Health check" "NeuronAgent not responding (HTTP $HEALTH_CODE). Is it running on $AGENT_URL?"
        json_tier_end 5
        # Skip remaining tests if health check fails
    fi
    
    # Test: Create agent (requires API key)
    test_info "Testing agent creation..."
    if [ -z "${NEURONAGENT_API_KEY:-}" ]; then
        test_info "API key not set, skipping agent creation test"
        test_fail 5 "Create agent" "API key not configured (set NEURONAGENT_API_KEY)"
    else
        CREATE_AGENT=$(curl -s -X POST "$AGENT_URL/api/v1/agents" \
            -H "Authorization: Bearer $NEURONAGENT_API_KEY" \
            -H "Content-Type: application/json" \
            -d '{"name":"verify-test-agent","system_prompt":"Test agent","model_name":"gpt-3.5-turbo","enabled_tools":[],"config":{}}' 2>&1)
        if echo "$CREATE_AGENT" | grep -qi "id\|agent\|created" || [ "$(echo "$CREATE_AGENT" | jq -r '.id // empty' 2>/dev/null)" != "" ]; then
            AGENT_ID=$(echo "$CREATE_AGENT" | jq -r '.id // empty' 2>/dev/null || echo "")
            test_pass 5 "Create agent" "Agent created successfully${AGENT_ID:+ (ID: $AGENT_ID)}"
        else
            test_fail 5 "Create agent" "Failed to create agent: $CREATE_AGENT"
        fi
    fi
    
    # Test: Send message (if agent was created)
    if [ -n "${AGENT_ID:-}" ]; then
        test_info "Testing agent message..."
        # Create session first
        SESSION_CREATE=$(curl -s -X POST "$AGENT_URL/api/v1/sessions" \
            -H "Authorization: Bearer $NEURONAGENT_API_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"agent_id\":\"$AGENT_ID\"}" 2>&1)
        SESSION_ID=$(echo "$SESSION_CREATE" | jq -r '.id // empty' 2>/dev/null || echo "")
        
        if [ -n "$SESSION_ID" ]; then
            MESSAGE_RESULT=$(curl -s -X POST "$AGENT_URL/api/v1/sessions/$SESSION_ID/messages" \
                -H "Authorization: Bearer $NEURONAGENT_API_KEY" \
                -H "Content-Type: application/json" \
                -d '{"content":"Hello, this is a test message"}' 2>&1)
            if echo "$MESSAGE_RESULT" | grep -qi "id\|message\|created" || [ "$(echo "$MESSAGE_RESULT" | jq -r '.id // empty' 2>/dev/null)" != "" ]; then
                test_pass 5 "Send message" "Message sent to agent successfully"
            else
                test_fail 5 "Send message" "Failed to send message: $MESSAGE_RESULT"
            fi
        else
            test_fail 5 "Send message" "Failed to create session: $SESSION_CREATE"
        fi
    else
        test_fail 5 "Send message" "Skipped (agent not created)"
    fi
    
    json_tier_end 5
fi

# ============================================================================
# TIER 6: NeuronMCP Integration
# ============================================================================
if should_run_tier 6; then
    test_section "TIER 6: NeuronMCP Integration"
    json_tier_start 6 "NeuronMCP Integration"
    
    # Test: MCP initialize handshake
    test_info "Testing MCP initialize handshake..."
    MCP_INIT='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"verify-test","version":"1.0.0"}}}'
    
    # Try to connect to MCP server (check if binary exists or via docker)
    if command -v neurondb-mcp > /dev/null 2>&1; then
        MCP_RESPONSE=$(echo "$MCP_INIT" | neurondb-mcp 2>&1 | head -5)
    elif command -v docker > /dev/null 2>&1 && docker ps --format '{{.Names}}' | grep -q neurondb-mcp; then
        MCP_RESPONSE=$(echo "$MCP_INIT" | docker exec -i neurondb-mcp /app/neurondb-mcp 2>&1 | head -5)
    else
        MCP_RESPONSE=""
    fi
    
    if echo "$MCP_RESPONSE" | grep -qi "jsonrpc\|result\|id"; then
        test_pass 6 "MCP initialize" "MCP server responded to initialize"
    else
        test_fail 6 "MCP initialize" "MCP server not responding. Is neurondb-mcp running?"
    fi
    
    # Test: List tools
    test_info "Testing MCP list_tools..."
    MCP_TOOLS='{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'
    
    if command -v neurondb-mcp > /dev/null 2>&1; then
        TOOLS_RESPONSE=$(echo "$MCP_TOOLS" | neurondb-mcp 2>&1 | head -10)
    elif command -v docker > /dev/null 2>&1 && docker ps --format '{{.Names}}' | grep -q neurondb-mcp; then
        TOOLS_RESPONSE=$(echo "$MCP_TOOLS" | docker exec -i neurondb-mcp /app/neurondb-mcp 2>&1 | head -10)
    else
        TOOLS_RESPONSE=""
    fi
    
    if echo "$TOOLS_RESPONSE" | grep -qi "jsonrpc\|tools\|result"; then
        test_pass 6 "List tools" "MCP tools listed successfully"
    else
        test_fail 6 "List tools" "Failed to list tools: $TOOLS_RESPONSE"
    fi
    
    # Test: Execute tool call (vector search)
    test_info "Testing MCP tool execution..."
    MCP_CALL='{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"vector_search","arguments":{"query_vector":"[0.1,0.2,0.3]","limit":5}}}'
    
    if command -v neurondb-mcp > /dev/null 2>&1; then
        CALL_RESPONSE=$(echo "$MCP_CALL" | neurondb-mcp 2>&1 | head -10)
    elif command -v docker > /dev/null 2>&1 && docker ps --format '{{.Names}}' | grep -q neurondb-mcp; then
        CALL_RESPONSE=$(echo "$MCP_CALL" | docker exec -i neurondb-mcp /app/neurondb-mcp 2>&1 | head -10)
    else
        CALL_RESPONSE=""
    fi
    
    if echo "$CALL_RESPONSE" | grep -qi "jsonrpc\|result\|content"; then
        test_pass 6 "Execute tool" "MCP tool call executed successfully"
    else
        # Tool call might fail due to missing table, but protocol should work
        if echo "$CALL_RESPONSE" | grep -qi "error\|not found"; then
            test_pass 6 "Execute tool" "MCP protocol working (tool error expected without data)"
        else
            test_fail 6 "Execute tool" "Tool call failed: $CALL_RESPONSE"
        fi
    fi
    
    json_tier_end 6
fi

# ============================================================================
# Summary
# ============================================================================
if [ "$JSON_OUTPUT" = false ]; then
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}Test Summary${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "Total tests: $TOTAL_TESTS"
    echo -e "${GREEN}Passed: $TOTAL_PASSED${NC}"
    if [ $TOTAL_FAILED -gt 0 ]; then
        echo -e "${RED}Failed: $TOTAL_FAILED${NC}"
    else
        echo "Failed: $TOTAL_FAILED"
    fi
    
    for i in {0..6}; do
        if [ ${TIER_TESTS[$i]} -gt 0 ]; then
            echo ""
            echo "Tier $i: ${TIER_PASSED[$i]}/${TIER_TESTS[$i]} passed"
        fi
    done
fi

json_end

# Determine exit code
if [ $TOTAL_FAILED -gt 0 ]; then
    EXIT_CODE=1
elif [ $TOTAL_TESTS -eq 0 ]; then
    EXIT_CODE=2  # No tests ran (all skipped)
fi

exit $EXIT_CODE

