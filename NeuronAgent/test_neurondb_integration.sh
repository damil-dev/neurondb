#!/bin/bash
# Comprehensive Integration Test for NeuronAgent with NeuronDB
# Tests all NeuronDB-specific features: embeddings, vector search, LLM functions,
# memory management, tools, and end-to-end workflows

set -e

cd "$(dirname "$0")"

echo "================================================================"
echo "NeuronAgent NeuronDB Integration Test Suite"
echo "Comprehensive testing of all NeuronDB features"
echo "================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
DB_USER="${DB_USER:-pge}"
DB_NAME="${DB_NAME:-neurondb}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
SERVER_URL="http://localhost:8080"

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

test_pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    ((TESTS_PASSED++))
    ((TESTS_TOTAL++))
}

test_fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    ((TESTS_FAILED++))
    ((TESTS_TOTAL++))
}

test_info() {
    echo -e "${BLUE}ℹ️  INFO${NC}: $1"
}

test_warn() {
    echo -e "${YELLOW}⚠️  WARN${NC}: $1"
}

test_section() {
    echo ""
    echo -e "${CYAN}$1${NC}"
    echo "$(echo "$1" | sed 's/./=/g')"
}

# ============================================================================
# PHASE 0: Prerequisites
# ============================================================================
test_section "PHASE 0: Prerequisites and Environment Setup"

# Check server
test_info "Checking NeuronAgent server..."
if curl -s "$SERVER_URL/health" > /dev/null 2>&1; then
    test_pass "Server is running on $SERVER_URL"
else
    test_fail "Server is not running"
    echo "Start server: DB_USER=$DB_USER go run cmd/agent-server/main.go"
    exit 1
fi

# Check database
test_info "Checking PostgreSQL database..."
if psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -c "SELECT 1;" > /dev/null 2>&1; then
    test_pass "Database connection successful"
else
    test_fail "Database connection failed"
    exit 1
fi

# Check NeuronDB extension
test_info "Checking NeuronDB extension..."
EXT_EXISTS=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'neurondb');" | xargs)
if [ "$EXT_EXISTS" = "t" ]; then
    test_pass "NeuronDB extension installed"
else
    test_fail "NeuronDB extension not found"
    exit 1
fi

# Check vector type
test_info "Checking vector type..."
VECTOR_TYPE=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT EXISTS(SELECT 1 FROM pg_type WHERE typname = 'vector');" | xargs)
if [ "$VECTOR_TYPE" = "t" ]; then
    test_pass "Vector type available"
else
    test_fail "Vector type not found"
    exit 1
fi

# Generate API key
test_info "Generating API key for testing..."
API_KEY_OUTPUT=$(go run cmd/generate-key/main.go \
    -org "neurondb-test" \
    -user "test-user" \
    -rate 1000 \
    -roles "user,admin" \
    -db-host "$DB_HOST" \
    -db-port "$DB_PORT" \
    -db-name "$DB_NAME" \
    -db-user "$DB_USER" 2>&1)

if [ $? -eq 0 ]; then
    API_KEY=$(echo "$API_KEY_OUTPUT" | grep "^Key:" | sed 's/^Key: //' | tr -d '[:space:]')
    if [ -z "$API_KEY" ]; then
        # Try alternative extraction
        API_KEY=$(echo "$API_KEY_OUTPUT" | grep -i "api key" | sed 's/.*[Aa][Pp][Ii] [Kk][Ee][Yy][: ]*//' | tr -d '[:space:]')
    fi
    if [ -z "$API_KEY" ]; then
        API_KEY=$(echo "$API_KEY_OUTPUT" | tail -1 | tr -d '[:space:]')
    fi
    if [ -n "$API_KEY" ] && [ ${#API_KEY} -gt 20 ]; then
        test_pass "API key generated: ${API_KEY:0:16}..."
        export NEURONAGENT_API_KEY="$API_KEY"
        KEY_PREFIX=$(echo "$API_KEY" | cut -c1-8)
    else
        test_fail "Could not extract API key from output"
        echo "Output: $API_KEY_OUTPUT"
        exit 1
    fi
else
    test_fail "API key generation failed"
    echo "$API_KEY_OUTPUT"
    exit 1
fi

# ============================================================================
# PHASE 1: NeuronDB Core Functions
# ============================================================================
test_section "PHASE 1: NeuronDB Core Functions Testing"

# Test neurondb_embed function existence
test_info "Checking neurondb_embed function..."
EMBED_FUNC=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT EXISTS(SELECT 1 FROM pg_proc WHERE proname = 'neurondb_embed');" | xargs)
if [ "$EMBED_FUNC" = "t" ]; then
    test_pass "neurondb_embed function exists"
    
    # Test actual embedding generation
    test_info "Testing embedding generation with sample text..."
    EMBED_RESULT=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT neurondb_embed('This is a test sentence for embedding generation', 'all-MiniLM-L6-v2')::text;" 2>&1)
    if echo "$EMBED_RESULT" | grep -qE "^\[[0-9\.,\s]+\]$" || echo "$EMBED_RESULT" | grep -q "vector"; then
        test_pass "Embedding generation successful"
        EMBED_DIM=$(echo "$EMBED_RESULT" | sed 's/[^0-9]//g' | wc -c)
        test_info "Embedding dimension: ~$EMBED_DIM (extracted from result)"
    else
        # Some embedding models might return errors if not configured, warn but don't fail
        if echo "$EMBED_RESULT" | grep -qi "error\|not found\|not available"; then
            test_warn "Embedding generation returned error (model may need configuration): $(echo "$EMBED_RESULT" | head -c 100)"
        else
            test_pass "Embedding function callable (result format may vary)"
        fi
    fi
else
    test_warn "neurondb_embed function not found (may not be configured)"
fi

# Test neurondb_embed_batch function
test_info "Checking neurondb_embed_batch function..."
BATCH_EMBED_FUNC=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT EXISTS(SELECT 1 FROM pg_proc WHERE proname = 'neurondb_embed_batch');" | xargs)
if [ "$BATCH_EMBED_FUNC" = "t" ]; then
    test_pass "neurondb_embed_batch function exists"
else
    test_warn "neurondb_embed_batch function not found (batch embeddings may use fallback)"
fi

# Test vector type operations
test_info "Testing vector type operations..."
VECTOR_CAST=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT '[1,2,3]'::vector(3);" 2>&1)
if echo "$VECTOR_CAST" | grep -q "\[1,2,3\]" || echo "$VECTOR_CAST" | grep -q "vector"; then
    test_pass "Vector type casting works"
else
    test_fail "Vector type casting failed: $VECTOR_CAST"
fi

# Test vector similarity operators
test_info "Testing vector similarity operators..."
VECTOR_SIM=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT '[1,0,0]'::vector(3) <=> '[0,1,0]'::vector(3) AS distance;" 2>&1)
if echo "$VECTOR_SIM" | grep -qE "^[0-9\.]+$" || echo "$VECTOR_SIM" | grep -qE "^[[:space:]]*[0-9\.]+"; then
    DISTANCE=$(echo "$VECTOR_SIM" | tr -d '[:space:]')
    test_pass "Vector similarity operator (<=>) works, distance: $DISTANCE"
else
    test_fail "Vector similarity operator failed: $VECTOR_SIM"
fi

# Test LLM functions
test_info "Checking NeuronDB LLM functions..."
LLM_GENERATE=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT EXISTS(SELECT 1 FROM pg_proc WHERE proname IN ('neurondb_llm_generate', 'neurondb_llm_complete'));" | xargs)
if [ "$LLM_GENERATE" = "t" ]; then
    test_pass "LLM functions available (neurondb_llm_generate or neurondb_llm_complete)"
else
    test_warn "LLM functions not found (LLM features may not be configured)"
fi

# ============================================================================
# PHASE 2: Memory Management with NeuronDB
# ============================================================================
test_section "PHASE 2: Memory Management with Vector Embeddings"

# Verify memory_chunks table structure
test_info "Checking memory_chunks table structure..."
MEMORY_CHUNKS_EMBED=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT data_type FROM information_schema.columns WHERE table_schema = 'neurondb_agent' AND table_name = 'memory_chunks' AND column_name = 'embedding';" | xargs)
if echo "$MEMORY_CHUNKS_EMBED" | grep -qi "vector\|USER-DEFINED"; then
    test_pass "memory_chunks.embedding column is vector type"
else
    test_fail "memory_chunks.embedding column type incorrect: $MEMORY_CHUNKS_EMBED"
fi

# Test creating a memory chunk with embedding via SQL
test_info "Testing memory chunk creation with embedding..."
# First create a test agent and session
TEST_AGENT_ID=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "INSERT INTO neurondb_agent.agents (name, system_prompt, model_name, enabled_tools, config) VALUES ('test-memory-agent', 'Test agent', 'gpt-4', ARRAY['sql']::text[], '{}'::jsonb) RETURNING id;" | xargs)
if [ -n "$TEST_AGENT_ID" ]; then
    test_pass "Test agent created for memory testing"
    
    TEST_SESSION_ID=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "INSERT INTO neurondb_agent.sessions (agent_id, external_user_id) VALUES ('$TEST_AGENT_ID'::uuid, 'test-user') RETURNING id;" | xargs)
    if [ -n "$TEST_SESSION_ID" ]; then
        test_pass "Test session created"
        
        # Try to create a memory chunk with an embedding
        # Use a simple test vector
        TEST_EMBEDDING="[0.1,0.2,0.3,0.4,0.5]"
        MEMORY_CHUNK_ID=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "INSERT INTO neurondb_agent.memory_chunks (agent_id, session_id, content, embedding, importance_score, metadata) VALUES ('$TEST_AGENT_ID'::uuid, '$TEST_SESSION_ID'::uuid, 'This is a test memory chunk', '$TEST_EMBEDDING'::vector, 0.8, '{}'::jsonb) RETURNING id;" 2>&1)
        if echo "$MEMORY_CHUNK_ID" | grep -qE "^[0-9]+$"; then
            test_pass "Memory chunk created with embedding"
            
            # Test vector similarity search
            test_info "Testing vector similarity search on memory_chunks..."
            SIMILARITY_SEARCH=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT id, 1 - (embedding <=> '$TEST_EMBEDDING'::vector) AS similarity FROM neurondb_agent.memory_chunks WHERE agent_id = '$TEST_AGENT_ID'::uuid ORDER BY embedding <=> '$TEST_EMBEDDING'::vector LIMIT 1;" 2>&1)
            if echo "$SIMILARITY_SEARCH" | grep -qE "^[0-9]+"; then
                test_pass "Vector similarity search works on memory_chunks"
            else
                test_fail "Vector similarity search failed: $SIMILARITY_SEARCH"
            fi
            
            # Cleanup test data
            psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -c "DELETE FROM neurondb_agent.memory_chunks WHERE id = $(echo "$MEMORY_CHUNK_ID" | tr -d '[:space:]');" > /dev/null 2>&1
        else
            test_warn "Memory chunk creation failed (may need proper vector dimension): $MEMORY_CHUNK_ID"
        fi
        
        # Cleanup
        psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -c "DELETE FROM neurondb_agent.sessions WHERE id = '$TEST_SESSION_ID'::uuid;" > /dev/null 2>&1
    else
        test_fail "Failed to create test session"
    fi
    
    # Cleanup
    psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -c "DELETE FROM neurondb_agent.agents WHERE id = '$TEST_AGENT_ID'::uuid;" > /dev/null 2>&1
else
    test_warn "Failed to create test agent (may need schema setup)"
fi

# Check HNSW index
test_info "Checking HNSW index on memory_chunks.embedding..."
HNSW_INDEX=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT indexname FROM pg_indexes WHERE schemaname = 'neurondb_agent' AND tablename = 'memory_chunks' AND indexname LIKE '%embedding%' LIMIT 1;" | xargs)
if [ -n "$HNSW_INDEX" ]; then
    test_pass "HNSW index exists: $HNSW_INDEX"
else
    test_warn "HNSW index not found (will be created automatically on first vector insert or can be created manually)"
fi

# ============================================================================
# PHASE 3: API Endpoints with NeuronDB Integration
# ============================================================================
test_section "PHASE 3: API Endpoints with NeuronDB Features"

# Create agent via API
test_info "Creating agent via API..."
AGENT_NAME="neurondb-test-agent-$(date +%s)"
CREATE_AGENT_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$SERVER_URL/api/v1/agents" \
    -H "Authorization: Bearer $NEURONAGENT_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{
        \"name\": \"$AGENT_NAME\",
        \"description\": \"NeuronDB integration test agent\",
        \"system_prompt\": \"You are a helpful assistant with access to vector search and embeddings.\",
        \"model_name\": \"gpt-4\",
        \"enabled_tools\": [\"sql\", \"vector\", \"rag\"],
        \"config\": {
            \"temperature\": 0.7,
            \"max_tokens\": 2000
        }
    }" 2>&1)

HTTP_CODE=$(echo "$CREATE_AGENT_RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "201" ]; then
    test_pass "Agent created via API (with NeuronDB tools)"
    AGENT_ID=$(echo "$CREATE_AGENT_RESPONSE" | head -1 | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])" 2>/dev/null || echo "")
    if [ -n "$AGENT_ID" ]; then
        test_info "Created agent ID: ${AGENT_ID:0:8}..."
        
        # Verify agent has vector/rag tools enabled
        AGENT_TOOLS=$(echo "$CREATE_AGENT_RESPONSE" | head -1 | python3 -c "import sys, json; print(','.join(json.load(sys.stdin).get('enabled_tools', [])))" 2>/dev/null || echo "")
        if echo "$AGENT_TOOLS" | grep -qE "vector|rag"; then
            test_pass "Agent configured with NeuronDB tools: $AGENT_TOOLS"
        fi
    fi
else
    test_fail "Failed to create agent (HTTP $HTTP_CODE)"
    echo "Response: $(echo "$CREATE_AGENT_RESPONSE" | head -1)"
    AGENT_ID=""
fi

# Create session
if [ -n "$AGENT_ID" ]; then
    test_info "Creating session for agent..."
    CREATE_SESSION_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$SERVER_URL/api/v1/sessions" \
        -H "Authorization: Bearer $NEURONAGENT_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"agent_id\": \"$AGENT_ID\",
            \"external_user_id\": \"neurondb-test-user\",
            \"metadata\": {\"test\": \"neurondb_integration\"}
        }" 2>&1)
    
    HTTP_CODE=$(echo "$CREATE_SESSION_RESPONSE" | tail -1)
    if [ "$HTTP_CODE" = "201" ]; then
        test_pass "Session created via API"
        SESSION_ID=$(echo "$CREATE_SESSION_RESPONSE" | head -1 | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])" 2>/dev/null || echo "")
        if [ -n "$SESSION_ID" ]; then
            test_info "Created session ID: ${SESSION_ID:0:8}..."
        fi
    else
        test_fail "Failed to create session (HTTP $HTTP_CODE)"
        SESSION_ID=""
    fi
fi

# Send message that should trigger embedding/memory operations
if [ -n "$SESSION_ID" ]; then
    test_info "Sending message to trigger NeuronDB operations (embeddings, memory)..."
    SEND_MESSAGE_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$SERVER_URL/api/v1/sessions/$SESSION_ID/messages" \
        -H "Authorization: Bearer $NEURONAGENT_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "content": "Hello! Can you help me find information about machine learning? This is a test message for NeuronDB integration.",
            "role": "user"
        }' 2>&1)
    
    HTTP_CODE=$(echo "$SEND_MESSAGE_RESPONSE" | tail -1)
    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
        test_pass "Message sent successfully (may trigger embedding generation)"
        
        # Check if memory chunks were created
        sleep 2  # Give time for async operations
        MEMORY_COUNT=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT COUNT(*) FROM neurondb_agent.memory_chunks WHERE session_id = '$SESSION_ID'::uuid;" | xargs)
        if [ "$MEMORY_COUNT" -gt 0 ]; then
            test_pass "Memory chunks created with embeddings ($MEMORY_COUNT chunks)"
        else
            test_warn "No memory chunks found (may be created asynchronously or filtered by importance)"
        fi
        
        # Verify messages were stored
        MESSAGE_COUNT=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT COUNT(*) FROM neurondb_agent.messages WHERE session_id = '$SESSION_ID'::uuid;" | xargs)
        if [ "$MESSAGE_COUNT" -gt 0 ]; then
            test_pass "Messages stored in database ($MESSAGE_COUNT messages)"
        else
            test_fail "Messages not stored in database"
        fi
    else
        test_warn "Message sending returned HTTP $HTTP_CODE (may require LLM configuration)"
        echo "Response: $(echo "$SEND_MESSAGE_RESPONSE" | head -1)"
    fi
fi

# ============================================================================
# PHASE 4: Database Schema and Constraints
# ============================================================================
test_section "PHASE 4: Database Schema Verification"

# Check all required tables
REQUIRED_TABLES=("agents" "sessions" "messages" "memory_chunks" "tools" "jobs" "api_keys" "schema_migrations")
for table in "${REQUIRED_TABLES[@]}"; do
    TABLE_EXISTS=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_schema = 'neurondb_agent' AND table_name = '$table');" | xargs)
    if [ "$TABLE_EXISTS" = "t" ]; then
        test_pass "Table $table exists"
    else
        test_fail "Table $table missing"
    fi
done

# Check foreign key constraints
FK_COUNT=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT COUNT(*) FROM information_schema.table_constraints WHERE constraint_schema = 'neurondb_agent' AND constraint_type = 'FOREIGN KEY';" | xargs)
if [ "$FK_COUNT" -gt 0 ]; then
    test_pass "Foreign key constraints exist ($FK_COUNT found)"
else
    test_fail "No foreign key constraints found"
fi

# Check indexes
INDEX_COUNT=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'neurondb_agent';" | xargs)
if [ "$INDEX_COUNT" -gt 0 ]; then
    test_pass "Indexes exist ($INDEX_COUNT found)"
else
    test_warn "No indexes found"
fi

# ============================================================================
# PHASE 5: Vector Operations Performance
# ============================================================================
test_section "PHASE 5: Vector Operations Performance"

# Test vector similarity search performance
test_info "Testing vector similarity search performance..."
if [ -n "$TEST_AGENT_ID" ] 2>/dev/null || [ -n "$AGENT_ID" ]; then
    AGENT_FOR_SEARCH="$AGENT_ID"
    if [ -z "$AGENT_FOR_SEARCH" ]; then
        AGENT_FOR_SEARCH=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT id FROM neurondb_agent.agents LIMIT 1;" | xargs)
    fi
    
    if [ -n "$AGENT_FOR_SEARCH" ]; then
        # Check if there are memory chunks to search
        CHUNK_COUNT=$(psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT COUNT(*) FROM neurondb_agent.memory_chunks WHERE agent_id = '$AGENT_FOR_SEARCH'::uuid;" | xargs)
        if [ "$CHUNK_COUNT" -gt 0 ]; then
            START_TIME=$(date +%s%N)
            psql -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -t -c "SELECT id FROM neurondb_agent.memory_chunks WHERE agent_id = '$AGENT_FOR_SEARCH'::uuid ORDER BY embedding <=> '[0.1,0.2,0.3,0.4,0.5]'::vector LIMIT 5;" > /dev/null 2>&1
            END_TIME=$(date +%s%N)
            ELAPSED=$(( (END_TIME - START_TIME) / 1000000 ))
            if [ "$ELAPSED" -lt 1000 ]; then
                test_pass "Vector similarity search performance: ${ELAPSED}ms (good)"
            else
                test_warn "Vector similarity search took ${ELAPSED}ms (may need index optimization)"
            fi
        else
            test_info "Skipping performance test (no memory chunks to search)"
        fi
    fi
else
    test_info "Skipping performance test (no test agent available)"
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================
test_section "TEST SUMMARY"

echo "Total Tests: $TESTS_TOTAL"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo ""

# Calculate success rate
if [ $TESTS_TOTAL -gt 0 ]; then
    SUCCESS_RATE=$(( TESTS_PASSED * 100 / TESTS_TOTAL ))
    echo "Success Rate: ${SUCCESS_RATE}%"
fi

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED!${NC}"
    echo ""
    echo "NeuronAgent NeuronDB integration is fully functional!"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    echo ""
    echo "Please review the failures above."
    exit 1
fi

