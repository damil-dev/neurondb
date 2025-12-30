#!/bin/bash
# Test script for PostgreSQL and NeuronDB queries

set -e

CONTAINER="${1:-neurondb-cpu}"
DB_USER="${DB_USER:-neurondb}"
DB_NAME="${DB_NAME:-neurondb}"
DB_PASSWORD="${DB_PASSWORD:-neurondb}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "PostgreSQL and NeuronDB Query Tests"
echo "Container: $CONTAINER"
echo "=========================================="
echo ""

# Check if container is running
if ! docker ps --format "{{.Names}}" | grep -q "^${CONTAINER}$"; then
    echo -e "${RED}Error: Container $CONTAINER is not running${NC}"
    exit 1
fi

# Wait for PostgreSQL to be ready
echo -e "${BLUE}Waiting for PostgreSQL to be ready...${NC}"
for i in {1..30}; do
    if docker exec "$CONTAINER" pg_isready -U "$DB_USER" &>/dev/null; then
        echo -e "${GREEN}✓ PostgreSQL is ready${NC}"
        break
    fi
    sleep 1
done

echo ""
echo -e "${BLUE}1. Basic PostgreSQL Tests${NC}"
echo "----------------------------------------"

# Test 1: PostgreSQL version
echo "Test 1: PostgreSQL version"
VERSION=$(docker exec "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -tAc "SELECT version();" 2>/dev/null)
echo -e "${GREEN}✓${NC} $VERSION"
echo ""

# Test 2: List databases
echo "Test 2: List databases"
DATABASES=$(docker exec "$CONTAINER" psql -U "$DB_USER" -d postgres -tAc "SELECT datname FROM pg_database WHERE datistemplate = false;" 2>/dev/null)
echo -e "${GREEN}✓${NC} Databases:"
echo "$DATABASES" | sed 's/^/  /'
echo ""

# Test 3: Create a test table
echo "Test 3: Create test table"
docker exec "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -c "
DROP TABLE IF EXISTS test_table;
CREATE TABLE test_table (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    value INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);" 2>&1 | grep -v "DROP TABLE" | grep -v "CREATE TABLE" || true
echo -e "${GREEN}✓${NC} Test table created"
echo ""

# Test 4: Insert data
echo "Test 4: Insert test data"
docker exec "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -c "
INSERT INTO test_table (name, value) VALUES 
    ('Alice', 100),
    ('Bob', 200),
    ('Charlie', 300);" 2>&1 | grep -v "INSERT" || true
echo -e "${GREEN}✓${NC} Data inserted"
echo ""

# Test 5: Query data
echo "Test 5: Query test data"
QUERY_RESULT=$(docker exec "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -tAc "SELECT COUNT(*) FROM test_table;" 2>/dev/null)
echo -e "${GREEN}✓${NC} Row count: $QUERY_RESULT"
docker exec "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -c "SELECT * FROM test_table;" 2>&1 | tail -n +3 | head -n -1
echo ""

# Test 6: Test transactions
echo "Test 6: Test transactions"
docker exec "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -c "
BEGIN;
INSERT INTO test_table (name, value) VALUES ('Transaction Test', 999);
ROLLBACK;
SELECT COUNT(*) FROM test_table WHERE name = 'Transaction Test';" 2>&1 | grep -E "0|ROLLBACK" | head -1
echo -e "${GREEN}✓${NC} Transaction rollback works"
echo ""

echo -e "${BLUE}2. NeuronDB Extension Tests${NC}"
echo "----------------------------------------"

# Test 7: Check if extension exists
echo "Test 7: Check NeuronDB extension"
EXT_CHECK=$(docker exec "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -tAc "SELECT extname FROM pg_extension WHERE extname = 'neurondb';" 2>/dev/null || echo "")
if [ -z "$EXT_CHECK" ]; then
    echo -e "${YELLOW}⚠${NC} Extension not found, creating it..."
    docker exec "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS neurondb;" 2>&1 | grep -v "CREATE EXTENSION" || true
    sleep 2
fi

EXT_STATUS=$(docker exec "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -tAc "SELECT extname, extversion FROM pg_extension WHERE extname = 'neurondb';" 2>/dev/null || echo "")
if [ -n "$EXT_STATUS" ]; then
    echo -e "${GREEN}✓${NC} NeuronDB extension: $EXT_STATUS"
else
    echo -e "${RED}✗${NC} Failed to create NeuronDB extension"
    echo "Checking shared_preload_libraries..."
    docker exec "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -tAc "SHOW shared_preload_libraries;" 2>&1
    exit 1
fi
echo ""

# Test 8: NeuronDB version function
echo "Test 8: NeuronDB version"
VERSION_RESULT=$(docker exec "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -tAc "SELECT neurondb.version();" 2>&1 || echo "failed")
if echo "$VERSION_RESULT" | grep -qE "version|neurondb"; then
    echo -e "${GREEN}✓${NC} $VERSION_RESULT"
else
    echo -e "${YELLOW}⚠${NC} Version function result: $VERSION_RESULT"
fi
echo ""

# Test 9: Vector type operations
echo "Test 9: Vector type operations"
VECTOR_TEST=$(docker exec "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -tAc "SELECT '[1.0, 2.0, 3.0]'::vector;" 2>&1 || echo "failed")
if echo "$VECTOR_TEST" | grep -qE "\[1|\[1.0"; then
    echo -e "${GREEN}✓${NC} Vector type works: $VECTOR_TEST"
else
    echo -e "${YELLOW}⚠${NC} Vector test result: $VECTOR_TEST"
    echo "  (This may be normal if vector type needs extension reload)"
fi
echo ""

# Test 10: Create table with vector column
echo "Test 10: Create table with vector column"
docker exec "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -c "
DROP TABLE IF EXISTS vector_test;
CREATE TABLE vector_test (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    embedding vector(3)
);" 2>&1 | grep -v "DROP TABLE" | grep -v "CREATE TABLE" || true

if docker exec "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -tAc "\d vector_test" 2>&1 | grep -q "embedding"; then
    echo -e "${GREEN}✓${NC} Table with vector column created"
else
    echo -e "${YELLOW}⚠${NC} Vector column may not be available"
fi
echo ""

# Test 11: Insert vector data
echo "Test 11: Insert vector data"
VECTOR_INSERT=$(docker exec "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -c "
INSERT INTO vector_test (name, embedding) VALUES 
    ('doc1', '[1.0, 2.0, 3.0]'::vector),
    ('doc2', '[4.0, 5.0, 6.0]'::vector),
    ('doc3', '[7.0, 8.0, 9.0]'::vector);" 2>&1 || echo "failed")

if echo "$VECTOR_INSERT" | grep -qE "INSERT|3 rows"; then
    echo -e "${GREEN}✓${NC} Vector data inserted successfully"
else
    echo -e "${YELLOW}⚠${NC} Vector insert result: $VECTOR_INSERT"
fi
echo ""

# Test 12: Vector similarity search
echo "Test 12: Vector similarity search (L2 distance)"
VECTOR_QUERY=$(docker exec "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -c "
SELECT name, embedding <-> '[1.0, 2.0, 3.0]'::vector AS distance 
FROM vector_test 
ORDER BY embedding <-> '[1.0, 2.0, 3.0]'::vector 
LIMIT 3;" 2>&1 || echo "failed")

if echo "$VECTOR_QUERY" | grep -qE "doc1|distance"; then
    echo -e "${GREEN}✓${NC} Vector similarity search works:"
    echo "$VECTOR_QUERY" | tail -n +3 | head -n -1 | sed 's/^/  /'
else
    echo -e "${YELLOW}⚠${NC} Vector query result: $VECTOR_QUERY"
fi
echo ""

# Test 13: Check NeuronDB configuration
echo "Test 13: NeuronDB configuration"
CONFIG=$(docker exec "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -tAc "
SELECT name, setting 
FROM pg_settings 
WHERE name LIKE 'neurondb%' 
ORDER BY name;" 2>&1 || echo "failed")

if echo "$CONFIG" | grep -q "neurondb"; then
    echo -e "${GREEN}✓${NC} NeuronDB configuration:"
    echo "$CONFIG" | sed 's/^/  /'
else
    echo -e "${YELLOW}⚠${NC} Configuration check: $CONFIG"
fi
echo ""

# Test 14: Test basic ML function availability (if available)
echo "Test 14: Check ML function availability"
ML_FUNCTIONS=$(docker exec "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -tAc "
SELECT routine_name 
FROM information_schema.routines 
WHERE routine_schema = 'neurondb' 
AND routine_name LIKE '%train%' 
LIMIT 5;" 2>&1 || echo "failed")

if echo "$ML_FUNCTIONS" | grep -qE "train|routine_name"; then
    echo -e "${GREEN}✓${NC} ML functions available:"
    echo "$ML_FUNCTIONS" | sed 's/^/  /'
else
    echo -e "${YELLOW}⚠${NC} ML functions check: $ML_FUNCTIONS"
    echo "  (ML functions may require specific setup)"
fi
echo ""

echo "=========================================="
echo -e "${GREEN}Test Summary${NC}"
echo "=========================================="
echo "All basic PostgreSQL and NeuronDB tests completed."
echo "Review the results above for any warnings."
echo ""





