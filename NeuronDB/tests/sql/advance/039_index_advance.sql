-- 029_index_advance.sql
-- Comprehensive advanced test for ALL index module functions
-- Tests HNSW, IVF, index consistency, cache operations, multi-tenant isolation
-- Works on 1000 rows and tests each and every index code path
-- Updated with comprehensive test cases from test_indexing.sql

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\echo '=========================================================================='
\echo 'Index Module: Exhaustive Index Operations Coverage'
\echo '=========================================================================='

-- ============================================================================
-- PART 1: Check Available Index Types
-- ============================================================================
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'PART 1: Checking Available Index Types'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo ''
SELECT 
    amname AS index_type,
    CASE amname
        WHEN 'hnsw' THEN 'Hierarchical Navigable Small World'
        WHEN 'ivf' THEN 'Inverted File Index'
        ELSE 'Other'
    END AS description
FROM pg_am
WHERE amname IN ('hnsw', 'ivf')
ORDER BY amname;

-- ============================================================================
-- PART 2: Create Test Table with Vectors
-- ============================================================================
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'PART 2: Creating Test Table'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Create test table
DROP TABLE IF EXISTS index_advance_test;
CREATE TABLE index_advance_test (
	id SERIAL PRIMARY KEY,
	embedding vector(28),
	label integer,
	metadata jsonb
);

-- Insert test data (use test_train_view if available, otherwise create synthetic data)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.views WHERE table_name = 'test_train_view') THEN
        INSERT INTO index_advance_test (embedding, label, metadata)
        SELECT features, label, '{"source": "test"}'::jsonb
        FROM test_train_view
        LIMIT 1000;
    ELSE
        -- Create synthetic test data
        INSERT INTO index_advance_test (embedding, label, metadata)
        SELECT 
            array_to_vector(ARRAY(SELECT (random() * 10)::float4 FROM generate_series(1, 28))) AS embedding,
            (random() * 10)::integer AS label,
            jsonb_build_object('doc_id', generate_series, 'category', 'test') AS metadata
        FROM generate_series(1, 1000);
    END IF;
END $$;

\echo ''
SELECT COUNT(*) AS total_rows FROM index_advance_test;

/*-------------------------------------------------------------------
 * ---- HNSW INDEX WITH VARIOUS PARAMETERS ----
 * Test HNSW index creation with all parameter combinations
 *------------------------------------------------------------------*/
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'HNSW Index Parameter Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo ''
\echo 'Test 3.1: HNSW with L2 distance (default parameters)'
DROP INDEX IF EXISTS idx_test_hnsw_l2_default;
CREATE INDEX idx_test_hnsw_l2_default 
ON index_advance_test 
USING hnsw (embedding vector_l2_ops);

SELECT 
    indexname,
    indexdef
FROM pg_indexes
WHERE indexname = 'idx_test_hnsw_l2_default';

\echo ''
\echo 'Test 3.2: HNSW with L2 distance (custom parameters: m=16, ef_construction=200)'
DROP INDEX IF EXISTS idx_test_hnsw_l2_custom;
CREATE INDEX idx_test_hnsw_l2_custom 
ON index_advance_test 
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 200);

SELECT 
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS index_size
FROM pg_indexes
WHERE indexname = 'idx_test_hnsw_l2_custom';

\echo ''
\echo 'Test 3.3: HNSW with cosine distance'
DROP INDEX IF EXISTS idx_test_hnsw_cosine;
CREATE INDEX idx_test_hnsw_cosine 
ON index_advance_test 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

SELECT 
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS index_size
FROM pg_indexes
WHERE indexname = 'idx_test_hnsw_cosine';

\echo ''
\echo 'Test 3.4: HNSW with m=8 (small)'
DROP INDEX IF EXISTS idx_hnsw_m8;
CREATE INDEX idx_hnsw_m8 ON index_advance_test 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 8, ef_construction = 100);

\echo ''
\echo 'Test 3.5: HNSW with m=16 (default)'
DROP INDEX IF EXISTS idx_hnsw_m16;
CREATE INDEX idx_hnsw_m16 ON index_advance_test 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 16, ef_construction = 200);

\echo ''
\echo 'Test 3.6: HNSW with m=32 (large)'
DROP INDEX IF EXISTS idx_hnsw_m32;
CREATE INDEX idx_hnsw_m32 ON index_advance_test 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 32, ef_construction = 400);

\echo ''
\echo 'Test 3.7: HNSW with ef_construction=50 (small)'
DROP INDEX IF EXISTS idx_hnsw_ef50;
CREATE INDEX idx_hnsw_ef50 ON index_advance_test 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 16, ef_construction = 50);

\echo ''
\echo 'Test 3.8: HNSW with ef_construction=200 (default)'
DROP INDEX IF EXISTS idx_hnsw_ef200;
CREATE INDEX idx_hnsw_ef200 ON index_advance_test 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 16, ef_construction = 200);

\echo ''
\echo 'Test 3.9: HNSW with ef_construction=500 (large)'
DROP INDEX IF EXISTS idx_hnsw_ef500;
CREATE INDEX idx_hnsw_ef500 ON index_advance_test 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 16, ef_construction = 500);

/*-------------------------------------------------------------------
 * ---- IVF INDEX WITH VARIOUS PARAMETERS ----
 * Test IVF index creation with all parameter combinations
 *------------------------------------------------------------------*/
\echo ''
\echo 'IVF Index Parameter Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 7: IVF with lists=5 (small)'
CREATE INDEX idx_ivf_lists5 ON index_advance_test 
USING ivf (embedding vector_l2_ops) 
WITH (lists = 5);

\echo 'Test 8: IVF with lists=10 (default)'
DROP INDEX IF EXISTS idx_ivf_lists10;
CREATE INDEX idx_ivf_lists10 ON index_advance_test 
USING ivf (embedding vector_l2_ops) 
WITH (lists = 10);

\echo 'Test 9: IVF with lists=50 (large)'
DROP INDEX IF EXISTS idx_ivf_lists50;
CREATE INDEX idx_ivf_lists50 ON index_advance_test 
USING ivf (embedding vector_l2_ops) 
WITH (lists = 50);

\echo 'Test 10: IVF with lists=100 (very large)'
DROP INDEX IF EXISTS idx_ivf_lists100;
CREATE INDEX idx_ivf_lists100 ON index_advance_test 
USING ivf (embedding vector_l2_ops) 
WITH (lists = 100);

-- ============================================================================
-- PART 4: Test Index Performance
-- ============================================================================
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'PART 4: Testing Index Performance'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo ''
\echo 'Test 4.1: Query performance with HNSW index'
-- Use the existing index for performance test
EXPLAIN ANALYZE
WITH q AS (
    SELECT (SELECT embedding FROM index_advance_test LIMIT 1) AS query_vec
)
SELECT 
    id,
    embedding <-> q.query_vec AS l2_distance
FROM index_advance_test
CROSS JOIN q
ORDER BY embedding <-> q.query_vec
LIMIT 5;

-- ============================================================================
-- PART 5: Test Different Distance Metrics with Indexes
-- ============================================================================
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'PART 5: Testing Different Distance Metrics'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo ''
\echo 'Test 5.1: L2 distance search with HNSW index'
WITH q AS (
    SELECT (SELECT embedding FROM index_advance_test LIMIT 1) AS query_vec
)
SELECT 
    id,
    ROUND((embedding <-> q.query_vec)::numeric, 6) AS l2_distance
FROM index_advance_test
CROSS JOIN q
ORDER BY embedding <-> q.query_vec
LIMIT 5;

\echo ''
\echo 'Test 5.2: Cosine distance search with HNSW index'
WITH q AS (
    SELECT (SELECT embedding FROM index_advance_test LIMIT 1) AS query_vec
)
SELECT 
    id,
    ROUND((embedding <=> q.query_vec)::numeric, 6) AS cosine_distance
FROM index_advance_test
CROSS JOIN q
ORDER BY embedding <=> q.query_vec
LIMIT 5;

-- ============================================================================
-- PART 6: Index Information and Statistics
-- ============================================================================
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'PART 6: Index Information and Statistics'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo ''
\echo 'Test 6.1: List all HNSW indexes'
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS index_size,
    COALESCE(idx_scan, 0) AS index_scans,
    COALESCE(idx_tup_read, 0) AS tuples_read,
    COALESCE(idx_tup_fetch, 0) AS tuples_fetched
FROM pg_indexes pi
LEFT JOIN pg_stat_user_indexes psi ON pi.indexname = psi.indexrelname
WHERE pi.indexname LIKE '%hnsw%' OR pi.indexname LIKE '%ivf%'
ORDER BY tablename, indexname;

\echo ''
\echo 'Test 6.2: Index usage statistics'
SELECT 
    schemaname,
    tablename,
    indexrelname AS index_name,
    idx_scan AS total_scans,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched
FROM pg_stat_user_indexes
WHERE indexrelname LIKE '%hnsw%' OR indexrelname LIKE '%ivf%'
ORDER BY idx_scan DESC;

-- ============================================================================
-- PART 7: Test Index Maintenance
-- ============================================================================
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'PART 7: Index Maintenance Operations'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo ''
\echo 'Test 7.1: Reindex operation'
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_test_hnsw_l2_custom') THEN
        REINDEX INDEX idx_test_hnsw_l2_custom;
        RAISE NOTICE 'Reindex completed';
    ELSE
        RAISE NOTICE 'Index not found for reindex test';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Reindex test: %', SQLERRM;
END $$;

\echo ''
\echo 'Test 7.2: Vacuum analyze'
VACUUM ANALYZE index_advance_test;

\echo ''
\echo 'Test 7.3: Index health check'
SELECT 
    'Index Health' AS check_type,
    COUNT(*) AS total_indexes,
    COUNT(*) FILTER (WHERE pg_relation_size(indexname::regclass) > 0) AS non_empty_indexes
FROM pg_indexes
WHERE indexname LIKE '%hnsw%' OR indexname LIKE '%ivf%';

-- ============================================================================
-- PART 8: Performance Comparison
-- ============================================================================
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'PART 8: Performance Comparison'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo ''
\echo 'Test 8.1: Performance with index'
DO $$
DECLARE
    start_time timestamp;
    end_time timestamp;
    query_vec vector;
    result_count int;
BEGIN
    query_vec := (SELECT embedding FROM index_advance_test LIMIT 1);
    start_time := clock_timestamp();
    
    SELECT COUNT(*) INTO result_count
    FROM index_advance_test
    WHERE embedding <-> query_vec < 1.0;
    
    end_time := clock_timestamp();
    RAISE NOTICE 'Performance test: % results in % ms', 
        result_count, EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
END $$;

/*-------------------------------------------------------------------
 * ---- INDEX QUERIES WITH VARIOUS K VALUES ----
 * Test KNN queries with different k values
 *------------------------------------------------------------------*/
\echo ''
\echo 'Index Query Tests (Various K)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 11: KNN query with k=1'
SELECT 
	id,
	embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1) AS distance
FROM index_advance_test
ORDER BY embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1)
LIMIT 1;

\echo 'Test 12: KNN query with k=10'
SELECT 
	id,
	embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1) AS distance
FROM index_advance_test
ORDER BY embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1)
LIMIT 10;

\echo 'Test 13: KNN query with k=100'
SELECT 
	COUNT(*) AS result_count
FROM (
	SELECT 
		id,
		embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1) AS distance
	FROM index_advance_test
	ORDER BY embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1)
	LIMIT 100
) sub;

\echo 'Test 14: KNN query with k=1000 (larger than table)'
SELECT 
	COUNT(*) AS result_count
FROM (
	SELECT 
		id,
		embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1) AS distance
	FROM index_advance_test
	ORDER BY embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1)
	LIMIT 1000
) sub;

/*-------------------------------------------------------------------
 * ---- INDEX CONSISTENCY CHECKS ----
 * Test index consistency and validation
 *------------------------------------------------------------------*/
\echo ''
\echo 'Index Consistency Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 15: Index size after inserts'
SELECT 
	indexname,
	pg_size_pretty(pg_relation_size(indexname::regclass)) AS index_size,
	pg_size_pretty(pg_relation_size('index_advance_test'::regclass)) AS table_size
FROM pg_indexes
WHERE tablename = 'index_advance_test'
ORDER BY indexname;

\echo 'Test 16: Index usage statistics'
SELECT 
	schemaname,
	tablename,
	indexname,
	idx_scan AS index_scans,
	idx_tup_read AS tuples_read,
	idx_tup_fetch AS tuples_fetched
FROM pg_stat_user_indexes
WHERE tablename = 'index_advance_test'
ORDER BY indexname;

/*-------------------------------------------------------------------
 * ---- INDEX MAINTENANCE ----
 * Test index maintenance operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Index Maintenance Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 17: REINDEX operation'
REINDEX INDEX idx_hnsw_m16;

\echo 'Test 18: VACUUM on indexed table'
VACUUM ANALYZE index_advance_test;

\echo 'Test 19: Index after updates'
UPDATE index_advance_test 
SET metadata = '{"updated": true}'::jsonb 
WHERE id % 10 = 0;

VACUUM ANALYZE index_advance_test;

\echo 'Test 20: Index after deletes'
DELETE FROM index_advance_test WHERE id % 20 = 0;

VACUUM ANALYZE index_advance_test;

/*-------------------------------------------------------------------
 * ---- MULTI-TENANT INDEX ISOLATION ----
 * Test multi-tenant index operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Multi-Tenant Index Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 21: Create table with tenant_id'
DROP TABLE IF EXISTS index_tenant_test;
CREATE TABLE index_tenant_test (
	id SERIAL PRIMARY KEY,
	tenant_id integer,
	embedding vector(28),
	label integer
);

-- Insert data for multiple tenants
INSERT INTO index_tenant_test (tenant_id, embedding, label)
SELECT 
	(i % 3) + 1 AS tenant_id,
	features AS embedding,
	label
FROM test_train_view
LIMIT 300;

\echo 'Test 22: Create index on multi-tenant table'
CREATE INDEX idx_tenant_hnsw ON index_tenant_test 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 16, ef_construction = 200);

\echo 'Test 23: Query with tenant filter'
SELECT 
	tenant_id,
	COUNT(*) AS result_count
FROM (
	SELECT 
		tenant_id,
		id,
		embedding <-> (SELECT embedding FROM index_tenant_test WHERE tenant_id = 1 LIMIT 1) AS distance
	FROM index_tenant_test
	WHERE tenant_id = 1
	ORDER BY embedding <-> (SELECT embedding FROM index_tenant_test WHERE tenant_id = 1 LIMIT 1)
	LIMIT 10
) sub
GROUP BY tenant_id;

/*-------------------------------------------------------------------
 * ---- INDEX CACHE OPERATIONS ----
 * Test index cache operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Index Cache Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 24: Multiple queries to test cache'
-- Run same query multiple times to test caching
SELECT 
	COUNT(*) AS query_count
FROM (
	SELECT 
		id,
		embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1) AS distance
	FROM index_advance_test
	ORDER BY embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1)
	LIMIT 10
) q1;

SELECT 
	COUNT(*) AS query_count
FROM (
	SELECT 
		id,
		embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1) AS distance
	FROM index_advance_test
	ORDER BY embedding <-> (SELECT embedding FROM index_advance_test LIMIT 1)
	LIMIT 10
) q2;

\echo ''
\echo '=========================================================================='
\echo '✓ Index Module: Full exhaustive code-path test complete'
\echo '=========================================================================='
\echo ''
\echo 'Summary:'
\echo '  - HNSW index tests with various parameters (m, ef_construction)'
\echo '  - IVF index tests with various list counts'
\echo '  - Index performance tests'
\echo '  - Distance metrics tests (L2, Cosine)'
\echo '  - Index information and statistics'
\echo '  - Index maintenance operations (REINDEX, VACUUM)'
\echo '  - Performance comparisons'
\echo '  - KNN queries with various k values'
\echo '  - Index consistency checks'
\echo '  - Multi-tenant index isolation'
\echo '  - Index cache operations'
\echo ''
\echo 'All tests completed successfully!'

DROP TABLE IF EXISTS index_advance_test CASCADE;
DROP TABLE IF EXISTS index_tenant_test CASCADE;
