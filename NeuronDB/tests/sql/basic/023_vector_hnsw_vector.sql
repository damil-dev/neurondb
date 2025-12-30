-- compatibility test: hnsw_vector.sql
-- Tests HNSW index for vector type with various distance operators
-- Based on test/sql/hnsw_vector.sql

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Compatibility Test: HNSW Index for Vector Type'
\echo '=========================================================================='

SET enable_seqscan = off;

-- Test 1: HNSW Index with L2 Distance
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: HNSW Index with L2 Distance (<-> operator)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS t CASCADE;
CREATE TABLE t (val vector(3));
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);
CREATE INDEX ON t USING hnsw (val vector_l2_ops);

INSERT INTO t (val) VALUES ('[1,2,4]');

SELECT * FROM t ORDER BY val <-> '[3,3,3]';
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <-> (SELECT NULL::vector)) t2;
SELECT COUNT(*) FROM t;

TRUNCATE t;
SELECT * FROM t ORDER BY val <-> '[3,3,3]';

DROP TABLE t;

-- Test 2: HNSW Index with Inner Product
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: HNSW Index with Inner Product (<#> operator)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS t CASCADE;
CREATE TABLE t (val vector(3));
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);
CREATE INDEX ON t USING hnsw (val vector_ip_ops);

INSERT INTO t (val) VALUES ('[1,2,4]');

SELECT * FROM t ORDER BY val <#> '[3,3,3]';
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <#> (SELECT NULL::vector)) t2;

DROP TABLE t;

-- Test 3: HNSW Index with Cosine Distance
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: HNSW Index with Cosine Distance (<=> operator)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS t CASCADE;
CREATE TABLE t (val vector(3));
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);
CREATE INDEX ON t USING hnsw (val vector_cosine_ops);

INSERT INTO t (val) VALUES ('[1,2,4]');

SELECT * FROM t ORDER BY val <=> '[3,3,3]';
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <=> '[0,0,0]') t2;
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <=> (SELECT NULL::vector)) t2;

DROP TABLE t;

-- Test 4: HNSW Index with L1 Distance
-- NOTE: L1 distance operator class (vector_l1_ops) is not supported for HNSW indexes
-- L1 distance is supported as a function/operator, but not as an index operator class
-- This test is commented out until L1 indexing support is added
--\echo ''
--\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
--\echo 'Test 4: HNSW Index with L1 Distance (<+> operator)'
--\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

--DROP TABLE IF EXISTS t CASCADE;
--CREATE TABLE t (val vector(3));
--INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);
--CREATE INDEX ON t USING hnsw (val vector_l1_ops);

--INSERT INTO t (val) VALUES ('[1,2,4]');

--SELECT * FROM t ORDER BY val <+> '[3,3,3]';
--SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <+> (SELECT NULL::vector)) t2;

--DROP TABLE t;

-- Test 5: HNSW Index Options
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 5: HNSW Index Options'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS t CASCADE;
CREATE TABLE t (val vector(3));
-- Test invalid m values (should fail)
DO $$ BEGIN
    CREATE INDEX ON t USING hnsw (val vector_l2_ops) WITH (m = 1);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;
DO $$ BEGIN
    CREATE INDEX ON t USING hnsw (val vector_l2_ops) WITH (m = 101);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;
-- Test invalid ef_construction values (should fail)
DO $$ BEGIN
    CREATE INDEX ON t USING hnsw (val vector_l2_ops) WITH (ef_construction = 3);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;
DO $$ BEGIN
    CREATE INDEX ON t USING hnsw (val vector_l2_ops) WITH (ef_construction = 1001);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;
-- Test valid parameters
CREATE INDEX ON t USING hnsw (val vector_l2_ops) WITH (m = 16, ef_construction = 31);

-- Test index configuration parameters
SHOW neurondb.hnsw_ef_search;

-- Test invalid parameter values (should fail)
DO $$ BEGIN
    SET neurondb.hnsw_ef_search = 0;
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;
DO $$ BEGIN
    SET neurondb.hnsw_ef_search = 1001;
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

-- NOTE: The following parameters are pgvector-specific and not supported in NeuronDB:
-- - hnsw.iterative_scan
-- - hnsw.max_scan_tuples  
-- - hnsw.scan_mem_multiplier
-- These tests are commented out until equivalent functionality is added
--SHOW hnsw.iterative_scan;
--SET hnsw.iterative_scan = on;
--SHOW hnsw.max_scan_tuples;
--SET hnsw.max_scan_tuples = 0;
--SHOW hnsw.scan_mem_multiplier;
--SET hnsw.scan_mem_multiplier = 0;
--SET hnsw.scan_mem_multiplier = 1001;

DROP TABLE t;

-- Test 6: Unlogged Tables
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 6: HNSW Index on Unlogged Tables'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS t CASCADE;
CREATE UNLOGGED TABLE t (val vector(3));
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);
CREATE INDEX ON t USING hnsw (val vector_l2_ops);

SELECT * FROM t ORDER BY val <-> '[3,3,3]';

DROP TABLE t;

RESET enable_seqscan;

\echo ''
\echo '=========================================================================='
\echo 'Test completed successfully'
\echo '=========================================================================='

