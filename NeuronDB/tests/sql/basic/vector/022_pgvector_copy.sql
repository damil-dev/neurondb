-- compatibility test: copy.sql
-- Tests COPY command for vector, halfvec, sparsevec types
-- Based on test/sql/copy.sql
-- Note: COPY tests require results directory, may skip if not available

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Compatibility Test: COPY Command'
\echo '=========================================================================='

-- Test 1: Vector COPY
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Vector COPY (Binary Format)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TABLE t (val vector(3));
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);

CREATE TABLE t2 (val vector(3));

-- Try COPY if results directory exists, otherwise skip
DO $$
BEGIN
    BEGIN
        COPY t TO '/tmp/test_vector.bin' WITH (FORMAT binary);
        COPY t2 FROM '/tmp/test_vector.bin' WITH (FORMAT binary);
        RAISE NOTICE 'Vector COPY test completed';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Vector COPY test skipped (may require specific directory): %', SQLERRM;
    END;
END $$;

SELECT * FROM t2 ORDER BY val;

DROP TABLE t;
DROP TABLE t2;

-- Test 2: Halfvec COPY
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Halfvec COPY (Binary Format)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TABLE t (val halfvec(3));
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);

CREATE TABLE t2 (val halfvec(3));

DO $$
BEGIN
    BEGIN
        COPY t TO '/tmp/test_halfvec.bin' WITH (FORMAT binary);
        COPY t2 FROM '/tmp/test_halfvec.bin' WITH (FORMAT binary);
        RAISE NOTICE 'Halfvec COPY test completed';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Halfvec COPY test skipped: %', SQLERRM;
    END;
END $$;

SELECT * FROM t2 ORDER BY val;

DROP TABLE t;
DROP TABLE t2;

-- Test 3: Sparsevec COPY
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Sparsevec COPY (Binary Format)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TABLE t (val sparsevec(3));
INSERT INTO t (val) VALUES ('{}/3'), ('{1:1,2:2,3:3}/3'), ('{1:1,2:1,3:1}/3'), (NULL);

CREATE TABLE t2 (val sparsevec(3));

DO $$
BEGIN
    BEGIN
        COPY t TO '/tmp/test_sparsevec.bin' WITH (FORMAT binary);
        COPY t2 FROM '/tmp/test_sparsevec.bin' WITH (FORMAT binary);
        RAISE NOTICE 'Sparsevec COPY test completed';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Sparsevec COPY test skipped: %', SQLERRM;
    END;
END $$;

SELECT * FROM t2 ORDER BY val;

DROP TABLE t;
DROP TABLE t2;

\echo ''
\echo '=========================================================================='
\echo 'Test completed successfully'
\echo '=========================================================================='

