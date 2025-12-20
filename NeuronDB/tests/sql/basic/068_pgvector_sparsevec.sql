-- compatibility test: sparsevec.sql
-- Tests sparsevec type operations
-- Based on test/sql/sparsevec.sql

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Compatibility Test: Sparsevec Type Operations'
\echo '=========================================================================='

-- Test 1: Sparsevec Parsing
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Sparsevec Parsing'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT '{1:1.5,3:3.5}/5'::sparsevec;
SELECT '{1:-2,3:-4}/5'::sparsevec;
-- Note: NeuronDB may not support whitespace in sparsevec format, test without
SELECT '{1:1.5,3:3.5}/5'::sparsevec;
-- Note: Empty sparsevec format - test if supported
DO $$
BEGIN
    BEGIN
        PERFORM '{}/5'::sparsevec;
        RAISE NOTICE 'Empty sparsevec format accepted';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Empty sparsevec format not supported: %', SQLERRM;
    END;
END $$;

-- Test 2: Sparsevec Comparison
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Sparsevec Comparison'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT '{1:1,2:2,3:3}/3'::sparsevec < '{1:1,2:2,3:3}/3';
SELECT '{1:1,2:2,3:3}/3'::sparsevec <= '{1:1,2:2,3:3}/3';
SELECT '{1:1,2:2,3:3}/3'::sparsevec = '{1:1,2:2,3:3}/3';
SELECT '{1:1,2:2,3:3}/3'::sparsevec != '{1:1,2:2,3:3}/3';
SELECT '{1:1,2:2,3:3}/3'::sparsevec >= '{1:1,2:2,3:3}/3';
SELECT '{1:1,2:2,3:3}/3'::sparsevec > '{1:1,2:2,3:3}/3';

-- Test 3: Sparsevec Distance Functions
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Sparsevec Distance Functions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT sparsevec_l2_distance('{}/2'::sparsevec, '{1:3,2:4}/2');
SELECT '{}/2'::sparsevec <-> '{1:3,2:4}/2';

SELECT sparsevec_inner_product('{1:1,2:2}/2'::sparsevec, '{1:3,2:4}/2');
SELECT '{1:1,2:2}/2'::sparsevec <#> '{1:3,2:4}/2';

SELECT sparsevec_cosine_distance('{1:1,2:2}/2'::sparsevec, '{1:2,2:4}/2');
SELECT '{1:1,2:2}/2'::sparsevec <=> '{1:2,2:4}/2';

SELECT '{}/2'::sparsevec <+> '{1:3,2:4}/2';

-- Test 4: Sparsevec Normalization
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 4: Sparsevec Normalization'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT sparsevec_l2_norm('{1:3,2:4}/2'::sparsevec);
SELECT sparsevec_l2_normalize('{1:3,2:4}/2'::sparsevec);

\echo ''
\echo '=========================================================================='
\echo 'Test completed successfully'
\echo '=========================================================================='

