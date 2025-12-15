-- compatibility test: btree.sql
-- Tests B-tree indexes on vector, halfvec, sparsevec types
-- Based on test/sql/btree.sql

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Compatibility Test: B-tree Indexes'
\echo '=========================================================================='

SET enable_seqscan = off;

-- Test 1: B-tree Index on Vector
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: B-tree Index on Vector Type'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TABLE t (val vector(3));
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);
CREATE INDEX ON t (val);

SELECT * FROM t WHERE val = '[1,2,3]';
SELECT * FROM t ORDER BY val;

DROP TABLE t;

-- Test 2: B-tree Index on Halfvec
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: B-tree Index on Halfvec Type'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TABLE t (val halfvec(3));
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);
CREATE INDEX ON t (val);

SELECT * FROM t WHERE val = '[1,2,3]';
SELECT * FROM t ORDER BY val;

DROP TABLE t;

-- Test 3: B-tree Index on Sparsevec
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: B-tree Index on Sparsevec Type'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TABLE t (val sparsevec(3));
INSERT INTO t (val) VALUES ('{}/3'), ('{1:1,2:2,3:3}/3'), ('{1:1,2:1,3:1}/3'), (NULL);
CREATE INDEX ON t (val);

SELECT * FROM t WHERE val = '{1:1,2:2,3:3}/3';
SELECT * FROM t ORDER BY val;

DROP TABLE t;

RESET enable_seqscan;

\echo ''
\echo '=========================================================================='
\echo 'Test completed successfully'
\echo '=========================================================================='

