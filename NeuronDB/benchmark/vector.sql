-- ============================================================================
-- NeuronDB pgbench Benchmark: Vector Operations
-- ============================================================================
-- This file benchmarks vector operations including:
-- - Distance calculations (L2, cosine, inner product)
-- - Vector arithmetic (add, subtract, multiply, divide)
-- - Vector normalization and norms
-- - Similarity searches
-- ============================================================================

-- Setup: Create test vectors with random values
\set v1 random(1, 1000)
\set v2 random(1, 1000)
\set dim random(64, 512)

-- Benchmark 1: L2 (Euclidean) Distance
SELECT vector_l2_distance(
    ('[' || :v1 || ',' || :v2 || ',' || (:v1 + :v2) || ']')::vector,
    ('[' || :v2 || ',' || :v1 || ',' || (:v1 * 2) || ']')::vector
) AS l2_distance;

-- Benchmark 2: Cosine Distance
SELECT vector_cosine_distance(
    ('[' || :v1 || ',' || :v2 || ',' || (:v1 + :v2) || ']')::vector,
    ('[' || :v2 || ',' || :v1 || ',' || (:v1 * 2) || ']')::vector
) AS cosine_distance;

-- Benchmark 3: Inner Product
SELECT vector_inner_product(
    ('[' || :v1 || ',' || :v2 || ',' || (:v1 + :v2) || ']')::vector,
    ('[' || :v2 || ',' || :v1 || ',' || (:v1 * 2) || ']')::vector
) AS inner_product;

-- Benchmark 4: Vector Addition
SELECT ('[' || :v1 || ',' || :v2 || ']')::vector + ('[' || :v2 || ',' || :v1 || ']')::vector AS vector_add;

-- Benchmark 5: Vector Subtraction
SELECT ('[' || :v1 || ',' || :v2 || ']')::vector - ('[' || :v2 || ',' || :v1 || ']')::vector AS vector_sub;

-- Benchmark 6: Scalar Multiplication
SELECT ('[' || :v1 || ',' || :v2 || ']')::vector * 2.5 AS vector_scale;

-- Benchmark 7: Vector Norm
SELECT vector_norm(('[' || :v1 || ',' || :v2 || ',' || (:v1 + :v2) || ']')::vector) AS vector_norm;

-- Benchmark 8: Vector Normalization
SELECT vector_normalize(('[' || :v1 || ',' || :v2 || ',' || (:v1 + :v2) || ']')::vector) AS vector_normalized;

-- Benchmark 9: Cosine Similarity
SELECT vector_cosine_sim(
    ('[' || :v1 || ',' || :v2 || ']')::vector,
    ('[' || :v2 || ',' || :v1 || ']')::vector
) AS cosine_similarity;

-- Benchmark 10: L1 (Manhattan) Distance
SELECT vector_l1_distance(
    ('[' || :v1 || ',' || :v2 || ',' || (:v1 + :v2) || ']')::vector,
    ('[' || :v2 || ',' || :v1 || ',' || (:v1 * 2) || ']')::vector
) AS l1_distance;

