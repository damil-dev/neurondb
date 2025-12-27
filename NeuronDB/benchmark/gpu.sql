-- ============================================================================
-- NeuronDB pgbench Benchmark: GPU-Accelerated Operations
-- ============================================================================
-- This file benchmarks GPU-accelerated operations including:
-- - GPU distance calculations
-- - GPU vector operations
-- - GPU availability checks
-- ============================================================================

-- Setup: Random values for GPU operations
\set v1 random(1, 1000)
\set v2 random(1, 1000)
\set dim random(64, 512)

-- Benchmark 1: GPU L2 Distance
SELECT vector_l2_distance_gpu(
    ('[' || :v1 || ',' || :v2 || ',' || (:v1 + :v2) || ']')::vector,
    ('[' || :v2 || ',' || :v1 || ',' || (:v1 * 2) || ']')::vector
) AS gpu_l2_distance;

-- Benchmark 2: GPU Cosine Distance
SELECT vector_cosine_distance_gpu(
    ('[' || :v1 || ',' || :v2 || ',' || (:v1 + :v2) || ']')::vector,
    ('[' || :v2 || ',' || :v1 || ',' || (:v1 * 2) || ']')::vector
) AS gpu_cosine_distance;

-- Benchmark 3: GPU Inner Product
SELECT vector_inner_product_gpu(
    ('[' || :v1 || ',' || :v2 || ',' || (:v1 + :v2) || ']')::vector,
    ('[' || :v2 || ',' || :v1 || ',' || (:v1 * 2) || ']')::vector
) AS gpu_inner_product;

-- Benchmark 4: GPU Availability Check
SELECT 
    device_id,
    device_name,
    is_available,
    total_memory_mb,
    free_memory_mb
FROM neurondb_gpu_info()
WHERE is_available = true
LIMIT 1;

-- Benchmark 5: GPU Batch Distance (if available)
SELECT vector_l2_distance_batch(
    ARRAY[
        ('[' || :v1 || ',' || :v2 || ']')::vector,
        ('[' || :v2 || ',' || :v1 || ']')::vector,
        ('[' || (:v1 + :v2) || ',' || (:v1 * 2) || ']')::vector
    ]::vector[],
    ('[' || :v1 || ',' || :v2 || ']')::vector
) AS gpu_batch_distance;

-- Benchmark 6: GPU-accelerated Embedding (if GPU mode enabled)
-- Note: This requires GPU compute mode to be enabled
SELECT vector_dims(embed_text(
    'GPU-accelerated embedding test',
    'all-MiniLM-L6-v2'
)) AS gpu_embedding_dims;

-- Benchmark 7: GPU vs CPU Distance Comparison
-- Compare GPU and CPU implementations
SELECT 
    vector_l2_distance(
        ('[' || :v1 || ',' || :v2 || ']')::vector,
        ('[' || :v2 || ',' || :v1 || ']')::vector
    ) AS cpu_distance,
    vector_l2_distance_gpu(
        ('[' || :v1 || ',' || :v2 || ']')::vector,
        ('[' || :v2 || ',' || :v1 || ']')::vector
    ) AS gpu_distance;

