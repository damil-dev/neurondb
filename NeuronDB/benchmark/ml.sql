-- ============================================================================
-- NeuronDB pgbench Benchmark: Machine Learning Operations
-- ============================================================================
-- This file benchmarks ML operations including:
-- - Model training (lightweight operations)
-- - Model prediction
-- - Model evaluation
-- - Feature transformations
-- ============================================================================

-- Setup: Random values for ML operations
\set model_id random(1, 100)
\set feature_val1 random(1, 1000)
\set feature_val2 random(1, 1000)
\set feature_val3 random(1, 1000)

-- Note: Full ML training is expensive, so we focus on prediction and evaluation
-- For training benchmarks, use separate long-running tests

-- Benchmark 1: Model Prediction (assumes model exists)
-- This will fail if no model exists, but that's expected for pgbench
SELECT neurondb.predict(
    :model_id,
    ('[' || :feature_val1 || ',' || :feature_val2 || ',' || :feature_val3 || ']')::vector
) AS ml_prediction;

-- Benchmark 2: Model Information Lookup
SELECT 
    model_id,
    algorithm,
    status,
    num_samples,
    num_features
FROM neurondb.ml_models
WHERE model_id = :model_id
LIMIT 1;

-- Benchmark 3: List Available Algorithms
SELECT algorithm, category, supervised
FROM neurondb.list_algorithms()
WHERE category = 'regression'
LIMIT 5;

-- Benchmark 4: Feature Vector Creation
SELECT ('[' || :feature_val1 || ',' || :feature_val2 || ',' || :feature_val3 || ']')::vector AS feature_vector;

-- Benchmark 5: Vector Statistics (useful for ML preprocessing)
SELECT 
    vector_norm(('[' || :feature_val1 || ',' || :feature_val2 || ',' || :feature_val3 || ']')::vector) AS feature_norm,
    vector_dims(('[' || :feature_val1 || ',' || :feature_val2 || ',' || :feature_val3 || ']')::vector) AS feature_dims;

-- Benchmark 6: Model Metrics Lookup (if model exists)
SELECT 
    model_id,
    metrics->>'mse' AS mse,
    metrics->>'mae' AS mae,
    metrics->>'r_squared' AS r_squared
FROM neurondb.ml_models
WHERE model_id = :model_id
  AND metrics IS NOT NULL
LIMIT 1;

-- Benchmark 7: Batch Prediction (if model supports it)
-- Note: This assumes a model exists and accepts vector inputs
SELECT neurondb.predict(
    :model_id,
    ('[' || :feature_val1 || ',' || :feature_val2 || ']')::vector
) AS batch_pred_1,
neurondb.predict(
    :model_id,
    ('[' || :feature_val2 || ',' || :feature_val3 || ']')::vector
) AS batch_pred_2;

