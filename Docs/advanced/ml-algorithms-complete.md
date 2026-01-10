# Machine Learning Algorithms Complete Reference

**Complete reference for all 19 ML algorithms in NeuronDB with implementation details and usage examples.**

> **Version:** 1.0  
> **Last Updated:** 2025-01-01

## Table of Contents

- [Clustering Algorithms](#clustering-algorithms)
- [Classification Algorithms](#classification-algorithms)
- [Regression Algorithms](#regression-algorithms)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Quantization](#quantization)
- [Outlier Detection](#outlier-detection)
- [Time Series](#time-series)
- [Recommendation Systems](#recommendation-systems)
- [Model Management](#model-management)

---

## Clustering Algorithms

### K-Means

**File:** `ml/ml_kmeans.c`

**Function:** `cluster_kmeans(text, text, integer, integer) → integer[]`

**Parameters:**
- `table`: Table name
- `vector_col`: Vector column name
- `num_clusters`: Number of clusters
- `max_iters`: Maximum iterations (default: 100)

**Returns:** Array of cluster assignments (1-based)

**Algorithm:**
1. Initialize centroids randomly
2. Assign points to nearest centroid
3. Update centroids
4. Repeat until convergence or max iterations

**Example:**
```sql
SELECT cluster_kmeans('points', 'embedding', 5, 100);
```

**GPU Acceleration:**
- GPU-accelerated distance computation
- GPU-accelerated centroid update
- Available via CUDA/ROCm kernels

---

### Mini-Batch K-Means

**File:** `ml/ml_minibatch_kmeans.c`

**Function:** `cluster_minibatch_kmeans(text, text, integer, integer, integer) → integer[]`

**Parameters:**
- `table`: Table name
- `vector_col`: Vector column name
- `num_clusters`: Number of clusters
- `batch_size`: Mini-batch size (default: 100)
- `max_iters`: Maximum iterations (default: 100)

**Benefits:**
- Faster than standard K-Means
- Lower memory usage
- Good for large datasets

**Example:**
```sql
SELECT cluster_minibatch_kmeans('points', 'embedding', 5, 100, 100);
```

---

### DBSCAN

**File:** `ml/ml_dbscan.c`

**Function:** `cluster_dbscan(text, text, double precision, integer) → integer[]`

**Parameters:**
- `table`: Table name
- `vector_col`: Vector column name
- `eps`: Maximum distance for neighborhood (default: 0.5)
- `min_samples`: Minimum samples per cluster (default: 5)

**Returns:** Array of cluster assignments (-1 for noise)

**Algorithm:**
1. Find core points (points with min_samples neighbors within eps)
2. Expand clusters from core points
3. Mark remaining points as noise

**Example:**
```sql
SELECT cluster_dbscan('points', 'embedding', 0.5, 5);
```

---

### Gaussian Mixture Model (GMM)

**File:** `ml/ml_gmm.c`

**Function:** `cluster_gmm(text, text, integer, integer) → integer[]`

**Parameters:**
- `table`: Table name
- `vector_col`: Vector column name
- `num_components`: Number of components
- `max_iters`: Maximum EM iterations (default: 100)

**Algorithm:**
- Expectation-Maximization (EM) algorithm
- Models clusters as Gaussian distributions

**Example:**
```sql
SELECT cluster_gmm('points', 'embedding', 5, 100);
```

**GPU Acceleration:**
- GPU-accelerated EM algorithm
- Available via CUDA/ROCm

---

### Hierarchical Clustering

**File:** `ml/ml_hierarchical.c`

**Function:** `cluster_hierarchical(text, text, integer, text) → integer[]`

**Parameters:**
- `table`: Table name
- `vector_col`: Vector column name
- `num_clusters`: Number of clusters
- `linkage`: Linkage method (default: 'ward')
  - Options: `'ward'`, `'complete'`, `'average'`, `'single'`

**Algorithm:**
- Agglomerative hierarchical clustering
- Builds dendrogram bottom-up

**Example:**
```sql
SELECT cluster_hierarchical('points', 'embedding', 5, 'ward');
```

---

## Classification Algorithms

### Random Forest

**File:** `ml/ml_random_forest.c`

**Training Function:** `train_random_forest_classifier(text, text, text, integer, integer, integer, integer) → integer`

**Parameters:**
- `table_name`: Training table
- `feature_col`: Feature column name
- `label_col`: Label column name
- `n_trees`: Number of trees (default: 100)
- `max_depth`: Maximum tree depth (default: 10)
- `min_samples_split`: Minimum samples to split (default: 2)
- `max_features`: Maximum features per split (default: 0 = sqrt)

**Returns:** Model ID

**Example:**
```sql
SELECT train_random_forest_classifier(
    'iris',
    'features',
    'species',
    100,  -- n_trees
    10,   -- max_depth
    2,    -- min_samples_split
    0     -- max_features (sqrt)
);
```

**Prediction Function:** `predict_random_forest(integer, vector) → double precision`

**Example:**
```sql
SELECT predict_random_forest(1, '[1.0, 2.0, 3.0]'::vector);
```

**Evaluation Function:** `evaluate_random_forest_by_model_id(integer, text, text, text) → jsonb`

**Returns:**
```json
{
  "accuracy": 0.95,
  "precision": 0.94,
  "recall": 0.96,
  "f1": 0.95,
  "confusion_matrix": [[90, 5], [3, 92]]
}
```

**GPU Acceleration:**
- GPU-accelerated tree construction
- GPU-accelerated split evaluation
- GPU-accelerated prediction

---

### Logistic Regression

**File:** `ml/ml_logistic_regression.c`

**Training Function:** `train_logistic_regression(text, text, text, integer, float8, float8) → integer`

**Parameters:**
- `table_name`: Training table
- `feature_col`: Feature column name
- `label_col`: Label column name
- `max_iterations`: Maximum iterations (default: 1000)
- `learning_rate`: Learning rate (default: 0.01)
- `regularization`: Regularization parameter (default: 0.001)

**Returns:** Model ID

**Algorithm:**
- Gradient descent with L2 regularization
- Binary classification

**Example:**
```sql
SELECT train_logistic_regression(
    'iris',
    'features',
    'species',
    1000,   -- max_iterations
    0.01,   -- learning_rate
    0.001   -- regularization
);
```

**Prediction Function:** `predict_logistic_regression(integer, vector) → float8`

**Returns:** Probability (0.0-1.0)

**GPU Acceleration:**
- GPU-accelerated gradient descent
- GPU-accelerated batch processing

---

### Support Vector Machine (SVM)

**File:** `ml/ml_svm.c`

**Training Function:** `train_svm_classifier(text, text, text, float8, integer) → integer`

**Parameters:**
- `table_name`: Training table
- `feature_col`: Feature column name
- `label_col`: Label column name
- `C`: Regularization parameter (default: 1.0)
- `max_iters`: Maximum iterations (default: 1000)

**Returns:** Model ID

**Algorithm:**
- Support Vector Machine with RBF kernel
- Binary classification

**Example:**
```sql
SELECT train_svm_classifier(
    'iris',
    'features',
    'species',
    1.0,    -- C
    1000    -- max_iters
);
```

**Prediction Function:** `predict_svm_model_id(integer, vector) → float8`

**GPU Acceleration:**
- GPU-accelerated kernel computation
- GPU-accelerated optimization

---

### Naive Bayes

**File:** `ml/ml_naive_bayes.c`

**Training Function:** `train_naive_bayes_classifier_model_id(text, text, text) → integer`

**Parameters:**
- `table_name`: Training table
- `feature_col`: Feature column name
- `label_col`: Label column name

**Returns:** Model ID

**Algorithm:**
- Gaussian Naive Bayes
- Assumes feature independence

**Example:**
```sql
SELECT train_naive_bayes_classifier_model_id(
    'iris',
    'features',
    'species'
);
```

**Prediction Function:** `predict_naive_bayes_model_id(integer, vector) → integer`

**Returns:** Class prediction

---

### Decision Tree

**File:** `ml/ml_decision_tree.c`

**Training Function:** `train_decision_tree_classifier(text, text, text, integer, integer) → integer`

**Parameters:**
- `table_name`: Training table
- `feature_col`: Feature column name
- `label_col`: Label column name
- `max_depth`: Maximum tree depth (default: 10)
- `min_samples_split`: Minimum samples to split (default: 2)

**Returns:** Model ID

**Algorithm:**
- CART (Classification and Regression Trees)
- Gini impurity for splitting

**Example:**
```sql
SELECT train_decision_tree_classifier(
    'iris',
    'features',
    'species',
    10,  -- max_depth
    2    -- min_samples_split
);
```

**Prediction Function:** `predict_decision_tree_model_id(integer, vector) → float8`

**GPU Acceleration:**
- GPU-accelerated split evaluation
- GPU-accelerated tree traversal

---

### Gradient Boosting

#### XGBoost

**File:** `ml/ml_xgboost.c`

**Training Function:** `train_xgboost_classifier(text, text, text, integer, integer, float8) → integer`

**Parameters:**
- `table_name`: Training table
- `feature_col`: Feature column name
- `label_col`: Label column name
- `n_estimators`: Number of trees (default: 100)
- `max_depth`: Maximum tree depth (default: 6)
- `learning_rate`: Learning rate (default: 0.3)

**Example:**
```sql
SELECT train_xgboost_classifier(
    'iris',
    'features',
    'species',
    100,  -- n_estimators
    6,    -- max_depth
    0.3   -- learning_rate
);
```

**GPU Acceleration:**
- GPU-accelerated tree construction
- GPU-accelerated prediction

#### CatBoost

**File:** `ml/ml_catboost.c`

**Training Function:** `train_catboost_classifier(text, text, text, integer, float8, integer) → integer`

**Parameters:**
- `table_name`: Training table
- `feature_col`: Feature column name
- `label_col`: Label column name
- `iterations`: Number of iterations (default: 1000)
- `learning_rate`: Learning rate (default: 0.03)
- `depth`: Tree depth (default: 6)

**Example:**
```sql
SELECT train_catboost_classifier(
    'iris',
    'features',
    'species',
    1000,  -- iterations
    0.03,   -- learning_rate
    6       -- depth
);
```

#### LightGBM

**File:** `ml/ml_lightgbm.c`

**Training Function:** `train_lightgbm_classifier(text, text, text, integer, integer, float8) → integer`

**Parameters:**
- `table_name`: Training table
- `feature_col`: Feature column name
- `label_col`: Label column name
- `n_estimators`: Number of trees (default: 100)
- `num_leaves`: Number of leaves (default: 31)
- `learning_rate`: Learning rate (default: 0.1)

**Example:**
```sql
SELECT train_lightgbm_classifier(
    'iris',
    'features',
    'species',
    100,  -- n_estimators
    31,   -- num_leaves
    0.1   -- learning_rate
);
```

---

## Regression Algorithms

### Linear Regression

**File:** `ml/ml_linear_regression.c`

**Training Function:** `train_linear_regression(text, text, text) → integer`

**Parameters:**
- `table_name`: Training table
- `feature_col`: Feature column name
- `label_col`: Label column name

**Returns:** Model ID

**Algorithm:**
- Ordinary Least Squares (OLS)
- Closed-form solution

**Example:**
```sql
SELECT train_linear_regression(
    'housing',
    'features',
    'price'
);
```

**Prediction Function:** `predict_linear_regression_model_id(integer, vector) → float8`

**GPU Acceleration:**
- GPU-accelerated matrix operations
- GPU-accelerated batch prediction

---

### Ridge Regression

**File:** `ml/ml_ridge_lasso.c`

**Training Function:** `train_ridge_regression(text, text, text, float8) → integer`

**Parameters:**
- `table_name`: Training table
- `feature_col`: Feature column name
- `label_col`: Label column name
- `alpha`: Regularization parameter

**Returns:** Model ID

**Algorithm:**
- L2 regularization
- Prevents overfitting

**Example:**
```sql
SELECT train_ridge_regression(
    'housing',
    'features',
    'price',
    0.1  -- alpha
);
```

**Prediction Function:** `predict_ridge_regression_model_id(integer, vector) → float8`

---

### Lasso Regression

**File:** `ml/ml_ridge_lasso.c`

**Training Function:** `train_lasso_regression(text, text, text, float8, integer) → integer`

**Parameters:**
- `table_name`: Training table
- `feature_col`: Feature column name
- `label_col`: Label column name
- `alpha`: Regularization parameter
- `max_iterations`: Maximum iterations (default: 1000)

**Returns:** Model ID

**Algorithm:**
- L1 regularization
- Feature selection

**Example:**
```sql
SELECT train_lasso_regression(
    'housing',
    'features',
    'price',
    0.1,    -- alpha
    1000    -- max_iterations
);
```

**Prediction Function:** `predict_lasso_regression_model_id(integer, vector) → float8`

**GPU Acceleration:**
- GPU-accelerated coordinate descent
- GPU-accelerated batch processing

---

## Dimensionality Reduction

### Principal Component Analysis (PCA)

**File:** `ml/ml_dimensionality_reduction.c`

**Function:** `reduce_pca(text, text, integer) → real[][]`

**Parameters:**
- `table`: Table name
- `column`: Vector column name
- `n_components`: Number of components

**Returns:** Array of reduced vectors

**Algorithm:**
- Singular Value Decomposition (SVD)
- Preserves maximum variance

**Example:**
```sql
SELECT reduce_pca('points', 'embedding', 2);
```

---

### PCA Whitening

**File:** `ml/ml_pca_whitening.c`

**Function:** `reduce_pca_whitening(text, text, integer) → real[][]`

**Parameters:**
- `table`: Table name
- `column`: Vector column name
- `n_components`: Number of components

**Returns:** Array of whitened vectors

**Algorithm:**
- PCA with unit variance
- Decorrelates features

**Example:**
```sql
SELECT reduce_pca_whitening('points', 'embedding', 2);
```

---

## Quantization

### Product Quantization (PQ)

**File:** `ml/ml_product_quantization.c`

**Function:** `train_pq_codebook(text, text, integer, integer) → bytea`

**Parameters:**
- `table`: Table name
- `vector_col`: Vector column name
- `num_subspaces`: Number of subspaces
- `num_centroids`: Number of centroids per subspace

**Returns:** Codebook data

**Algorithm:**
- Vector quantization
- Compresses vectors for storage

**Example:**
```sql
SELECT train_pq_codebook('vectors', 'embedding', 8, 256);
```

**GPU Acceleration:**
- GPU-accelerated codebook training
- GPU-accelerated encoding

---

### Optimized Product Quantization (OPQ)

**File:** `ml/ml_opq.c`

**Function:** `train_opq_rotation(text, text, integer, integer) → real[][]`

**Parameters:**
- `table`: Table name
- `vector_col`: Vector column name
- `num_subspaces`: Number of subspaces
- `num_centroids`: Number of centroids

**Returns:** Rotation matrix

**Algorithm:**
- OPQ with rotation optimization
- Better quantization quality

**Example:**
```sql
SELECT train_opq_rotation('vectors', 'embedding', 8, 256);
```

---

## Outlier Detection

### Z-Score

**File:** `ml/ml_outlier_detection.c`

**Function:** `detect_outliers_zscore(text, text, double precision) → integer[]`

**Parameters:**
- `table`: Table name
- `vector_col`: Vector column name
- `threshold`: Z-score threshold (default: 3.0)

**Returns:** Array of outlier indices

**Algorithm:**
- Statistical outlier detection
- Based on standard deviations

**Example:**
```sql
SELECT detect_outliers_zscore('points', 'embedding', 3.0);
```

---

### Modified Z-Score

**File:** `ml/ml_outlier_detection.c`

**Function:** `detect_outliers_modified_zscore(text, text, double precision) → integer[]`

**Parameters:**
- `table`: Table name
- `vector_col`: Vector column name
- `threshold`: Modified Z-score threshold (default: 3.5)

**Returns:** Array of outlier indices

**Algorithm:**
- Robust to outliers
- Uses median instead of mean

**Example:**
```sql
SELECT detect_outliers_modified_zscore('points', 'embedding', 3.5);
```

---

## Time Series

### ARIMA

**File:** `ml/ml_timeseries.c`

**Training Function:** `train_timeseries_cpu(text, text, text, integer, integer, integer) → integer`

**Parameters:**
- `table`: Table name
- `feature_col`: Feature column name
- `label_col`: Label column name
- `p`: AR order
- `d`: Differencing order
- `q`: MA order

**Returns:** Model ID

**Algorithm:**
- ARIMA (AutoRegressive Integrated Moving Average)
- Time series forecasting

**Example:**
```sql
SELECT train_timeseries_cpu(
    'sales',
    'features',
    'value',
    1,  -- p (AR order)
    1,  -- d (differencing)
    1   -- q (MA order)
);
```

---

## Recommendation Systems

### Collaborative Filtering

**File:** `ml/ml_recommender.c`

**Training Function:** `train_collaborative_filter(text, text, text, text) → integer`

**Parameters:**
- `table`: Table name
- `user_col`: User column name
- `item_col`: Item column name
- `rating_col`: Rating column name

**Returns:** Model ID

**Algorithm:**
- Matrix factorization
- User-item recommendations

**Example:**
```sql
SELECT train_collaborative_filter(
    'ratings',
    'user_id',
    'item_id',
    'rating'
);
```

**Prediction Function:** `predict_collaborative_filter(integer, integer, integer) → float8`

**Example:**
```sql
SELECT predict_collaborative_filter(1, 123, 456);  -- model_id, user_id, item_id
```

---

## Model Management

### Model Catalog

**File:** `ml/ml_catalog.c`

All trained models are stored in the `neurondb.ml_models` catalog table.

**Schema:**
```sql
CREATE TABLE neurondb.ml_models (
    model_id SERIAL PRIMARY KEY,
    algorithm text NOT NULL,
    parameters jsonb,
    model_data bytea,
    metrics jsonb,
    created_at timestamptz DEFAULT now()
);
```

### List Models

**Function:** `list_models(text, text) → TABLE(...)`

**Parameters:**
- `algorithm` (optional): Filter by algorithm
- `project` (optional): Filter by project

**Example:**
```sql
SELECT * FROM list_models('random_forest', NULL);
```

### Get Model Info

**Function:** `get_model_info(integer) → jsonb`

**Parameters:**
- `model_id`: Model ID

**Returns:**
```json
{
  "model_id": 1,
  "algorithm": "random_forest",
  "parameters": {
    "n_trees": 100,
    "max_depth": 10
  },
  "metrics": {
    "accuracy": 0.95
  },
  "created_at": "2025-01-01T00:00:00Z"
}
```

### Delete Model

**Function:** `delete_model(integer) → boolean`

**Parameters:**
- `model_id`: Model ID

**Example:**
```sql
SELECT delete_model(1);
```

---

## Unified API

### Train Model

**File:** `ml/ml_unified_api.c`

**Function:** `neurondb_train(text, text, text, text, jsonb) → integer`

**Parameters:**
- `algorithm`: Algorithm name
- `table`: Training table
- `features`: Feature column
- `labels`: Label column
- `params`: Hyperparameters (JSONB)

**Example:**
```sql
SELECT neurondb_train(
    'random_forest',
    'iris',
    'features',
    'species',
    '{"n_trees": 100, "max_depth": 10}'::jsonb
);
```

### Predict

**Function:** `neurondb_predict(integer, real[]) → float8`

**Parameters:**
- `model_id`: Model ID
- `features`: Feature vector

**Example:**
```sql
SELECT neurondb_predict(1, ARRAY[1.0, 2.0, 3.0]::real[]);
```

---

## Performance Considerations

### GPU Acceleration

Most algorithms support GPU acceleration:

- **Random Forest:** GPU-accelerated tree construction
- **Logistic Regression:** GPU-accelerated gradient descent
- **Linear Regression:** GPU-accelerated matrix operations
- **K-Means:** GPU-accelerated distance computation
- **SVM:** GPU-accelerated kernel computation

**Enable GPU:**
```sql
SET neurondb.compute_mode = 2;  -- Auto (try GPU first)
```

### Batch Processing

Use batch operations for better performance:

```sql
-- Batch prediction
SELECT neurondb_predict_batch(1, ARRAY[
    ARRAY[1.0, 2.0, 3.0],
    ARRAY[4.0, 5.0, 6.0]
]::real[][]);
```

---

## Related Documentation

- [SQL API Reference](../reference/sql-api-complete.md)
- [GPU Acceleration](gpu-acceleration-complete.md)
- [Configuration Reference](../reference/configuration-complete.md)

---

**Last Updated:** 2025-01-01  
**Documentation Version:** 1.0.0




