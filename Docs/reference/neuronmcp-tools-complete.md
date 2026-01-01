# NeuronMCP Tools Complete Reference

**Complete reference for all 100+ NeuronMCP tools with detailed parameters, examples, and error handling.**

> **Version:** 1.0  
> **Protocol:** MCP (Model Context Protocol)  
> **Last Updated:** 2025-01-01

## Table of Contents

- [Vector Operations](#vector-operations)
- [Embedding Functions](#embedding-functions)
- [Hybrid Search](#hybrid-search)
- [Reranking](#reranking)
- [Machine Learning](#machine-learning)
- [RAG Operations](#rag-operations)
- [Indexing](#indexing)
- [PostgreSQL Administration](#postgresql-administration)
- [Analytics](#analytics)
- [ONNX Tools](#onnx-tools)
- [GPU Tools](#gpu-tools)
- [Workers](#workers)

---

## Vector Operations

### `vector_search`

Perform vector similarity search using various distance metrics.

**Parameters:**
- `table` (string, required): Table name containing vectors
- `vector_column` (string, required): Name of the vector column
- `query_vector` (array, required): Query vector for similarity search
- `limit` (number, default: 10): Maximum number of results (1-1000)
- `distance_metric` (string, default: "l2"): Distance metric
  - Options: `"l2"`, `"cosine"`, `"inner_product"`, `"l1"`, `"hamming"`, `"chebyshev"`, `"minkowski"`
- `additional_columns` (array, optional): Additional columns to return

**Example:**
```json
{
  "name": "vector_search",
  "arguments": {
    "table": "documents",
    "vector_column": "embedding",
    "query_vector": [0.1, 0.2, 0.3],
    "limit": 10,
    "distance_metric": "cosine",
    "additional_columns": ["id", "content"]
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "id": 1,
      "content": "Document content",
      "distance": 0.123,
      "embedding": [0.1, 0.2, 0.3]
    }
  ],
  "count": 10
}
```

---

### `vector_search_l2`

Perform vector search using L2 (Euclidean) distance.

**Parameters:**
- Same as `vector_search` but distance_metric fixed to "l2"

---

### `vector_search_cosine`

Perform vector search using cosine distance.

**Parameters:**
- Same as `vector_search` but distance_metric fixed to "cosine"

---

### `vector_search_inner_product`

Perform vector search using inner product.

**Parameters:**
- Same as `vector_search` but distance_metric fixed to "inner_product"

---

### `vector_arithmetic`

Perform vector arithmetic operations.

**Parameters:**
- `operation` (string, required): Operation type
  - Options: `"add"`, `"subtract"`, `"multiply"`, `"normalize"`, `"concat"`, `"norm"`, `"dims"`
- `vector1` (array, required): First vector
- `vector2` (array, optional): Second vector (for add, subtract, concat)
- `scalar` (number, optional): Scalar value (for multiply)

**Example:**
```json
{
  "name": "vector_arithmetic",
  "arguments": {
    "operation": "add",
    "vector1": [1.0, 2.0, 3.0],
    "vector2": [4.0, 5.0, 6.0]
  }
}
```

---

### `vector_distance`

Compute distance between two vectors.

**Parameters:**
- `vector1` (array, required): First vector
- `vector2` (array, required): Second vector
- `metric` (string, default: "l2"): Distance metric
- `p_value` (number, default: 3.0): P value for Minkowski distance
- `covariance` (array, optional): Inverse covariance matrix for Mahalanobis distance

**Example:**
```json
{
  "name": "vector_distance",
  "arguments": {
    "vector1": [1.0, 2.0, 3.0],
    "vector2": [4.0, 5.0, 6.0],
    "metric": "l2"
  }
}
```

---

### `vector_quantize`

Quantize or dequantize vectors.

**Parameters:**
- `operation` (string, required): Operation type
  - Options: `"to_int8"`, `"from_int8"`, `"to_fp16"`, `"from_fp16"`, `"to_binary"`, `"from_binary"`, `"to_uint8"`, `"from_uint8"`, `"to_ternary"`, `"from_ternary"`, `"to_int4"`, `"from_int4"`
- `vector` (array, optional): Input vector (for quantization)
- `data` (string, optional): Base64-encoded quantized data (for dequantization)

**Example:**
```json
{
  "name": "vector_quantize",
  "arguments": {
    "operation": "to_int8",
    "vector": [1.5, 2.3, 3.7]
  }
}
```

---

## Embedding Functions

### `generate_embedding`

Generate text embedding using configured model.

**Parameters:**
- `text` (string, required): Text to embed
- `model` (string, optional): Model name (uses default if not specified)

**Example:**
```json
{
  "name": "generate_embedding",
  "arguments": {
    "text": "Hello world",
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  }
}
```

---

### `embed_image`

Generate image embedding from image bytes.

**Parameters:**
- `image_data` (string, required): Base64-encoded image data
- `model` (string, default: "clip"): Model name

**Example:**
```json
{
  "name": "embed_image",
  "arguments": {
    "image_data": "base64_encoded_image_data",
    "model": "clip"
  }
}
```

---

### `embed_multimodal`

Generate multimodal embedding from text and image.

**Parameters:**
- `text` (string, required): Text input
- `image_data` (string, required): Base64-encoded image data
- `model` (string, default: "clip"): Model name

**Example:**
```json
{
  "name": "embed_multimodal",
  "arguments": {
    "text": "A cat",
    "image_data": "base64_encoded_image_data",
    "model": "clip"
  }
}
```

---

### `configure_embedding_model`

Configure embedding model settings.

**Parameters:**
- `model_name` (string, required): Model name
- `config_json` (string, required): JSON configuration string

**Example:**
```json
{
  "name": "configure_embedding_model",
  "arguments": {
    "model_name": "all-MiniLM-L6-v2",
    "config_json": "{\"batch_size\": 32, \"max_length\": 512}"
  }
}
```

---

## Hybrid Search

### `hybrid_search`

Perform hybrid semantic + lexical search.

**Parameters:**
- `table` (string, required): Table name
- `query_vector` (array, required): Query vector
- `query_text` (string, required): Text query
- `vector_column` (string, required): Vector column name
- `text_column` (string, required): Text column name
- `vector_weight` (number, default: 0.7): Weight for vector search (0.0-1.0)
- `limit` (number, default: 10): Maximum results
- `filters` (object, optional): Optional filters

**Example:**
```json
{
  "name": "hybrid_search",
  "arguments": {
    "table": "documents",
    "query_vector": [0.1, 0.2, 0.3],
    "query_text": "machine learning",
    "vector_column": "embedding",
    "text_column": "content",
    "vector_weight": 0.7,
    "limit": 10
  }
}
```

---

### `reciprocal_rank_fusion`

Perform reciprocal rank fusion on multiple rankings.

**Parameters:**
- `rankings` (array, required): Array of ranking arrays
- `k` (number, default: 60.0): RRF k parameter

**Example:**
```json
{
  "name": "reciprocal_rank_fusion",
  "arguments": {
    "rankings": [[1, 2, 3], [2, 1, 4]],
    "k": 60.0
  }
}
```

---

### `semantic_keyword_search`

Perform semantic + keyword search.

**Parameters:**
- `table` (string, required): Table name
- `semantic_query` (array, required): Semantic query vector
- `keyword_query` (string, required): Keyword query text
- `top_k` (number, default: 10): Number of results

---

### `multi_vector_search`

Search with multiple query vectors.

**Parameters:**
- `table` (string, required): Table name
- `query_vectors` (array, required): Array of query vectors
- `agg_method` (string, default: "max"): Aggregation method
  - Options: `"max"`, `"avg"`, `"sum"`
- `top_k` (number, default: 10): Number of results

---

### `faceted_vector_search`

Faceted vector search with per-facet results.

**Parameters:**
- `table` (string, required): Table name
- `query_vec` (array, required): Query vector
- `facet_column` (string, required): Facet column name
- `per_facet_limit` (number, default: 3): Results per facet

---

### `temporal_vector_search`

Temporal vector search with time decay.

**Parameters:**
- `table` (string, required): Table name
- `query_vec` (array, required): Query vector
- `timestamp_col` (string, required): Timestamp column name
- `decay_rate` (number, default: 0.01): Decay rate per day
- `top_k` (number, default: 10): Number of results

---

### `diverse_vector_search`

Diverse search using Maximal Marginal Relevance (MMR).

**Parameters:**
- `table` (string, required): Table name
- `query_vec` (array, required): Query vector
- `lambda` (number, default: 0.5): Diversity parameter
- `top_k` (number, default: 10): Number of results

---

## Reranking

### `rerank_cross_encoder`

Rerank documents using cross-encoder model.

**Parameters:**
- `query` (string, required): Query text
- `documents` (array, required): Array of document texts
- `model` (string, default: "ms-marco-MiniLM-L-6-v2"): Cross-encoder model name
- `top_k` (number, default: 10): Number of top results

**Example:**
```json
{
  "name": "rerank_cross_encoder",
  "arguments": {
    "query": "What is machine learning?",
    "documents": ["ML is...", "AI is...", "Deep learning..."],
    "model": "ms-marco-MiniLM-L-6-v2",
    "top_k": 10
  }
}
```

---

### `rerank_llm`

Rerank documents using LLM.

**Parameters:**
- `query` (string, required): Query text
- `documents` (array, required): Array of document texts
- `model` (string, default: "gpt-3.5-turbo"): LLM model name
- `top_k` (number, default: 10): Number of top results

---

### `rerank_cohere`

Cohere-style reranking.

**Parameters:**
- `query` (string, required): Query text
- `documents` (array, required): Array of document texts
- `top_n` (number, default: 10): Number of top results

---

### `rerank_colbert`

ColBERT late interaction reranking.

**Parameters:**
- `query` (string, required): Query text
- `docs` (array, required): Array of documents
- `model` (string, default: "colbert-v2"): Model name

---

### `rerank_ltr`

Learning-to-Rank reranking.

**Parameters:**
- `query` (string, required): Query text
- `docs` (array, required): Array of documents
- `features_json` (string, required): JSON features
- `model` (string, required): Model name

---

### `rerank_ensemble`

Ensemble reranking combining multiple models.

**Parameters:**
- `query` (string, required): Query text
- `docs` (array, required): Array of documents
- `models` (array, required): Array of model names
- `weights` (array, required): Array of weights

---

## Machine Learning

### `train_model`

Train an ML model using specified algorithm.

**Parameters:**
- `algorithm` (string, required): ML algorithm
  - Options: `"linear_regression"`, `"ridge"`, `"lasso"`, `"logistic"`, `"random_forest"`, `"svm"`, `"knn"`, `"decision_tree"`, `"naive_bayes"`, `"xgboost"`, `"catboost"`, `"lightgbm"`
- `table` (string, required): Training data table name
- `feature_col` (string, required): Feature column name (vector type)
- `label_col` (string, required): Label column name
- `params` (object, optional): Algorithm-specific parameters
- `project` (string, optional): ML project name

**Example:**
```json
{
  "name": "train_model",
  "arguments": {
    "algorithm": "random_forest",
    "table": "iris",
    "feature_col": "features",
    "label_col": "species",
    "params": {
      "n_trees": 100,
      "max_depth": 10
    }
  }
}
```

---

### `predict`

Perform prediction using a trained model.

**Parameters:**
- `model_id` (number, required): Trained model ID
- `features` (array, required): Feature vector

**Example:**
```json
{
  "name": "predict",
  "arguments": {
    "model_id": 1,
    "features": [1.0, 2.0, 3.0]
  }
}
```

---

### `evaluate_model`

Evaluate a trained model.

**Parameters:**
- `model_id` (number, required): Trained model ID
- `table` (string, required): Evaluation data table
- `feature_col` (string, required): Feature column name
- `label_col` (string, required): Label column name

**Response:**
```json
{
  "accuracy": 0.95,
  "precision": 0.94,
  "recall": 0.96,
  "f1": 0.95,
  "confusion_matrix": [[90, 5], [3, 92]]
}
```

---

### `list_models`

List all trained models.

**Parameters:**
- `algorithm` (string, optional): Filter by algorithm
- `project` (string, optional): Filter by project

**Response:**
```json
{
  "models": [
    {
      "model_id": 1,
      "algorithm": "random_forest",
      "created_at": "2025-01-01T00:00:00Z",
      "metrics": {
        "accuracy": 0.95
      }
    }
  ]
}
```

---

### `get_model_info`

Get detailed information about a model.

**Parameters:**
- `model_id` (number, required): Model ID

---

### `delete_model`

Delete a trained model.

**Parameters:**
- `model_id` (number, required): Model ID

---

## RAG Operations

### `process_document`

Process document for RAG pipeline.

**Parameters:**
- `document` (string, required): Document text
- `chunk_size` (number, default: 512): Chunk size
- `overlap` (number, default: 128): Overlap between chunks
- `separator` (string, optional): Separator text

**Response:**
```json
{
  "chunks": [
    {
      "chunk_id": 1,
      "chunk_text": "Chunk text...",
      "start_pos": 0,
      "end_pos": 512
    }
  ]
}
```

---

### `retrieve_context`

Retrieve context for RAG query.

**Parameters:**
- `query` (string, required): Query text
- `table` (string, required): Table name
- `vector_column` (string, required): Vector column name
- `top_k` (number, default: 10): Number of results

---

### `generate_response`

Generate response using RAG.

**Parameters:**
- `query` (string, required): Query text
- `context` (array, required): Array of context texts
- `model` (string, optional): LLM model name
- `params` (object, optional): LLM parameters

---

### `ingest_documents`

Ingest documents into RAG pipeline.

**Parameters:**
- `table` (string, required): Table name
- `documents` (array, required): Array of document texts
- `vector_column` (string, required): Vector column name
- `text_column` (string, required): Text column name

---

### `answer_with_citations`

Generate answer with citations.

**Parameters:**
- `query` (string, required): Query text
- `table` (string, required): Table name
- `vector_column` (string, required): Vector column name
- `text_column` (string, required): Text column name
- `top_k` (number, default: 5): Number of context chunks

---

## Indexing

### `create_hnsw_index`

Create HNSW index for vector column.

**Parameters:**
- `table` (string, required): Table name
- `vector_column` (string, required): Vector column name
- `index_name` (string, required): Index name
- `m` (number, default: 16): HNSW parameter M (2-128)
- `ef_construction` (number, default: 200): HNSW parameter ef_construction (4-2000)

**Example:**
```json
{
  "name": "create_hnsw_index",
  "arguments": {
    "table": "documents",
    "vector_column": "embedding",
    "index_name": "documents_hnsw_idx",
    "m": 16,
    "ef_construction": 200
  }
}
```

---

### `create_ivf_index`

Create IVF index for vector column.

**Parameters:**
- `table` (string, required): Table name
- `vector_column` (string, required): Vector column name
- `index_name` (string, required): Index name
- `num_lists` (number, default: 100): Number of lists
- `probes` (number, default: 10): Number of probes

---

### `index_status`

Get index status and statistics.

**Parameters:**
- `index_name` (string, required): Index name

**Response:**
```json
{
  "index_name": "documents_hnsw_idx",
  "type": "hnsw",
  "size_mb": 1024.5,
  "num_vectors": 1000000,
  "build_time_seconds": 120.5,
  "status": "ready"
}
```

---

### `drop_index`

Drop an index.

**Parameters:**
- `index_name` (string, required): Index name

---

### `tune_hnsw_index`

Automatically optimize HNSW index parameters.

**Parameters:**
- `table` (string, required): Table name
- `vector_column` (string, required): Vector column name

---

### `tune_ivf_index`

Automatically optimize IVF index parameters.

**Parameters:**
- `table` (string, required): Table name
- `vector_column` (string, required): Vector column name

---

## PostgreSQL Administration

NeuronMCP provides **27 PostgreSQL administration tools** for complete database management.

### Server Information

#### `postgresql_version`

Get PostgreSQL server version information.

**Parameters:** None

**Response:**
```json
{
  "version": "PostgreSQL 16.1",
  "server_version": "16.1",
  "server_version_num": 160001,
  "major_version": 16,
  "minor_version": 1,
  "patch_version": 1
}
```

---

#### `postgresql_stats`

Get comprehensive PostgreSQL server statistics.

**Parameters:**
- `include_database_stats` (boolean, default: true): Include database-level statistics
- `include_table_stats` (boolean, default: true): Include table statistics
- `include_connection_stats` (boolean, default: true): Include connection statistics
- `include_performance_stats` (boolean, default: true): Include performance metrics

**Response:**
```json
{
  "database_stats": {
    "total_size_mb": 10240,
    "num_databases": 5
  },
  "table_stats": {
    "total_tables": 100,
    "total_size_mb": 5120
  },
  "connection_stats": {
    "active_connections": 10,
    "max_connections": 100
  },
  "performance_stats": {
    "cache_hit_ratio": 0.95,
    "index_usage_ratio": 0.85
  }
}
```

---

#### `postgresql_databases`

List all PostgreSQL databases with sizes and connection counts.

**Parameters:**
- `include_system` (boolean, default: false): Include system databases

---

#### `postgresql_settings`

Get PostgreSQL configuration settings.

**Parameters:**
- `pattern` (string, optional): Filter settings by name pattern

---

#### `postgresql_extensions`

List installed PostgreSQL extensions.

**Parameters:** None

---

### Database Object Management

#### `postgresql_tables`

List all tables with metadata.

**Parameters:**
- `schema` (string, optional): Filter by schema name
- `include_system` (boolean, default: false): Include system tables

---

#### `postgresql_indexes`

List all indexes with statistics.

**Parameters:**
- `schema` (string, optional): Filter by schema name
- `table` (string, optional): Filter by table name
- `include_system` (boolean, default: false): Include system indexes

---

#### `postgresql_schemas`

List all schemas with ownership and permissions.

**Parameters:**
- `include_system` (boolean, default: false): Include system schemas

---

#### `postgresql_views`

List all views with definitions.

**Parameters:**
- `schema` (string, optional): Filter by schema name

---

#### `postgresql_functions`

List all functions with signatures.

**Parameters:**
- `schema` (string, optional): Filter by schema name

---

### Monitoring

#### `postgresql_connections`

Get connection information.

**Parameters:**
- `include_idle` (boolean, default: false): Include idle connections

---

#### `postgresql_locks`

Get lock information.

**Parameters:**
- `lock_type` (string, optional): Filter by lock type

---

#### `postgresql_replication`

Get replication status.

**Parameters:** None

---

#### `postgresql_queries`

Get active query information.

**Parameters:**
- `include_idle` (boolean, default: false): Include idle queries

---

#### `postgresql_wait_events`

Get wait event statistics.

**Parameters:** None

---

### Performance

#### `postgresql_cache_hit_ratio`

Get cache hit ratio statistics.

**Parameters:** None

---

#### `postgresql_table_bloat`

Get table bloat information.

**Parameters:**
- `schema` (string, optional): Filter by schema name

---

#### `postgresql_index_bloat`

Get index bloat information.

**Parameters:**
- `schema` (string, optional): Filter by schema name

---

#### `postgresql_vacuum_stats`

Get vacuum statistics.

**Parameters:** None

---

#### `postgresql_autovacuum`

Get autovacuum status.

**Parameters:** None

---

### Maintenance

#### `postgresql_analyze`

Run ANALYZE on tables.

**Parameters:**
- `table` (string, required): Table name
- `schema` (string, optional): Schema name

---

#### `postgresql_vacuum`

Run VACUUM on tables.

**Parameters:**
- `table` (string, required): Table name
- `schema` (string, optional): Schema name
- `full` (boolean, default: false): Full vacuum
- `analyze` (boolean, default: false): Run ANALYZE

---

#### `postgresql_reindex`

Reindex tables or indexes.

**Parameters:**
- `table` (string, optional): Table name
- `index` (string, optional): Index name
- `schema` (string, optional): Schema name

---

## Analytics

### `cluster_data`

Perform clustering on vector data.

**Parameters:**
- `algorithm` (string, default: "kmeans"): Clustering algorithm
  - Options: `"kmeans"`, `"gmm"`, `"dbscan"`, `"hierarchical"`, `"minibatch_kmeans"`
- `table` (string, required): Table name
- `vector_column` (string, required): Vector column name
- `k` (number, required): Number of clusters
- `eps` (number, default: 0.5): Maximum distance for DBSCAN

---

### `detect_outliers`

Detect outliers in vector data.

**Parameters:**
- `method` (string, required): Outlier detection method
  - Options: `"z_score"`, `"modified_z_score"`, `"isolation_forest"`
- `table` (string, required): Table name
- `vector_column` (string, required): Vector column name
- `threshold` (number, optional): Outlier threshold

---

### `reduce_dimensionality`

Reduce dimensionality using PCA.

**Parameters:**
- `table` (string, required): Table name
- `vector_column` (string, required): Vector column name
- `n_components` (number, required): Number of components

---

### `quality_metrics`

Compute quality metrics for search results.

**Parameters:**
- `metric` (string, required): Metric type
  - Options: `"recall_at_k"`, `"precision_at_k"`, `"f1_at_k"`, `"mrr"`, `"davies_bouldin"`
- `table` (string, required): Table name
- `k` (number, optional): K value for @K metrics
- `ground_truth_col` (string, optional): Ground truth column
- `predicted_col` (string, optional): Predicted results column

---

### `detect_drift`

Detect data drift.

**Parameters:**
- `method` (string, required): Drift detection method
  - Options: `"centroid"`, `"distribution"`, `"temporal"`
- `table` (string, required): Table name
- `vector_column` (string, required): Vector column name
- `reference_table` (string, optional): Reference table for comparison
- `threshold` (number, optional): Drift threshold

---

### `topic_discovery`

Discover topics in text data.

**Parameters:**
- `table` (string, required): Table name
- `text_column` (string, required): Text column name
- `num_topics` (number, required): Number of topics
- `vector_column` (string, optional): Vector column for embeddings

---

## ONNX Tools

### `onnx_info`

Get ONNX Runtime information.

**Parameters:** None

**Response:**
```json
{
  "available": true,
  "version": "1.16.0",
  "providers": ["CPUExecutionProvider", "CUDAExecutionProvider"]
}
```

---

### `onnx_predict`

Perform prediction using ONNX model.

**Parameters:**
- `model_path` (string, required): Path to ONNX model
- `input_data` (array, required): Input data
- `use_gpu` (boolean, default: true): Use GPU if available

---

## GPU Tools

### `gpu_info`

Get GPU information.

**Parameters:** None

**Response:**
```json
{
  "available": true,
  "backend": "cuda",
  "device_id": 0,
  "device_name": "NVIDIA GeForce RTX 4090",
  "total_memory_mb": 24576,
  "free_memory_mb": 20480
}
```

---

### `gpu_utilization`

Get GPU utilization metrics.

**Parameters:** None

**Response:**
```json
{
  "device_id": 0,
  "utilization_pct": 75.5,
  "memory_used_mb": 4096,
  "memory_total_mb": 24576,
  "temperature_c": 65,
  "power_w": 250.5
}
```

---

## Workers

### `worker_status`

Get background worker status.

**Parameters:** None

**Response:**
```json
{
  "neuranq": {
    "enabled": true,
    "queue_depth": 100,
    "processed_jobs": 1000
  },
  "neuranmon": {
    "enabled": true,
    "last_tune": "2025-01-01T00:00:00Z"
  },
  "neurandefrag": {
    "enabled": true,
    "last_maintenance": "2025-01-01T00:00:00Z"
  }
}
```

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Error message",
    "details": {}
  }
}
```

### Common Error Codes

- `VALIDATION_ERROR`: Invalid parameters
- `QUERY_ERROR`: SQL query execution error
- `NOT_FOUND`: Resource not found
- `PERMISSION_DENIED`: Insufficient permissions
- `INTERNAL_ERROR`: Internal server error

---

## Related Documentation

- [NeuronMCP Setup Guide](../neuronmcp/NEURONDB_MCP_SETUP.md)
- [PostgreSQL Tools](POSTGRESQL_TOOLS.md)
- [Tool Catalog](../neuronmcp/docs/tool-resource-catalog.md)

---

**Last Updated:** 2025-01-01  
**Documentation Version:** 1.0.0

