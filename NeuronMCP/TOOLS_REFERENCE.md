# NeuronMCP Tools Reference - Complete Guide

**The Most Comprehensive MCP Server Tool Reference**

Complete reference for all **100+ NeuronMCP tools** with detailed parameters, comprehensive examples, use cases, and error codes.

**Total Tools: 100+** covering:
- **27 PostgreSQL Administration Tools** - Complete database management and monitoring
- **70+ NeuronDB Tools** - Vector search, ML, RAG, analytics, and more

This is the definitive reference for the most complete and powerful MCP server available, providing 100% coverage of PostgreSQL administration combined with cutting-edge AI/ML capabilities through NeuronDB.

## Vector Operations

### vector_search
Perform vector similarity search using various distance metrics.

**Parameters:**
- `table` (string, required): Table name containing vectors
- `vector_column` (string, required): Name of the vector column
- `query_vector` (array, required): Query vector for similarity search
- `limit` (number, default: 10): Maximum number of results (1-1000)
- `distance_metric` (string, default: "l2"): Distance metric (l2, cosine, inner_product, l1, hamming, chebyshev, minkowski)
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
    "distance_metric": "cosine"
  }
}
```

### vector_arithmetic
Perform vector arithmetic operations.

**Parameters:**
- `operation` (string, required): Operation (add, subtract, multiply, normalize, concat, norm, dims)
- `vector1` (array, required): First vector
- `vector2` (array, optional): Second vector (for add, subtract, concat)
- `scalar` (number, optional): Scalar value (for multiply)

### vector_distance
Compute distance between two vectors using various metrics.

**Parameters:**
- `vector1` (array, required): First vector
- `vector2` (array, required): Second vector
- `metric` (string, default: "l2"): Distance metric
- `p_value` (number, default: 3.0): P value for Minkowski distance
- `covariance` (array, optional): Inverse covariance matrix for Mahalanobis distance

### vector_quantize
Quantize or dequantize vectors.

**Parameters:**
- `operation` (string, required): Operation (to_int8, from_int8, to_fp16, from_fp16, to_binary, from_binary, etc.)
- `vector` (array, optional): Input vector (for quantization)
- `data` (string, optional): Base64-encoded quantized data (for dequantization)

## Embedding Functions

### generate_embedding
Generate text embedding using configured model.

**Parameters:**
- `text` (string, required): Text to embed
- `model` (string, optional): Model name (uses default if not specified)

### embed_image
Generate image embedding from image bytes.

**Parameters:**
- `image_data` (string, required): Base64-encoded image data
- `model` (string, default: "clip"): Model name

### embed_multimodal
Generate multimodal embedding from text and image.

**Parameters:**
- `text` (string, required): Text input
- `image_data` (string, required): Base64-encoded image data
- `model` (string, default: "clip"): Model name

### configure_embedding_model
Configure embedding model settings.

**Parameters:**
- `model_name` (string, required): Model name
- `config_json` (string, required): JSON configuration string

## Hybrid Search

### hybrid_search
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

### reciprocal_rank_fusion
Perform reciprocal rank fusion on multiple rankings.

**Parameters:**
- `rankings` (array, required): Array of ranking arrays
- `k` (number, default: 60.0): RRF k parameter

### semantic_keyword_search
Perform semantic + keyword search.

**Parameters:**
- `table` (string, required): Table name
- `semantic_query` (array, required): Semantic query vector
- `keyword_query` (string, required): Keyword query text
- `top_k` (number, default: 10): Number of results

## Reranking

### rerank_cross_encoder
Rerank documents using cross-encoder model.

**Parameters:**
- `query` (string, required): Query text
- `documents` (array, required): Array of document texts
- `model` (string, default: "ms-marco-MiniLM-L-6-v2"): Cross-encoder model name
- `top_k` (number, default: 10): Number of top results

### rerank_llm
Rerank documents using LLM.

**Parameters:**
- `query` (string, required): Query text
- `documents` (array, required): Array of document texts
- `model` (string, default: "gpt-3.5-turbo"): LLM model name
- `top_k` (number, default: 10): Number of top results

## Machine Learning

### train_model
Train an ML model using specified algorithm.

**Parameters:**
- `algorithm` (string, required): ML algorithm (linear_regression, ridge, lasso, logistic, random_forest, svm, knn, decision_tree, naive_bayes)
- `table` (string, required): Training data table name
- `feature_col` (string, required): Feature column name (vector type)
- `label_col` (string, required): Label column name
- `params` (object, optional): Algorithm-specific parameters
- `project` (string, optional): ML project name

### predict
Perform prediction using a trained model.

**Parameters:**
- `model_id` (number, required): Trained model ID
- `features` (array, required): Feature vector

### evaluate_model
Evaluate a trained model.

**Parameters:**
- `model_id` (number, required): Trained model ID
- `table` (string, required): Evaluation data table
- `feature_col` (string, required): Feature column name
- `label_col` (string, required): Label column name

## Analytics

### cluster_data
Perform clustering on vector data.

**Parameters:**
- `algorithm` (string, default: "kmeans"): Clustering algorithm (kmeans, gmm, dbscan, hierarchical, minibatch_kmeans)
- `table` (string, required): Table name
- `vector_column` (string, required): Vector column name
- `k` (number, required): Number of clusters
- `eps` (number, default: 0.5): Maximum distance for DBSCAN

### quality_metrics
Compute quality metrics for search results.

**Parameters:**
- `metric` (string, required): Metric type (recall_at_k, precision_at_k, f1_at_k, mrr, davies_bouldin)
- `table` (string, required): Table name
- `k` (number, optional): K value for @K metrics
- `ground_truth_col` (string, optional): Ground truth column
- `predicted_col` (string, optional): Predicted results column

### detect_drift
Detect data drift.

**Parameters:**
- `method` (string, required): Drift detection method (centroid, distribution, temporal)
- `table` (string, required): Table name
- `vector_column` (string, required): Vector column name
- `reference_table` (string, optional): Reference table for comparison
- `threshold` (number, optional): Drift threshold

## Index Management

### create_hnsw_index
Create HNSW index for vector column.

**Parameters:**
- `table` (string, required): Table name
- `vector_column` (string, required): Vector column name
- `index_name` (string, required): Index name
- `m` (number, default: 16): HNSW parameter M (2-128)
- `ef_construction` (number, default: 200): HNSW parameter ef_construction (4-2000)

### create_ivf_index
Create IVF index for vector column.

**Parameters:**
- `table` (string, required): Table name
- `vector_column` (string, required): Vector column name
- `index_name` (string, required): Index name
- `num_lists` (number, default: 100): Number of lists
- `probes` (number, default: 10): Number of probes

### tune_hnsw_index
Automatically optimize HNSW index parameters.

**Parameters:**
- `table` (string, required): Table name
- `vector_column` (string, required): Vector column name

## PostgreSQL Tools

Complete PostgreSQL administration and monitoring tools (29 tools total).

### Server Information

#### postgresql_version
Get PostgreSQL server version information.

**Parameters:** None

#### postgresql_stats
Get comprehensive PostgreSQL server statistics.

**Parameters:**
- `include_database_stats` (boolean, default: true): Include database-level statistics
- `include_table_stats` (boolean, default: true): Include table statistics
- `include_connection_stats` (boolean, default: true): Include connection statistics
- `include_performance_stats` (boolean, default: true): Include performance metrics

#### postgresql_databases
List all PostgreSQL databases with sizes and connection counts.

**Parameters:**
- `include_system` (boolean, default: false): Include system databases

#### postgresql_settings
Get PostgreSQL configuration settings.

**Parameters:**
- `pattern` (string, optional): Filter settings by name pattern

#### postgresql_extensions
List installed PostgreSQL extensions.

**Parameters:** None

### Database Object Management

#### postgresql_tables
List all tables with metadata (schema, owner, size, row count).

**Parameters:**
- `schema` (string, optional): Filter by schema name
- `include_system` (boolean, default: false): Include system tables

#### postgresql_indexes
List all indexes with statistics (size, usage, scan counts).

**Parameters:**
- `schema` (string, optional): Filter by schema name
- `table` (string, optional): Filter by table name
- `include_system` (boolean, default: false): Include system indexes

#### postgresql_schemas
List all schemas with ownership and permissions.

**Parameters:**
- `include_system` (boolean, default: false): Include system schemas

#### postgresql_views
List all views with definitions.

**Parameters:**
- `schema` (string, optional): Filter by schema name
- `include_system` (boolean, default: false): Include system views
- `include_definition` (boolean, default: true): Include view definition SQL

#### postgresql_sequences
List all sequences with current values and ranges.

**Parameters:**
- `schema` (string, optional): Filter by schema name
- `include_system` (boolean, default: false): Include system sequences

#### postgresql_functions
List all functions with parameters and return types.

**Parameters:**
- `schema` (string, optional): Filter by schema name
- `include_system` (boolean, default: false): Include system functions

#### postgresql_triggers
List all triggers with event types and functions.

**Parameters:**
- `schema` (string, optional): Filter by schema name
- `table` (string, optional): Filter by table name

#### postgresql_constraints
List constraints (primary keys, foreign keys, unique, check).

**Parameters:**
- `schema` (string, optional): Filter by schema name
- `table` (string, optional): Filter by table name
- `constraint_type` (string, optional): Filter by type (primary_key, foreign_key, unique, check, not_null)

### User and Role Management

#### postgresql_users
List all users with login and connection info.

**Parameters:** None

#### postgresql_roles
List all roles with membership and attributes.

**Parameters:** None

#### postgresql_permissions
List database object permissions (tables, functions, etc.).

**Parameters:**
- `schema` (string, optional): Filter by schema name
- `object_type` (string, optional): Filter by type (table, function, sequence, schema)

### Performance and Statistics

#### postgresql_table_stats
Get detailed per-table statistics (scans, inserts, updates, deletes, tuples).

**Parameters:**
- `schema` (string, optional): Filter by schema name
- `table` (string, optional): Filter by table name

#### postgresql_index_stats
Get detailed per-index statistics (scans, size, bloat).

**Parameters:**
- `schema` (string, optional): Filter by schema name
- `table` (string, optional): Filter by table name

#### postgresql_active_queries
Show currently active/running queries with details.

**Parameters:**
- `include_idle` (boolean, default: false): Include idle queries
- `limit` (number, default: 100, max: 1000): Maximum number of queries to return

#### postgresql_wait_events
Show wait events and blocking queries.

**Parameters:** None

#### postgresql_connections
Get detailed PostgreSQL connection information.

**Parameters:** None

#### postgresql_locks
Get PostgreSQL lock information.

**Parameters:** None

#### postgresql_replication
Get PostgreSQL replication status.

**Parameters:** None

### Size and Storage

#### postgresql_table_size
Get size of specific tables (with options for total, indexes, toast).

**Parameters:**
- `schema` (string, optional): Schema name
- `table` (string, optional): Table name
- `include_indexes` (boolean, default: true): Include index sizes
- `include_toast` (boolean, default: true): Include TOAST size

#### postgresql_index_size
Get size of specific indexes.

**Parameters:**
- `schema` (string, optional): Filter by schema name
- `table` (string, optional): Filter by table name
- `index` (string, optional): Filter by index name

#### postgresql_bloat
Check table and index bloat (estimated).

**Parameters:**
- `schema` (string, optional): Filter by schema name
- `table` (string, optional): Filter by table name
- `min_bloat_percent` (number, default: 10, range: 0-100): Minimum bloat percentage to report

#### postgresql_vacuum_stats
Get vacuum statistics and recommendations.

**Parameters:**
- `schema` (string, optional): Filter by schema name
- `table` (string, optional): Filter by table name

## Vector Graph Operations

### vector_graph
Perform graph operations on vgraph type: BFS, DFS, PageRank, community detection.

**Parameters:**
- `operation` (string, required): Operation (bfs, dfs, pagerank, community_detection)
- `graph` (string, required): vgraph value as string
- `start_node` (number, optional): Starting node index (for BFS, DFS)
- `max_depth` (number, optional): Maximum depth for BFS (-1 for unlimited)
- `damping_factor` (number, default: 0.85): Damping factor for PageRank
- `max_iterations` (number, default: 100): Maximum iterations
- `tolerance` (number, default: 1e-6): Convergence tolerance for PageRank

## Vecmap Operations

### vecmap_operations
Perform operations on vecmap (sparse vector) type.

**Parameters:**
- `operation` (string, required): Operation (l2_distance, cosine_distance, inner_product, l1_distance, add, subtract, multiply_scalar, norm)
- `vecmap1` (string, required): First vecmap (base64-encoded bytea)
- `vecmap2` (string, optional): Second vecmap (for distance/arithmetic)
- `scalar` (number, optional): Scalar value (for multiply_scalar)

## Dataset Loading

### load_dataset
Load HuggingFace dataset into database.

**Parameters:**
- `dataset_name` (string, required): HuggingFace dataset name
- `split` (string, default: "train"): Dataset split
- `config` (string, optional): Dataset configuration name
- `streaming` (boolean, default: false): Enable streaming mode
- `cache_dir` (string, optional): Cache directory path

## Error Codes

- `VALIDATION_ERROR`: Invalid parameters provided
- `EXECUTION_ERROR`: Query execution failed
- `SEARCH_ERROR`: Vector search failed
- `TRAINING_ERROR`: Model training failed
- `PREDICTION_ERROR`: Prediction failed
- `QUERY_ERROR`: Database query error
- `CONNECTION_ERROR`: Database connection error

