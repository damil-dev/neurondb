# NeuronMCP Registered Tools

Complete list of all tools registered in NeuronMCP server.

## Tool Categories

NeuronMCP registers tools in the following categories:

### 1. Vector Search Tools (8 tools)

- `vector_search` - Vector search with default distance metric
- `vector_search_l2` - Vector search with L2 (Euclidean) distance
- `vector_search_cosine` - Vector search with cosine similarity
- `vector_search_inner_product` - Vector search with inner product
- `vector_search_l1` - Vector search with L1 (Manhattan) distance
- `vector_search_hamming` - Vector search with Hamming distance
- `vector_search_chebyshev` - Vector search with Chebyshev distance
- `vector_search_minkowski` - Vector search with Minkowski distance

### 2. Embedding Tools (9 tools)

- `generate_embedding` - Generate embedding for a single text
- `batch_embedding` - Generate embeddings for multiple texts
- `embed_image` - Generate embedding for an image
- `embed_multimodal` - Generate multimodal embedding
- `embed_cached` - Generate cached embedding
- `configure_embedding_model` - Configure embedding model
- `get_embedding_model_config` - Get embedding model configuration
- `list_embedding_model_configs` - List all embedding model configurations
- `delete_embedding_model_config` - Delete embedding model configuration

### 3. Vector Operations Tools (6 tools)

- `vector_similarity` - Calculate similarity between two vectors
- `vector_similarity_unified` - Unified vector similarity calculation
- `vector_arithmetic` - Perform arithmetic operations on vectors
- `vector_distance` - Calculate distance between vectors
- `vector_add` - Add vectors
- `vector_norm` - Calculate vector norm

### 4. Index Management Tools (8 tools)

- `create_vector_index` - Create a vector index
- `create_hnsw_index` - Create HNSW index
- `create_ivf_index` - Create IVF index
- `index_status` - Get index status
- `drop_index` - Drop an index
- `tune_hnsw_index` - Tune HNSW index parameters
- `tune_ivf_index` - Tune IVF index parameters
- `vector_quantization` - Vector quantization operations
- `quantization_analysis` - Quantization analysis

### 5. RAG (Retrieval-Augmented Generation) Tools (6 tools)

- `process_document` - Process and chunk a document
- `retrieve_context` - Retrieve relevant context for a query
- `generate_response` - Generate response using retrieved context
- `chunk_document` - Chunk a document into segments
- `ingest_documents` - Ingest multiple documents
- `answer_with_citations` - Generate answer with citations

### 6. Machine Learning Tools (9 tools)

- `train_model` - Train an ML model
- `predict` - Make predictions using a trained model
- `predict_batch` - Make batch predictions
- `evaluate_model` - Evaluate model performance
- `list_models` - List all trained models
- `get_model_info` - Get model information
- `delete_model` - Delete a model
- `export_model` - Export a model
- `automl` - Automated machine learning

### 7. Analytics Tools (7 tools)

- `cluster_data` - Perform clustering on data
- `detect_outliers` - Detect outliers in data
- `reduce_dimensionality` - Reduce data dimensionality
- `analyze_data` - Analyze data
- `quality_metrics` - Compute quality metrics
- `detect_drift` - Detect data drift
- `topic_discovery` - Discover topics in text data

### 8. Hybrid Search Tools (8 tools)

- `hybrid_search` - Hybrid semantic + lexical search
- `text_search` - Text-only search
- `reciprocal_rank_fusion` - Reciprocal rank fusion
- `semantic_keyword_search` - Semantic + keyword search
- `multi_vector_search` - Multi-vector search
- `faceted_vector_search` - Faceted vector search
- `temporal_vector_search` - Temporal vector search
- `diverse_vector_search` - Diverse vector search

### 9. Reranking Tools (6 tools)

- `rerank_cross_encoder` - Rerank using cross-encoder
- `rerank_llm` - Rerank using LLM
- `rerank_cohere` - Rerank using Cohere
- `rerank_colbert` - Rerank using ColBERT
- `rerank_ltr` - Rerank using Learning-to-Rank
- `rerank_ensemble` - Ensemble reranking

### 10. PostgreSQL Tools (29+ tools)

#### Server Information (5 tools)
- `postgresql_version` - Get PostgreSQL version information
- `postgresql_stats` - Get PostgreSQL server statistics
- `postgresql_databases` - List all databases
- `postgresql_settings` - Get configuration settings
- `postgresql_extensions` - List installed extensions

#### Database Object Management (8 tools)
- `postgresql_tables` - List all tables
- `postgresql_indexes` - List all indexes
- `postgresql_schemas` - List all schemas
- `postgresql_views` - List all views
- `postgresql_sequences` - List all sequences
- `postgresql_functions` - List all functions
- `postgresql_triggers` - List all triggers
- `postgresql_constraints` - List all constraints

#### User and Role Management (3 tools)
- `postgresql_users` - List all users
- `postgresql_roles` - List all roles
- `postgresql_permissions` - List permissions

#### Performance and Statistics (7 tools)
- `postgresql_table_stats` - Get table statistics
- `postgresql_index_stats` - Get index statistics
- `postgresql_active_queries` - Get active queries
- `postgresql_wait_events` - Get wait events
- `postgresql_connections` - Get connection information
- `postgresql_locks` - Get lock information
- `postgresql_replication` - Get replication status

#### Query Execution (4 tools)
- `postgresql_execute_query` - Execute SQL query
- `postgresql_query_plan` - Get query execution plan
- `postgresql_query_history` - Get query history
- `postgresql_query_optimization` - Get query optimization suggestions

#### Size and Storage (4 tools)
- `postgresql_table_sizes` - Get table sizes
- `postgresql_index_sizes` - Get index sizes
- `postgresql_bloat` - Detect table/index bloat
- `postgresql_vacuum_stats` - Get vacuum statistics

### 11. Advanced Tools (6 tools)

- `time_series` - Time series operations
- `onnx_model` - ONNX model operations
- `vector_graph` - Vector graph operations
- `vecmap_operations` - Vecmap operations
- `worker_management` - Worker management
- `gpu_info` - GPU information and monitoring

### 12. Dataset Loading (1 tool)

- `load_dataset` - Load dataset from HuggingFace or other sources

## Total Tool Count

Approximately **100+ tools** are registered in NeuronMCP.

## Usage in Claude Desktop

When NeuronMCP is configured in Claude Desktop, all these tools become available. You can:

1. **Check PostgreSQL version:**
   ```
   Use the postgresql_version tool
   ```

2. **Generate embeddings:**
   ```
   Use the generate_embedding tool with text="your text here"
   ```

3. **Retrieve context for RAG:**
   ```
   Use the retrieve_context tool with query="your query", table="documents", vector_column="embedding"
   ```

## Tool Naming Convention

- Vector tools: `vector_*` or `*_vector_*`
- Embedding tools: `generate_embedding`, `batch_embedding`, `embed_*`
- RAG tools: `process_document`, `retrieve_context`, `generate_response`, `chunk_document`
- PostgreSQL tools: `postgresql_*`
- ML tools: `train_model`, `predict`, `evaluate_model`, `list_models`
- Analytics tools: `cluster_data`, `detect_outliers`, `analyze_data`

## See Also

- `TOOLS_REFERENCE.md` - Detailed tool reference with parameters
- `POSTGRESQL_TOOLS.md` - Complete PostgreSQL tools documentation
- `test_rag_sample_data.py` - Example script using MCP tools
- `README.md` - Main NeuronMCP documentation

