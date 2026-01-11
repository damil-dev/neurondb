# Top 20 NeuronDB Functions You Need

The most commonly used SQL functions in NeuronDB, organized by task. For the complete API reference, see [SQL API Reference](sql-api-complete.md).

## Vector Search & Similarity

### 1. `embed_text(text, model)` → `vector`
Generate text embeddings for semantic search.

```sql
SELECT embed_text('Hello world', 'all-MiniLM-L6-v2');
-- Returns: vector(384)
```

### 2. `embed_text_batch(text[], model)` → `vector[]`
Batch embedding generation (more efficient).

```sql
SELECT embed_text_batch(
  ARRAY['text1', 'text2', 'text3'],
  'all-MiniLM-L6-v2'
);
```

### 3. Distance Operators: `<->`, `<=>`, `<#>`
Vector similarity operators.

```sql
-- L2 distance
SELECT embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance;

-- Cosine distance
SELECT embedding <=> '[0.1, 0.2, 0.3]'::vector AS distance;

-- Inner product
SELECT embedding <#> '[0.1, 0.2, 0.3]'::vector AS distance;
```

## Indexing

### 4. `CREATE INDEX ... USING hnsw`
Create HNSW index for fast approximate nearest neighbor search.

```sql
CREATE INDEX documents_embedding_idx 
ON documents 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);
```

### 5. `CREATE INDEX ... USING ivf`
Create IVF index for faster builds on large datasets.

```sql
CREATE INDEX documents_embedding_idx 
ON documents 
USING ivf (embedding vector_cosine_ops)
WITH (lists = 1024);
```

## Search & Retrieval

### 6. Vector Similarity Search
Standard vector search query pattern.

```sql
SELECT id, content, embedding <=> query_vector AS distance
FROM documents
ORDER BY embedding <=> query_vector
LIMIT 10;
```

### 7. `mmr_rerank(table, column, query_vector, k, lambda)`
Maximal Marginal Relevance reranking for diversity.

```sql
SELECT * FROM neurondb.mmr_rerank(
  'documents', 'embedding', 
  '[0.1, 0.2, 0.3]'::vector,
  10, 0.7
);
```

### 8. `rerank_cross_encoder(query, candidates, model, top_k)`
Neural reranking using cross-encoder models.

```sql
SELECT * FROM neurondb.rerank_cross_encoder(
  'query text',
  ARRAY['candidate1', 'candidate2', 'candidate3'],
  NULL,  -- use default model
  5
);
```

## Hybrid Search

### 9. Hybrid Vector + Full-Text Search
Combine vector similarity with PostgreSQL full-text search.

```sql
SELECT id, content,
  (embedding <=> query_vector) * 0.7 + 
  (ts_rank(to_tsvector('english', content), query_tsquery) * 0.3) AS score
FROM documents
WHERE to_tsvector('english', content) @@ query_tsquery
ORDER BY score DESC
LIMIT 10;
```

### 10. `rrf_rerank(vector_results, text_results, k, rrf_k)`
Reciprocal Rank Fusion for combining multiple result sets.

```sql
SELECT * FROM neurondb.rrf_rerank(
  vector_results, text_results, 10, 60
);
```

## RAG Pipeline

### 11. `chunk_text(text, chunk_size, overlap)`
Split documents into chunks for RAG.

```sql
SELECT neurondb.chunk_text(
  'Long document text...',
  500,  -- chunk size in tokens
  50    -- overlap in tokens
);
```

### 12. `retrieve_context(query_vector, table, column, k, filters)`
Retrieve context for RAG queries.

```sql
SELECT * FROM neurondb.retrieve_context(
  query_vector,
  'documents', 'embedding',
  10,
  'category = ''tech'''
);
```

### 13. `rag_evaluate(query, retrieved_docs, generated_answer)`
Evaluate RAG pipeline quality using RAGAS metrics.

```sql
SELECT * FROM neurondb.rag_evaluate(
  'What is machine learning?',
  ARRAY['doc1', 'doc2', 'doc3'],
  'Machine learning is...'
);
-- Returns: faithfulness, relevancy, context_precision scores
```

## Machine Learning

### 14. `train(algorithm, table, feature_col, label_col, params)`
Train ML models in-database.

```sql
-- Train K-Means clustering
SELECT neurondb.train(
  'kmeans',
  'features_table', 'feature_vector',
  NULL,  -- no labels for clustering
  '{"k": 5, "max_iters": 100}'::jsonb
);

-- Train classifier
SELECT neurondb.train(
  'random_forest',
  'training_data', 'features', 'label',
  '{"n_trees": 100, "max_depth": 10}'::jsonb
);
```

### 15. `predict(model_id, features)`
Make predictions using trained models.

```sql
SELECT neurondb.predict(
  model_id,
  '[0.1, 0.2, 0.3]'::vector
);
```

### 16. `cluster_kmeans(table, column, k, max_iters)`
K-Means clustering.

```sql
SELECT * FROM neurondb.cluster_kmeans(
  'data_table', 'feature_vector', 5, 100
);
```

## Configuration & Management

### 17. `neurondb.version()`
Get NeuronDB version.

```sql
SELECT neurondb.version();
-- Returns: jsonb with version, postgresql_version, capabilities, and api fields
```

### 18. `set_llm_config(provider, api_key, endpoint)`
Configure LLM provider for embeddings.

```sql
SELECT neurondb.set_llm_config(
  'openai',
  'sk-...',
  'https://api.openai.com/v1'
);
```

### 19. `get_llm_config()`
Get current LLM configuration.

```sql
SELECT * FROM neurondb.get_llm_config();
```

### 20. `neurondb_gpu_info()`
Check GPU availability and information.

```sql
SELECT * FROM neurondb_gpu_info();
-- Returns: device_id, name, memory_total, compute_capability
```

## Function Categories Summary

| Category | Functions | Use Case |
|----------|-----------|----------|
| **Embedding** | `embed_text`, `embed_text_batch` | Generate vector embeddings |
| **Indexing** | `CREATE INDEX ... USING hnsw/ivf` | Build search indexes |
| **Search** | Distance operators, `mmr_rerank` | Vector similarity search |
| **Hybrid** | Hybrid queries, `rrf_rerank` | Combine vector + text search |
| **RAG** | `chunk_text`, `retrieve_context`, `rag_evaluate` | RAG pipeline operations |
| **ML** | `train`, `predict`, `cluster_kmeans` | Machine learning in-database |
| **Config** | `set_llm_config`, `neurondb_gpu_info` | Configuration and monitoring |

## Next Steps

- **Complete API Reference**: [SQL API Reference](sql-api-complete.md) - All 520+ functions
- **Examples**: [Examples Directory](../../examples/README.md) - Working code examples
- **Vector Search Guide**: [Indexing Documentation](../../NeuronDB/docs/vector-search/indexing.md)
- **RAG Guide**: [RAG Pipeline Documentation](../../NeuronDB/docs/rag/overview.md)
- **ML Guide**: [ML Algorithms Documentation](../../NeuronDB/docs/ml-algorithms/clustering.md)

## Quick Reference Card

```sql
-- 1. Generate embeddings
SELECT embed_text('text', 'model')::vector;

-- 2. Create index
CREATE INDEX idx ON table USING hnsw (vector vector_cosine_ops);

-- 3. Search
SELECT * FROM table ORDER BY vector <=> query LIMIT 10;

-- 4. Train model
SELECT train('kmeans', 'table', 'features', NULL, '{"k": 5}');

-- 5. Make prediction
SELECT predict(model_id, features);
```

