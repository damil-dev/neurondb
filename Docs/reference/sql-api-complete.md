# NeuronDB SQL API Complete Reference

**Complete reference for all 654+ SQL functions, types, operators, and aggregates in the NeuronDB extension.**

> **Version:** 1.0  
> **PostgreSQL Compatibility:** 16, 17, 18  
> **Last Updated:** 2025-01-01

## Table of Contents

- [Vector Types](#vector-types)
- [Vector Operations](#vector-operations)
- [Distance Metrics](#distance-metrics)
- [Quantization Functions](#quantization-functions)
- [Indexing Functions](#indexing-functions)
- [Embedding Generation](#embedding-generation)
- [Hybrid Search](#hybrid-search)
- [Reranking](#reranking)
- [Machine Learning](#machine-learning)
- [RAG Functions](#rag-functions)
- [LLM Functions](#llm-functions)
- [Utility Functions](#utility-functions)

---

## Vector Types

NeuronDB provides multiple vector types optimized for different use cases:

### `vector`

**Type:** `vector`  
**Storage:** Extended (varlena)  
**Description:** Main vector type using float32 (4 bytes per dimension)

**Structure:**
```c
typedef struct Vector {
    int32 vl_len_;      // varlena header
    int16 dim;          // number of dimensions
    int16 unused;       // padding
    float4 data[];      // flexible array member
} Vector;
```

**Maximum Dimensions:** 16,000

**Example:**
```sql
-- Create a vector
SELECT '[1.0, 2.0, 3.0]'::vector;

-- Create with dimension constraint
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    embedding vector(384)
);
```

### `halfvec`

**Type:** `halfvec`  
**Storage:** Extended (varlena)  
**Description:** Half-precision vector (FP16) for 2x compression

**Maximum Dimensions:** 4,000

**Example:**
```sql
-- Convert vector to halfvec
SELECT vector_to_halfvec('[1.0, 2.0, 3.0]'::vector);

-- Cast between types
SELECT '[1.0, 2.0, 3.0]'::vector::halfvec;
```

### `sparsevec`

**Type:** `sparsevec`  
**Storage:** Extended (varlena)  
**Description:** Sparse vector storing only non-zero values

**Limits:**
- Up to 1,000 non-zero entries
- Up to 1,000,000 dimensions

**Example:**
```sql
-- Convert to sparse vector
SELECT vector_to_sparsevec('[0, 0, 1.5, 0, 2.3]'::vector);
```

### `binaryvec`

**Type:** `binaryvec`  
**Storage:** Extended (varlena)  
**Description:** Binary vector for 32x compression using Hamming distance

**Example:**
```sql
-- Quantize to binary
SELECT vector_to_binary('[1.0, -1.0, 0.5, -0.5]'::vector);

-- Hamming distance
SELECT binaryvec_hamming_distance(v1, v2) FROM vectors;
```

### `vectorp`

**Type:** `vectorp`  
**Storage:** Extended (varlena)  
**Description:** Packed SIMD vector with optimized storage layout

**Features:**
- Dimension fingerprint for validation
- Endianness guard for portability
- Version tag for schema evolution

### `vecmap`

**Type:** `vecmap`  
**Storage:** Extended (varlena)  
**Description:** Sparse high-dimensional map storing only non-zero values

**Use Case:** Very high dimensions (>10K) with sparse data

### `vgraph`

**Type:** `vgraph`  
**Storage:** Extended (varlena)  
**Description:** Graph-based vector structure for neighbor relations and clustering

**Use Case:** Graph algorithms, community detection, PageRank

### `rtext`

**Type:** `rtext`  
**Storage:** Extended (varlena)  
**Description:** Retrievable text with token metadata for RAG pipelines

**Features:**
- Token offsets for highlighting
- Section IDs for structured documents
- Language tag for multilingual support

---

## Vector Operations

### Basic Operations

#### `vector_dims(vector) → integer`

Get the number of dimensions in a vector.

**Example:**
```sql
SELECT vector_dims('[1.0, 2.0, 3.0]'::vector);  -- Returns 3
```

#### `vector_norm(vector) → double precision`

Compute L2 norm (Euclidean length) of vector.

**Example:**
```sql
SELECT vector_norm('[3.0, 4.0]'::vector);  -- Returns 5.0
```

#### `vector_normalize(vector) → vector`

Normalize vector to unit length (L2 normalization).

**Example:**
```sql
SELECT vector_normalize('[3.0, 4.0]'::vector);  -- Returns [0.6, 0.8]
```

#### `vector_concat(vector, vector) → vector`

Concatenate two vectors.

**Example:**
```sql
SELECT vector_concat('[1, 2]'::vector, '[3, 4]'::vector);  -- Returns [1, 2, 3, 4]
```

### Arithmetic Operators

| Operator | Function | Description |
|----------|----------|-------------|
| `+` | `vector_add(vector, vector)` | Element-wise addition |
| `-` | `vector_sub(vector, vector)` | Element-wise subtraction |
| `*` | `vector_mul(vector, double precision)` | Scalar multiplication |
| `/` | `vector_div(vector, double precision)` | Scalar division |
| `-` (unary) | `vector_neg(vector)` | Negation |

**Example:**
```sql
SELECT '[1, 2]'::vector + '[3, 4]'::vector;  -- [4, 6]
SELECT '[2, 4]'::vector * 2.0;               -- [4, 8]
SELECT -'[1, 2]'::vector;                     -- [-1, -2]
```

### Advanced Operations

#### `vector_get(vector, integer) → real`

Get element at index (0-based).

**Example:**
```sql
SELECT vector_get('[1.0, 2.0, 3.0]'::vector, 1);  -- Returns 2.0
```

#### `vector_set(vector, integer, real) → vector`

Set element at index (returns new vector).

**Example:**
```sql
SELECT vector_set('[1.0, 2.0, 3.0]'::vector, 1, 5.0);  -- [1.0, 5.0, 3.0]
```

#### `vector_slice(vector, integer, integer) → vector`

Extract subvector from start to end index.

**Example:**
```sql
SELECT vector_slice('[1, 2, 3, 4, 5]'::vector, 1, 3);  -- [2, 3]
```

#### Element-wise Operations

| Function | Description |
|----------|-------------|
| `vector_abs(vector)` | Element-wise absolute value |
| `vector_square(vector)` | Element-wise square |
| `vector_sqrt(vector)` | Element-wise square root |
| `vector_pow(vector, double precision)` | Element-wise power |
| `vector_hadamard(vector, vector)` | Hadamard (element-wise) product |
| `vector_divide(vector, vector)` | Element-wise division |

**Example:**
```sql
SELECT vector_abs('[-1, -2, 3]'::vector);           -- [1, 2, 3]
SELECT vector_square('[2, 3]'::vector);             -- [4, 9]
SELECT vector_hadamard('[1, 2]'::vector, '[3, 4]'::vector);  -- [3, 8]
```

#### Statistical Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `vector_mean(vector)` | `double precision` | Mean of elements |
| `vector_variance(vector)` | `double precision` | Variance of elements |
| `vector_stddev(vector)` | `double precision` | Standard deviation |
| `vector_min(vector)` | `real` | Minimum element |
| `vector_max(vector)` | `real` | Maximum element |
| `vector_median(vector)` | `real` | Median element |
| `vector_percentile(vector, double precision)` | `real` | Percentile value |
| `vector_element_sum(vector)` | `double precision` | Sum of all elements |

**Example:**
```sql
SELECT vector_mean('[1, 2, 3, 4, 5]'::vector);      -- 3.0
SELECT vector_stddev('[1, 2, 3, 4, 5]'::vector);    -- ~1.58
SELECT vector_percentile('[1, 2, 3, 4, 5]'::vector, 0.95);  -- 4.8
```

### Batch Operations

#### `vector_l2_distance_batch(vector[], vector) → real[]`

Compute L2 distance between query vector and array of vectors.

**Example:**
```sql
SELECT vector_l2_distance_batch(
    ARRAY['[1, 2]'::vector, '[3, 4]'::vector],
    '[1, 1]'::vector
);  -- Returns array of distances
```

#### `vector_normalize_batch(vector[]) → vector[]`

Normalize array of vectors.

**Example:**
```sql
SELECT vector_normalize_batch(
    ARRAY['[3, 4]'::vector, '[5, 12]'::vector]
);
```

#### `vector_sum_batch(vector[]) → vector`

Sum array of vectors element-wise.

**Example:**
```sql
SELECT vector_sum_batch(
    ARRAY['[1, 2]'::vector, '[3, 4]'::vector]
);  -- Returns [4, 6]
```

#### `vector_avg_batch(vector[]) → vector`

Average array of vectors element-wise.

**Example:**
```sql
SELECT vector_avg_batch(
    ARRAY['[1, 2]'::vector, '[3, 4]'::vector]
);  -- Returns [2, 3]
```

### Aggregates

#### `vector_avg(vector) → vector`

Average of vectors (element-wise mean).

**Example:**
```sql
SELECT vector_avg(embedding) FROM documents;
```

#### `vector_sum(vector) → vector`

Sum of vectors (element-wise sum).

**Example:**
```sql
SELECT vector_sum(embedding) FROM documents;
```

---

## Distance Metrics

NeuronDB supports multiple distance metrics for vector similarity search:

### L2 (Euclidean) Distance

#### `vector_l2_distance(vector, vector) → real`

Compute L2 (Euclidean) distance between two vectors.

**Operator:** `<->`

**Example:**
```sql
SELECT vector_l2_distance('[1, 2]'::vector, '[4, 6]'::vector);  -- 5.0

-- Using operator
SELECT '[1, 2]'::vector <-> '[4, 6]'::vector;
```

### Inner Product

#### `vector_inner_product(vector, vector) → real`

Compute negative inner product (for ordering).

**Note:** Returns negative value for use with `<->` operator ordering.

**Operator:** `<#>`

**Example:**
```sql
SELECT vector_inner_product('[1, 2]'::vector, '[3, 4]'::vector);  -- -11.0
```

#### `vector_dot(vector, vector) → real`

Compute positive dot product.

**Example:**
```sql
SELECT vector_dot('[1, 2]'::vector, '[3, 4]'::vector);  -- 11.0
```

### Cosine Distance

#### `vector_cosine_distance(vector, vector) → real`

Compute cosine distance (1 - cosine similarity).

**Operator:** `<=>`

**Example:**
```sql
SELECT vector_cosine_distance('[1, 0]'::vector, '[0, 1]'::vector);  -- 1.0

-- Using operator
SELECT '[1, 0]'::vector <=> '[0, 1]'::vector;
```

#### `vector_cosine_similarity(vector, vector) → real`

Compute cosine similarity (returns similarity, not distance).

**Example:**
```sql
SELECT vector_cosine_similarity('[1, 0]'::vector, '[1, 0]'::vector);  -- 1.0
```

### L1 (Manhattan) Distance

#### `vector_l1_distance(vector, vector) → real`

Compute L1 (Manhattan) distance.

**Example:**
```sql
SELECT vector_l1_distance('[1, 2]'::vector, '[4, 6]'::vector);  -- 7.0
```

### Other Distance Metrics

| Function | Returns | Description |
|----------|---------|-------------|
| `vector_hamming_distance(vector, vector)` | `integer` | Hamming distance |
| `vector_chebyshev_distance(vector, vector)` | `double precision` | Chebyshev (L-infinity) distance |
| `vector_minkowski_distance(vector, vector, double precision)` | `double precision` | Minkowski distance with parameter p |
| `vector_squared_l2_distance(vector, vector)` | `double precision` | Squared Euclidean distance (faster, no sqrt) |
| `vector_jaccard_distance(vector, vector)` | `double precision` | Jaccard distance (1 - Jaccard similarity) |
| `vector_dice_distance(vector, vector)` | `double precision` | Dice distance (1 - Dice coefficient) |
| `vector_mahalanobis_distance(vector, vector, vector)` | `double precision` | Mahalanobis distance with diagonal covariance |

**Example:**
```sql
-- Minkowski distance (p=3)
SELECT vector_minkowski_distance('[1, 2]'::vector, '[4, 6]'::vector, 3.0);

-- Squared L2 (faster, no sqrt)
SELECT vector_squared_l2_distance('[1, 2]'::vector, '[4, 6]'::vector);  -- 25.0
```

---

## Quantization Functions

NeuronDB supports multiple quantization formats for vector compression:

### INT8 Quantization (8x compression)

#### `vector_to_int8(vector) → bytea`

Quantize vector to int8 format.

#### `int8_to_vector(bytea) → vector`

Dequantize int8 vector back to float32.

**Example:**
```sql
-- Quantize
SELECT vector_to_int8('[1.5, 2.3, 3.7]'::vector);

-- Dequantize
SELECT int8_to_vector(quantized_data);
```

### FP16 Quantization (2x compression)

#### `vector_to_fp16(vector) → bytea`

Quantize vector to FP16 (IEEE 754 half-precision).

#### `fp16_to_vector(bytea) → vector`

Dequantize FP16 vector back to float32.

**Example:**
```sql
SELECT vector_to_fp16('[1.5, 2.3, 3.7]'::vector);
SELECT fp16_to_vector(quantized_data);
```

### Binary Quantization (32x compression)

#### `vector_to_binary(vector) → bytea`

Convert vector to binary format.

#### `binary_quantize(vector) → bit`

Convert vector to PostgreSQL bit type.

**Example:**
```sql
SELECT vector_to_binary('[1.0, -1.0, 0.5, -0.5]'::vector);
SELECT binary_quantize('[1.0, -1.0, 0.5, -0.5]'::vector);
```

### UINT8 Quantization (4x compression)

#### `vector_to_uint8(vector) → bytea`

Quantize vector to uint8 (unsigned [0,255]).

#### `uint8_to_vector(bytea) → vector`

Dequantize uint8 vector.

**Example:**
```sql
SELECT vector_to_uint8('[1.5, 2.3, 3.7]'::vector);
```

### Ternary Quantization (16x compression)

#### `vector_to_ternary(vector) → bytea`

Quantize vector to ternary (values: -1, 0, +1).

#### `ternary_to_vector(bytea) → vector`

Dequantize ternary vector.

**Example:**
```sql
SELECT vector_to_ternary('[1.5, -0.3, 0.0, -2.1]'::vector);
```

### INT4 Quantization (8x compression)

#### `vector_to_int4(vector) → bytea`

Quantize vector to int4 (4 bits per dimension, signed [-8, 7]).

#### `int4_to_vector(bytea) → vector`

Dequantize int4 vector.

### Quantization Analysis

#### `quantize_analyze_int8(vector) → jsonb`

Analyze INT8 quantization accuracy (MSE, MAE, compression ratio).

**Returns:**
```json
{
  "mse": 0.001,
  "mae": 0.05,
  "compression_ratio": 8.0,
  "max_error": 0.1
}
```

**Example:**
```sql
SELECT quantize_analyze_int8('[1.5, 2.3, 3.7]'::vector);
```

Similar functions available for:
- `quantize_analyze_fp16(vector)`
- `quantize_analyze_binary(vector)`
- `quantize_analyze_uint8(vector)`
- `quantize_analyze_ternary(vector)`
- `quantize_analyze_int4(vector)`

#### `quantize_compare_distances(vector, vector, text) → jsonb`

Compare distance preservation before and after quantization.

**Example:**
```sql
SELECT quantize_compare_distances(
    '[1, 2, 3]'::vector,
    '[4, 5, 6]'::vector,
    'int8'
);
```

---

## Indexing Functions

NeuronDB supports multiple index types for efficient vector search:

### HNSW Index

**Access Method:** `hnsw`

**Operators:**
- `<->` (L2 distance)
- `<#>` (inner product)
- `<=>` (cosine distance)

**Example:**
```sql
-- Create HNSW index
CREATE INDEX ON documents USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 200);

-- Search using index
SELECT id, content, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY embedding <-> '[0.1, 0.2, 0.3]'::vector
LIMIT 10;
```

**Parameters:**
- `m`: Number of bi-directional links (default: 16)
- `ef_construction`: Size of candidate list during construction (default: 200)

### IVF Index

**Access Method:** `ivfflat`

**Example:**
```sql
-- Create IVF index
CREATE INDEX ON documents USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);

-- Search using index
SELECT id, content
FROM documents
ORDER BY embedding <-> '[0.1, 0.2, 0.3]'::vector
LIMIT 10;
```

**Parameters:**
- `lists`: Number of clusters (default: 100)

**Configuration:**
```sql
-- Set number of probes
SET neurondb.ivf_probes = 10;
```

---

## Embedding Generation

### Text Embeddings

#### `embed_text(text, text) → vector`

Generate text embedding with GPU acceleration support.

**Parameters:**
- `text`: Text to embed
- `model_name`: Optional model name (uses default if NULL)

**Example:**
```sql
SELECT embed_text('Hello world', 'sentence-transformers/all-MiniLM-L6-v2');
```

#### `embed_text_batch(text[], text) → vector[]`

Batch text embedding with GPU acceleration.

**Example:**
```sql
SELECT embed_text_batch(
    ARRAY['Hello', 'World', 'NeuronDB'],
    'sentence-transformers/all-MiniLM-L6-v2'
);
```

#### `embed_cached(text, text) → vector`

Cached text embedding (uses cache if available).

**Example:**
```sql
SELECT embed_cached('Hello world', 'all-MiniLM-L6-v2');
```

### Image Embeddings

#### `embed_image(bytea, text) → vector`

Generate image embedding.

**Parameters:**
- `image_data`: Image bytes
- `model_name`: Model name (default: 'clip')

**Example:**
```sql
SELECT embed_image(image_bytes, 'clip');
```

### Multimodal Embeddings

#### `embed_multimodal(text, bytea, text) → vector`

Generate multimodal embedding from text and image.

**Example:**
```sql
SELECT embed_multimodal('A cat', image_bytes, 'clip');
```

### Model Configuration

#### `configure_embedding_model(text, text) → boolean`

Configure embedding model settings.

**Parameters:**
- `model_name`: Model name
- `config_json`: JSON configuration string

**Example:**
```sql
SELECT configure_embedding_model(
    'all-MiniLM-L6-v2',
    '{"batch_size": 32, "max_length": 512}'
);
```

#### `get_embedding_model_config(text) → jsonb`

Retrieve stored configuration for an embedding model.

**Example:**
```sql
SELECT get_embedding_model_config('all-MiniLM-L6-v2');
```

#### `list_embedding_model_configs() → TABLE(...)`

List all stored embedding model configurations.

**Example:**
```sql
SELECT * FROM list_embedding_model_configs();
```

#### `delete_embedding_model_config(text) → boolean`

Delete stored configuration for an embedding model.

**Example:**
```sql
SELECT delete_embedding_model_config('all-MiniLM-L6-v2');
```

---

## Hybrid Search

### Basic Hybrid Search

#### `hybrid_search(text, vector, text, text, double precision, integer, text) → TABLE(id, score)`

Hybrid search combining vector and full-text search.

**Parameters:**
- `table`: Table name
- `query_vec`: Query vector
- `query_text`: Query text
- `filters`: JSON filters (default: '{}')
- `vector_weight`: Weight for vector component (default: 0.7)
- `limit`: Maximum results (default: 10)
- `query_type`: Query type - 'plain', 'to', or 'phrase' (default: 'plain')

**Example:**
```sql
SELECT * FROM hybrid_search(
    'documents',
    '[0.1, 0.2, 0.3]'::vector,
    'machine learning',
    '{}',
    0.7,
    10,
    'plain'
);
```

### Reciprocal Rank Fusion

#### `reciprocal_rank_fusion(anyarray, double precision) → anyarray`

Reciprocal Rank Fusion for combining multiple rankings.

**Parameters:**
- `rankings`: Array of rankings
- `k`: RRF parameter (default: 60.0)

**Example:**
```sql
SELECT reciprocal_rank_fusion(
    ARRAY[ARRAY[1, 2, 3], ARRAY[2, 1, 4]],
    60.0
);
```

### Semantic + Keyword Search

#### `semantic_keyword_search(text, vector, text, integer) → TABLE(id, score)`

Semantic + keyword search combination.

**Parameters:**
- `table`: Table name
- `semantic_query`: Semantic query vector
- `keyword_query`: Keyword query text
- `top_k`: Number of results (default: 10)

**Example:**
```sql
SELECT * FROM semantic_keyword_search(
    'documents',
    '[0.1, 0.2, 0.3]'::vector,
    'machine learning',
    10
);
```

### Multi-Vector Search

#### `multi_vector_search(text, vector[], text, integer) → TABLE(id, score)`

Search with multiple query vectors.

**Parameters:**
- `table`: Table name
- `query_vectors`: Array of query vectors
- `agg_method`: Aggregation method - 'max', 'avg', 'sum' (default: 'max')
- `top_k`: Number of results (default: 10)

**Example:**
```sql
SELECT * FROM multi_vector_search(
    'documents',
    ARRAY['[0.1, 0.2]'::vector, '[0.3, 0.4]'::vector],
    'max',
    10
);
```

### Faceted Search

#### `faceted_vector_search(text, vector, text, integer) → TABLE(facet, id, score)`

Faceted vector search with per-facet results.

**Parameters:**
- `table`: Table name
- `query_vec`: Query vector
- `facet_column`: Facet column name
- `per_facet_limit`: Results per facet (default: 3)

**Example:**
```sql
SELECT * FROM faceted_vector_search(
    'documents',
    '[0.1, 0.2, 0.3]'::vector,
    'category',
    3
);
```

### Temporal Search

#### `temporal_vector_search(text, vector, text, double precision, integer) → TABLE(id, score)`

Temporal vector search with time decay.

**Parameters:**
- `table`: Table name
- `query_vec`: Query vector
- `timestamp_col`: Timestamp column name
- `decay_rate`: Decay rate per day (default: 0.01)
- `top_k`: Number of results (default: 10)

**Example:**
```sql
SELECT * FROM temporal_vector_search(
    'documents',
    '[0.1, 0.2, 0.3]'::vector,
    'created_at',
    0.01,
    10
);
```

### Diverse Search

#### `diverse_vector_search(text, vector, double precision, integer) → TABLE(id, score)`

Diverse search using Maximal Marginal Relevance (MMR).

**Parameters:**
- `table`: Table name
- `query_vec`: Query vector
- `lambda`: Diversity parameter (default: 0.5)
- `top_k`: Number of results (default: 10)

**Example:**
```sql
SELECT * FROM diverse_vector_search(
    'documents',
    '[0.1, 0.2, 0.3]'::vector,
    0.5,
    10
);
```

### Full-Text Search

#### `full_text_search(text, text, text, text, text, integer) → TABLE(id, score)`

Full-text search (text-only, no vectors).

**Parameters:**
- `table`: Table name
- `query_text`: Query text
- `text_column`: Text column name (default: 'fts_vector')
- `query_type`: Query type - 'plain', 'to', or 'phrase' (default: 'plain')
- `filters`: JSON filters (default: '{}')
- `limit`: Maximum results (default: 10)

**Example:**
```sql
SELECT * FROM full_text_search(
    'documents',
    'machine learning',
    'content',
    'plain',
    '{}',
    10
);
```

---

## Reranking

### Cross-Encoder Reranking

#### `rerank_cross_encoder(text, text[], text, integer) → TABLE(idx, score)`

Cross-encoder reranking with GPU acceleration support.

**Parameters:**
- `query`: Query text
- `candidates`: Array of candidate texts
- `model`: Model name (default: 'ms-marco-MiniLM-L-6-v2')
- `top_k`: Number of top results (default: 10)

**Example:**
```sql
SELECT * FROM rerank_cross_encoder(
    'What is machine learning?',
    ARRAY['ML is...', 'AI is...', 'Deep learning...'],
    'ms-marco-MiniLM-L-6-v2',
    10
);
```

### LLM Reranking

#### `rerank_llm(text, text[], text, integer) → TABLE(idx, score)`

LLM-based reranking.

**Parameters:**
- `query`: Query text
- `candidates`: Array of candidate texts
- `model`: Model name (default: 'gpt-3.5-turbo')
- `top_k`: Number of top results (default: 10)

**Example:**
```sql
SELECT * FROM rerank_llm(
    'What is machine learning?',
    ARRAY['ML is...', 'AI is...'],
    'gpt-3.5-turbo',
    10
);
```

### Cohere Reranking

#### `rerank_cohere(text, text[], integer) → TABLE(idx, score)`

Cohere-style reranking.

**Example:**
```sql
SELECT * FROM rerank_cohere(
    'What is machine learning?',
    ARRAY['ML is...', 'AI is...'],
    10
);
```

### ColBERT Reranking

#### `rerank_colbert(text, text[], text) → TABLE(idx, score)`

ColBERT late interaction reranking.

**Parameters:**
- `query`: Query text
- `docs`: Array of documents
- `model`: Model name (default: 'colbert-v2')

**Example:**
```sql
SELECT * FROM rerank_colbert(
    'What is machine learning?',
    ARRAY['ML is...', 'AI is...'],
    'colbert-v2'
);
```

### Learning-to-Rank

#### `rerank_ltr(text, text[], text, text) → TABLE(idx, score)`

Learning-to-Rank reranking.

**Parameters:**
- `query`: Query text
- `docs`: Array of documents
- `features_json`: JSON features
- `model`: Model name

**Example:**
```sql
SELECT * FROM rerank_ltr(
    'What is machine learning?',
    ARRAY['ML is...', 'AI is...'],
    '{"feature1": 0.5, "feature2": 0.3}',
    'ltr_model'
);
```

### Ensemble Reranking

#### `rerank_ensemble(text, text[], text[], double precision[]) → TABLE(idx, score)`

Ensemble reranking combining multiple models.

**Parameters:**
- `query`: Query text
- `docs`: Array of documents
- `models`: Array of model names
- `weights`: Array of weights

**Example:**
```sql
SELECT * FROM rerank_ensemble(
    'What is machine learning?',
    ARRAY['ML is...', 'AI is...'],
    ARRAY['model1', 'model2'],
    ARRAY[0.6, 0.4]
);
```

---

## Machine Learning

### Regression

#### Linear Regression

##### `train_linear_regression(text, text, text) → integer`

Train linear regression and return model_id.

**Parameters:**
- `table_name`: Training table
- `feature_col`: Feature column name
- `label_col`: Label column name

**Example:**
```sql
SELECT train_linear_regression('housing', 'features', 'price');
```

##### `predict_linear_regression_model_id(integer, vector) → float8`

Predict using model_id from catalog.

**Example:**
```sql
SELECT predict_linear_regression_model_id(1, '[1.0, 2.0, 3.0]'::vector);
```

##### `evaluate_linear_regression_by_model_id(integer, text, text, text) → jsonb`

Evaluate linear regression model.

**Returns:**
```json
{
  "mse": 0.5,
  "rmse": 0.707,
  "mae": 0.4,
  "r2": 0.95
}
```

#### Ridge Regression

##### `train_ridge_regression(text, text, text, float8) → integer`

Train Ridge Regression with L2 regularization.

**Parameters:**
- `table_name`: Training table
- `feature_col`: Feature column name
- `label_col`: Label column name
- `alpha`: Regularization parameter

**Example:**
```sql
SELECT train_ridge_regression('housing', 'features', 'price', 0.1);
```

##### `predict_ridge_regression_model_id(integer, vector) → float8`

Predict using Ridge Regression model.

#### Lasso Regression

##### `train_lasso_regression(text, text, text, float8, integer) → integer`

Train Lasso Regression with L1 regularization.

**Parameters:**
- `table_name`: Training table
- `feature_col`: Feature column name
- `label_col`: Label column name
- `alpha`: Regularization parameter
- `max_iterations`: Maximum iterations (default: 1000)

**Example:**
```sql
SELECT train_lasso_regression('housing', 'features', 'price', 0.1, 1000);
```

##### `predict_lasso_regression_model_id(integer, vector) → float8`

Predict using Lasso Regression model.

### Classification

#### Logistic Regression

##### `train_logistic_regression(text, text, text, integer, float8, float8) → integer`

Train logistic regression for binary classification.

**Parameters:**
- `table_name`: Training table
- `feature_col`: Feature column name
- `label_col`: Label column name
- `max_iterations`: Maximum iterations (default: 1000)
- `learning_rate`: Learning rate (default: 0.01)
- `regularization`: Regularization parameter (default: 0.001)

**Example:**
```sql
SELECT train_logistic_regression('iris', 'features', 'species', 1000, 0.01, 0.001);
```

##### `predict_logistic_regression(integer, vector) → float8`

Predict probability using logistic regression model.

**Example:**
```sql
SELECT predict_logistic_regression(1, '[1.0, 2.0, 3.0]'::vector);
```

##### `evaluate_logistic_regression_by_model_id(integer, text, text, text, float8) → jsonb`

Evaluate logistic regression model.

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

#### Random Forest

##### `train_random_forest_classifier(text, text, text, integer, integer, integer, integer) → integer`

Train Random Forest classifier.

**Parameters:**
- `table_name`: Training table
- `feature_col`: Feature column name
- `label_col`: Label column name
- `n_trees`: Number of trees (default: 100)
- `max_depth`: Maximum tree depth (default: 10)
- `min_samples_split`: Minimum samples to split (default: 2)
- `max_features`: Maximum features per split (default: 0 = sqrt)

**Example:**
```sql
SELECT train_random_forest_classifier('iris', 'features', 'species', 100, 10, 2, 0);
```

##### `predict_random_forest(integer, vector) → double precision`

Predict using Random Forest model.

**Example:**
```sql
SELECT predict_random_forest(1, '[1.0, 2.0, 3.0]'::vector);
```

##### `evaluate_random_forest_by_model_id(integer, text, text, text) → jsonb`

Evaluate Random Forest model.

#### Decision Tree

##### `train_decision_tree_classifier(text, text, text, integer, integer) → integer`

Train Decision Tree (CART).

**Parameters:**
- `table_name`: Training table
- `feature_col`: Feature column name
- `label_col`: Label column name
- `max_depth`: Maximum tree depth (default: 10)
- `min_samples_split`: Minimum samples to split (default: 2)

**Example:**
```sql
SELECT train_decision_tree_classifier('iris', 'features', 'species', 10, 2);
```

##### `predict_decision_tree_model_id(integer, vector) → float8`

Predict using Decision Tree model.

#### SVM

##### `train_svm_classifier(text, text, text, float8, integer) → integer`

Train Support Vector Machine classifier.

**Parameters:**
- `table_name`: Training table
- `feature_col`: Feature column name
- `label_col`: Label column name
- `C`: Regularization parameter (default: 1.0)
- `max_iters`: Maximum iterations (default: 1000)

**Example:**
```sql
SELECT train_svm_classifier('iris', 'features', 'species', 1.0, 1000);
```

##### `predict_svm_model_id(integer, vector) → float8`

Predict using SVM model.

#### Naive Bayes

##### `train_naive_bayes_classifier_model_id(text, text, text) → integer`

Train Naive Bayes classifier.

**Example:**
```sql
SELECT train_naive_bayes_classifier_model_id('iris', 'features', 'species');
```

##### `predict_naive_bayes_model_id(integer, vector) → integer`

Predict using Naive Bayes model.

### Instance-Based Learning

#### K-Nearest Neighbors

##### `train_knn_model_id(text, text, text, integer) → integer`

Train KNN (lazy learner) and store metadata in catalog.

**Parameters:**
- `table_name`: Training table
- `feature_col`: Feature column name
- `label_col`: Label column name
- `k`: Number of neighbors

**Example:**
```sql
SELECT train_knn_model_id('iris', 'features', 'species', 5);
```

##### `predict_knn_model_id(integer, real[]) → double precision`

Predict with KNN model.

**Example:**
```sql
SELECT predict_knn_model_id(1, ARRAY[1.0, 2.0, 3.0]::real[]);
```

##### `knn_classify(text, text, text, vector, integer) → integer`

KNN classification (direct, no model storage).

**Example:**
```sql
SELECT knn_classify('iris', 'features', 'species', '[1.0, 2.0, 3.0]'::vector, 5);
```

##### `knn_regress(text, text, text, vector, integer) → float8`

KNN regression.

**Example:**
```sql
SELECT knn_regress('housing', 'features', 'price', '[1.0, 2.0, 3.0]'::vector, 5);
```

### Gradient Boosting

#### XGBoost

##### `train_xgboost_classifier(text, text, text, integer, integer, float8) → integer`

Train XGBoost classifier.

**Parameters:**
- `table_name`: Training table
- `feature_col`: Feature column name
- `label_col`: Label column name
- `n_estimators`: Number of trees (default: 100)
- `max_depth`: Maximum tree depth (default: 6)
- `learning_rate`: Learning rate (default: 0.3)

**Example:**
```sql
SELECT train_xgboost_classifier('iris', 'features', 'species', 100, 6, 0.3);
```

##### `train_xgboost_regressor(text, text, text, integer, integer, float8) → integer`

Train XGBoost regressor.

**Example:**
```sql
SELECT train_xgboost_regressor('housing', 'features', 'price', 100, 6, 0.3);
```

##### `predict_xgboost(integer, real[]) → float8`

Predict using XGBoost model.

##### `evaluate_xgboost_by_model_id(integer, text, text, text) → jsonb`

Evaluate XGBoost model.

#### CatBoost

##### `train_catboost_classifier(text, text, text, integer, float8, integer) → integer`

Train CatBoost classifier.

**Parameters:**
- `table_name`: Training table
- `feature_col`: Feature column name
- `label_col`: Label column name
- `iterations`: Number of iterations (default: 1000)
- `learning_rate`: Learning rate (default: 0.03)
- `depth`: Tree depth (default: 6)

**Example:**
```sql
SELECT train_catboost_classifier('iris', 'features', 'species', 1000, 0.03, 6);
```

##### `train_catboost_regressor(text, text, text, integer, float8, integer) → integer`

Train CatBoost regressor.

##### `predict_catboost(integer, real[]) → float8`

Predict using CatBoost model.

#### LightGBM

##### `train_lightgbm_classifier(text, text, text, integer, integer, float8) → integer`

Train LightGBM classifier.

**Parameters:**
- `table_name`: Training table
- `feature_col`: Feature column name
- `label_col`: Label column name
- `n_estimators`: Number of trees (default: 100)
- `num_leaves`: Number of leaves (default: 31)
- `learning_rate`: Learning rate (default: 0.1)

**Example:**
```sql
SELECT train_lightgbm_classifier('iris', 'features', 'species', 100, 31, 0.1);
```

##### `train_lightgbm_regressor(text, text, text, integer) → integer`

Train LightGBM regressor.

##### `predict_lightgbm(integer, real[]) → float8`

Predict using LightGBM model.

### Clustering

#### K-Means

##### `cluster_kmeans(text, text, integer, integer) → integer[]`

K-means clustering.

**Parameters:**
- `table_name`: Table name
- `vector_col`: Vector column name
- `num_clusters`: Number of clusters
- `max_iters`: Maximum iterations (default: 100)

**Returns:** Array of cluster assignments (1-based)

**Example:**
```sql
SELECT cluster_kmeans('points', 'embedding', 5, 100);
```

### Dimensionality Reduction

#### PCA

##### `reduce_pca(text, text, integer) → real[][]`

PCA dimensionality reduction.

**Parameters:**
- `table_name`: Table name
- `column`: Vector column name
- `n_components`: Number of components

**Returns:** Array of reduced vectors

**Example:**
```sql
SELECT reduce_pca('points', 'embedding', 2);
```

### Time Series

#### ARIMA

##### `train_timeseries_cpu(text, text, text, integer, integer, integer) → integer`

Train time series model (ARIMA).

**Parameters:**
- `table_name`: Table name
- `feature_col`: Feature column name
- `label_col`: Label column name
- `p`: AR order
- `d`: Differencing order
- `q`: MA order

**Example:**
```sql
SELECT train_timeseries_cpu('sales', 'features', 'value', 1, 1, 1);
```

### Unified Prediction

#### `neurondb_predict(integer, real[]) → float8`

Unified prediction function that routes to algorithm-specific prediction functions.

**Example:**
```sql
SELECT neurondb_predict(1, ARRAY[1.0, 2.0, 3.0]::real[]);
```

---

## RAG Functions

### Text Chunking

#### `neurondb_chunk_text(text, integer, integer, text) → text[]`

Chunk text for RAG with configurable size, overlap, and separator.

**Parameters:**
- `text`: Text to chunk
- `chunk_size`: Chunk size in characters (default: 512)
- `overlap`: Overlap between chunks (default: 128)
- `separator`: Separator text (default: NULL)

**Example:**
```sql
SELECT neurondb_chunk_text(
    'Long document text...',
    512,
    128,
    NULL
);
```

### Document Ranking

#### `neurondb_rank_documents(text, text[], text) → jsonb`

Rank documents based on query using various algorithms.

**Parameters:**
- `query`: Query text
- `documents`: Array of document texts
- `algorithm`: Ranking algorithm - 'bm25', 'cosine', or 'edit_distance' (default: 'bm25')

**Returns:**
```json
{
  "rankings": [
    {"idx": 0, "score": 0.95, "document": "..."},
    {"idx": 1, "score": 0.87, "document": "..."}
  ]
}
```

**Example:**
```sql
SELECT neurondb_rank_documents(
    'machine learning',
    ARRAY['ML is...', 'AI is...', 'Deep learning...'],
    'bm25'
);
```

### Data Transformation

#### `neurondb_transform_data(text, double precision[]) → double precision[]`

Apply transformation to float8 array.

**Parameters:**
- `pipeline`: Transformation type - 'normalize', 'standardize', or 'min_max'
- `input_data`: Input array

**Example:**
```sql
SELECT neurondb_transform_data(
    'normalize',
    ARRAY[1.0, 2.0, 3.0, 4.0, 5.0]::double precision[]
);
```

---

## LLM Functions

### Text Generation

#### `neurondb_llm_generate(text, text, jsonb) → text`

LLM text generation wrapper for NeuronAgent compatibility.

**Parameters:**
- `model`: Model name
- `prompt`: Prompt text
- `params`: JSONB parameters (temperature, max_tokens, etc.)

**Example:**
```sql
SELECT neurondb_llm_generate(
    'gpt-3.5-turbo',
    'What is machine learning?',
    '{"temperature": 0.7, "max_tokens": 100}'::jsonb
);
```

#### `neurondb_llm_complete(text, text, jsonb) → text`

Alias for `neurondb_llm_generate()`.

#### `neurondb_llm_generate_stream(text, text, jsonb) → SETOF text`

Streaming LLM generation (returns text in chunks).

**Example:**
```sql
SELECT * FROM neurondb_llm_generate_stream(
    'gpt-3.5-turbo',
    'What is machine learning?',
    '{}'::jsonb
);
```

### Core LLM Functions

#### `ndb_llm_complete(text, text) → text`

LLM completion with GPU acceleration support.

**Parameters:**
- `prompt`: Prompt text
- `params`: JSON text with optional model, max_tokens, temperature, etc.

**Example:**
```sql
SELECT ndb_llm_complete(
    'What is machine learning?',
    '{"model": "gpt-3.5-turbo", "temperature": 0.7}'
);
```

#### `ndb_llm_image_analyze(bytea, text, text, text) → text`

LLM image analysis (vision) using GPT-4 Vision or other vision-capable models.

**Parameters:**
- `image_data`: Image bytes
- `prompt`: Optional text query about the image
- `params`: JSON with temperature, max_tokens, etc.
- `model`: Optional model name (defaults to gpt-4o)

**Example:**
```sql
SELECT ndb_llm_image_analyze(
    image_bytes,
    'What is in this image?',
    '{"temperature": 0.7}',
    'gpt-4o'
);
```

#### `ndb_llm_embed(text, text) → vector`

LLM embedding with GPU acceleration support.

**Example:**
```sql
SELECT ndb_llm_embed('Hello world', 'text-embedding-ada-002');
```

#### `ndb_llm_rerank(text, text[], text, integer) → TABLE(idx, score)`

LLM-based reranking with GPU acceleration support.

**Example:**
```sql
SELECT * FROM ndb_llm_rerank(
    'What is machine learning?',
    ARRAY['ML is...', 'AI is...'],
    'ms-marco-MiniLM-L-6-v2',
    10
);
```

### Batch Operations

#### `ndb_llm_complete_batch(text[], text) → TABLE(idx, text, tokens_in, tokens_out, http_status)`

Batch LLM completion with GPU acceleration support.

**Example:**
```sql
SELECT * FROM ndb_llm_complete_batch(
    ARRAY['Prompt 1', 'Prompt 2', 'Prompt 3'],
    '{"temperature": 0.7}'
);
```

#### `ndb_llm_rerank_batch(text[], text[][], text, integer) → TABLE(query_idx, doc_idx, score)`

Batch LLM reranking with GPU acceleration support.

**Example:**
```sql
SELECT * FROM ndb_llm_rerank_batch(
    ARRAY['Query 1', 'Query 2'],
    ARRAY[
        ARRAY['Doc 1', 'Doc 2'],
        ARRAY['Doc 3', 'Doc 4']
    ],
    'ms-marco-MiniLM-L-6-v2',
    10
);
```

### GPU Information

#### `neurondb_llm_gpu_available() → boolean`

Check if GPU is available for LLM operations.

**Example:**
```sql
SELECT neurondb_llm_gpu_available();
```

#### `neurondb_llm_gpu_info() → TABLE(backend, device_id, device_name, total_memory_mb, free_memory_mb, is_available)`

Get GPU information for LLM operations.

**Example:**
```sql
SELECT * FROM neurondb_llm_gpu_info();
```

#### `neurondb_llm_gpu_utilization() → TABLE(device_id, utilization_pct, memory_used_mb, memory_total_mb, memory_utilization_pct, temperature_c, power_w, timestamp)`

Get GPU utilization metrics.

**Example:**
```sql
SELECT * FROM neurondb_llm_gpu_utilization();
```

### Job Queue

#### `ndb_llm_enqueue(text, text, text, text) → bigint`

Enqueue asynchronous LLM job for background processing.

**Parameters:**
- `operation`: Operation type
- `model`: Model name
- `input_text`: Input text
- `tenant_id`: Tenant ID

**Example:**
```sql
SELECT ndb_llm_enqueue(
    'complete',
    'gpt-3.5-turbo',
    'What is machine learning?',
    'tenant1'
);
```

---

## Utility Functions

### Type Conversions

#### `array_to_vector(real[]) → vector`

Convert float array to vector.

**Example:**
```sql
SELECT array_to_vector(ARRAY[1.0, 2.0, 3.0]::real[]);
```

#### `array_to_vector(double precision[]) → vector`

Convert double precision array to vector.

**Example:**
```sql
SELECT array_to_vector(ARRAY[1.0, 2.0, 3.0]::double precision[]);
```

#### `vector_to_array(vector) → real[]`

Convert vector to float array.

**Example:**
```sql
SELECT vector_to_array('[1.0, 2.0, 3.0]'::vector);
```

#### `vector_cast_dimension(vector, integer) → vector`

Change vector dimension (truncate or pad with zeros).

**Example:**
```sql
SELECT vector_cast_dimension('[1, 2, 3]'::vector, 5);  -- [1, 2, 3, 0, 0]
SELECT vector_cast_dimension('[1, 2, 3, 4, 5]'::vector, 3);  -- [1, 2, 3]
```

### ONNX Runtime

#### `neurondb_onnx_info() → text`

Return ONNX runtime availability and version metadata.

**Example:**
```sql
SELECT neurondb_onnx_info();
```

### Graph Operations

#### `vgraph_bfs(vgraph, integer, integer) → TABLE(node_idx, depth, parent_idx)`

Breadth-First Search: traverse graph from starting node.

**Parameters:**
- `graph`: Graph structure
- `start_node`: Starting node index
- `max_depth`: Maximum depth (default: -1 = unlimited)

**Example:**
```sql
SELECT * FROM vgraph_bfs(graph_data, 0, -1);
```

#### `vgraph_dfs(vgraph, integer) → TABLE(node_idx, discovery_time, finish_time, parent_idx)`

Depth-First Search: traverse graph from starting node.

**Example:**
```sql
SELECT * FROM vgraph_dfs(graph_data, 0);
```

#### `vgraph_pagerank(vgraph, double precision, integer, double precision) → TABLE(node_idx, pagerank_score)`

PageRank algorithm: compute importance scores for all nodes.

**Parameters:**
- `graph`: Graph structure
- `damping_factor`: Damping factor (default: 0.85)
- `max_iterations`: Maximum iterations (default: 100)
- `tolerance`: Convergence tolerance (default: 1e-6)

**Example:**
```sql
SELECT * FROM vgraph_pagerank(graph_data, 0.85, 100, 1e-6);
```

#### `vgraph_community_detection(vgraph, integer) → TABLE(node_idx, community_id, modularity)`

Community detection: detect communities in graph using simplified Louvain algorithm.

**Parameters:**
- `graph`: Graph structure
- `max_iterations`: Maximum iterations (default: 10)

**Example:**
```sql
SELECT * FROM vgraph_community_detection(graph_data, 10);
```

### Sparse Vector Operations

#### `vecmap_l2_distance(bytea, bytea) → real`

L2 (Euclidean) distance for sparse vectors.

**Example:**
```sql
SELECT vecmap_l2_distance(sparse_vec1, sparse_vec2);
```

#### `vecmap_cosine_distance(bytea, bytea) → real`

Cosine distance for sparse vectors.

#### `vecmap_inner_product(bytea, bytea) → real`

Inner product for sparse vectors.

#### `vecmap_l1_distance(bytea, bytea) → real`

L1 (Manhattan) distance for sparse vectors.

#### `vecmap_add(bytea, bytea) → bytea`

Add two sparse vectors.

#### `vecmap_sub(bytea, bytea) → bytea`

Subtract two sparse vectors.

#### `vecmap_mul_scalar(bytea, real) → bytea`

Multiply sparse vector by scalar.

#### `vecmap_norm(bytea) → real`

Compute L2 norm of sparse vector.

---

## Function Stability

Functions are classified by stability:

- **IMMUTABLE**: Function always returns the same result for the same inputs
- **STABLE**: Function returns the same result for the same inputs within a single query
- **VOLATILE**: Function may return different results on each call

Most vector operations are **IMMUTABLE** or **STABLE**. ML training functions are **STABLE**. LLM and embedding functions are **STABLE** (may use caching).

---

## Performance Considerations

### Index Usage

Always create appropriate indexes for vector columns:

```sql
-- HNSW for high-dimensional vectors
CREATE INDEX ON documents USING hnsw (embedding vector_l2_ops);

-- IVF for very large datasets
CREATE INDEX ON documents USING ivfflat (embedding vector_l2_ops);
```

### Batch Operations

Use batch functions when processing multiple vectors:

```sql
-- Better: batch operation
SELECT embed_text_batch(ARRAY['text1', 'text2', 'text3']);

-- Instead of: individual calls
SELECT embed_text('text1'), embed_text('text2'), embed_text('text3');
```

### GPU Acceleration

Enable GPU acceleration for better performance:

```sql
SET neurondb.compute_mode = 2;  -- auto (try GPU first)
SET neurondb.gpu_device = 0;
```

---

## Related Documentation

- [Data Types Reference](data-types.md)
- [Configuration Reference](configuration-complete.md)
- [Index Methods](../internals/index-methods.md)
- [GPU Acceleration](../advanced/gpu-acceleration-complete.md)
- [ML Algorithms](../advanced/ml-algorithms-complete.md)

---

**Last Updated:** 2025-01-01  
**Documentation Version:** 1.0.0


