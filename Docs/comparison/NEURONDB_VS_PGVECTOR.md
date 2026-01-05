# NeuronDB vs pgvector: Feature Comparison

A comprehensive comparison of NeuronDB and pgvector for PostgreSQL vector search.

## Quick Comparison

| Feature | NeuronDB | pgvector |
|---------|----------|----------|
| **Index Types** | HNSW, IVF, PQ, OPQ, Hybrid, Multi-vector | HNSW, IVFFlat |
| **GPU Acceleration** | CUDA, ROCm, Metal (3 backends) | CPU only |
| **Distance Functions** | L2, Cosine, Inner Product, Manhattan, Hamming, Jaccard | L2, Cosine, Inner Product |
| **ML Algorithms** | 52+ algorithms (classification, regression, clustering) | None |
| **SQL Functions** | 473+ functions | ~20 functions |
| **Hybrid Search** | Native vector + full-text integration | Manual combination required |
| **RAG Tooling** | Built-in RAG pipeline, RAGAS evaluation | None |
| **Embedding Generation** | Built-in with multiple providers | External tools required |
| **PostgreSQL Versions** | 16, 17, 18 | 11+ (wider compatibility) |
| **License** | Proprietary (free for personal/non-commercial) | MIT (open source) |

## Detailed Feature Matrix

### Index Types

| Index Type | NeuronDB | pgvector | Notes |
|------------|----------|----------|-------|
| **HNSW** | ✅ | ✅ | Both support HNSW with similar parameters |
| **IVF** | ✅ (IVF, IVF-PQ) | ✅ (IVFFlat) | pgvector's IVFFlat is simpler; NeuronDB supports PQ variants |
| **Product Quantization (PQ)** | ✅ | ❌ | NeuronDB supports PQ and OPQ for compression |
| **Hybrid Indexes** | ✅ | ❌ | NeuronDB supports multi-vector and hybrid indexes |
| **Index Tuning** | ✅ Auto-tuning | ⚠️ Manual | NeuronDB includes auto-tuning for index parameters |

### Distance Metrics

| Distance Metric | NeuronDB | pgvector | Operator |
|-----------------|----------|----------|----------|
| **L2 (Euclidean)** | ✅ | ✅ | `<->` |
| **Cosine** | ✅ | ✅ | `<=>` |
| **Inner Product** | ✅ | ✅ | `<#>` |
| **Manhattan (L1)** | ✅ | ❌ | `<~>` |
| **Hamming** | ✅ | ❌ | `<@>` |
| **Jaccard** | ✅ | ❌ | `<%>` |

### GPU Acceleration

| Feature | NeuronDB | pgvector |
|---------|----------|----------|
| **CUDA Support** | ✅ | ❌ |
| **ROCm Support** | ✅ | ❌ |
| **Metal Support** | ✅ (macOS) | ❌ |
| **GPU Batch Operations** | ✅ | ❌ |
| **GPU Memory Management** | ✅ | ❌ |
| **CPU Fallback** | ✅ Automatic | N/A |

**Performance:** NeuronDB GPU acceleration provides 2-3x QPS improvement for vector search. See [Benchmark Results](../benchmarks/BENCHMARK_RESULTS.md) for detailed numbers.

### Machine Learning

| Feature | NeuronDB | pgvector |
|---------|----------|----------|
| **Classification** | ✅ (12+ algorithms) | ❌ |
| **Regression** | ✅ (8+ algorithms) | ❌ |
| **Clustering** | ✅ (K-Means, DBSCAN, GMM, Hierarchical) | ❌ |
| **Dimensionality Reduction** | ✅ (PCA, Whitening) | ❌ |
| **Model Training** | ✅ In-database | ❌ |
| **Model Inference** | ✅ In-database | ❌ |

### Embedding Generation

| Feature | NeuronDB | pgvector |
|---------|----------|----------|
| **Text Embeddings** | ✅ Built-in | ❌ |
| **Image Embeddings** | ✅ (CLIP) | ❌ |
| **Multimodal Embeddings** | ✅ (ImageBind) | ❌ |
| **Provider Support** | ✅ (OpenAI, HuggingFace, Local) | ❌ |
| **Batch Embeddings** | ✅ | ❌ |
| **Caching** | ✅ | ❌ |

### Hybrid Search

| Feature | NeuronDB | pgvector |
|---------|----------|----------|
| **Vector + Full-Text** | ✅ Native integration | ⚠️ Manual (requires separate queries) |
| **Reranking** | ✅ (MMR, RRF, Cross-encoder) | ❌ |
| **Faceted Search** | ✅ | ❌ |
| **Multi-vector Search** | ✅ | ❌ |

### RAG Pipeline

| Feature | NeuronDB | pgvector |
|---------|----------|----------|
| **Document Processing** | ✅ Built-in | ❌ |
| **Chunking Strategies** | ✅ Multiple | ❌ |
| **RAG Evaluation** | ✅ (RAGAS integration) | ❌ |
| **Context Retrieval** | ✅ Optimized | ⚠️ Manual SQL |
| **LLM Integration** | ✅ Built-in | ❌ |

### SQL Functions

| Category | NeuronDB | pgvector |
|----------|----------|----------|
| **Vector Operations** | 50+ functions | ~10 functions |
| **Distance Functions** | 15+ functions | 3 operators |
| **Index Management** | 20+ functions | ~5 functions |
| **ML Functions** | 200+ functions | 0 |
| **Embedding Functions** | 10+ functions | 0 |
| **RAG Functions** | 30+ functions | 0 |
| **Total Functions** | 473+ | ~20 |

### Performance

| Metric | NeuronDB | pgvector | Notes |
|---------|----------|----------|-------|
| **HNSW QPS (CPU)** | 1,200-1,500 | 1,000-1,300 | Similar performance on CPU |
| **HNSW QPS (GPU)** | 3,500-4,200 | N/A | GPU acceleration available |
| **Index Build Time** | Similar | Similar | Both use similar HNSW implementation |
| **Memory Usage** | Similar | Similar | Comparable memory footprint |

See [Benchmark Results](../benchmarks/BENCHMARK_RESULTS.md) for detailed performance numbers.

## Compatibility

### PostgreSQL Versions

| Version | NeuronDB | pgvector |
|---------|---------|----------|
| PostgreSQL 11 | ❌ | ✅ |
| PostgreSQL 12 | ❌ | ✅ |
| PostgreSQL 13 | ❌ | ✅ |
| PostgreSQL 14 | ❌ | ✅ |
| PostgreSQL 15 | ❌ | ✅ |
| PostgreSQL 16 | ✅ | ✅ |
| PostgreSQL 17 | ✅ | ✅ |
| PostgreSQL 18 | ✅ | ✅ |

**Note:** NeuronDB focuses on modern PostgreSQL versions (16+) for better performance and features.

### Data Types

| Type | NeuronDB | pgvector | Compatibility |
|------|----------|----------|---------------|
| `vector(n)` | ✅ | ✅ | ✅ Compatible |
| `vectorp` | ✅ | ❌ | NeuronDB-specific |
| `vecmap` | ✅ | ❌ | NeuronDB-specific (sparse) |
| `vgraph` | ✅ | ❌ | NeuronDB-specific |

**Migration:** NeuronDB's `vector` type is compatible with pgvector's `vector` type. You can migrate data directly.

## Migration from pgvector

### Step 1: Install NeuronDB

```bash
# Build and install NeuronDB
cd NeuronDB
make
sudo make install
```

### Step 2: Create Extension

```sql
-- Drop pgvector extension (if needed)
DROP EXTENSION IF EXISTS vector;

-- Create NeuronDB extension
CREATE EXTENSION neurondb;
```

### Step 3: Verify Compatibility

```sql
-- Your existing tables should work without changes
SELECT * FROM your_vector_table LIMIT 1;

-- Verify vector type
SELECT pg_typeof(embedding) FROM your_vector_table LIMIT 1;
-- Should return: vector
```

### Step 4: Rebuild Indexes

```sql
-- Drop old pgvector indexes
DROP INDEX IF EXISTS your_table_embedding_idx;

-- Create NeuronDB HNSW index (same syntax)
CREATE INDEX your_table_embedding_idx 
ON your_vector_table 
USING hnsw (embedding vector_cosine_ops);
```

### Step 5: Update Queries (Optional)

```sql
-- pgvector syntax (still works)
SELECT * FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;

-- NeuronDB also supports additional operators
SELECT * FROM documents
ORDER BY embedding <~> '[0.1, 0.2, ...]'::vector  -- Manhattan distance
LIMIT 10;
```

## When to Choose NeuronDB

Choose NeuronDB if you need:

- ✅ **GPU acceleration** for faster vector search
- ✅ **Machine learning** algorithms in-database
- ✅ **Built-in embedding generation** (no external services)
- ✅ **RAG pipeline** with evaluation tooling
- ✅ **Hybrid search** with native integration
- ✅ **Advanced index types** (PQ, multi-vector)
- ✅ **Comprehensive SQL API** (473+ functions)
- ✅ **Production-grade features** (monitoring, auto-tuning)

## When to Choose pgvector

Choose pgvector if you need:

- ✅ **Open source license** (MIT)
- ✅ **Wider PostgreSQL compatibility** (PG 11+)
- ✅ **Simpler feature set** (just vector search)
- ✅ **Community-driven development**
- ✅ **Minimal dependencies**

## Feature Parity Summary

| Category | NeuronDB Advantage | pgvector Advantage |
|----------|-------------------|-------------------|
| **Core Vector Search** | GPU acceleration, more distance metrics | Simpler, open source |
| **Index Types** | More options (PQ, hybrid) | Sufficient for most use cases |
| **ML & Analytics** | 52+ algorithms | None |
| **Embeddings** | Built-in generation | External tools required |
| **RAG** | Complete pipeline | Manual implementation |
| **License** | Proprietary | MIT (open source) |
| **PostgreSQL Support** | Modern versions (16+) | Wider compatibility (11+) |

## Conclusion

**NeuronDB** is a comprehensive AI database extension with GPU acceleration, ML algorithms, and RAG tooling. It's ideal for production applications requiring high performance and advanced features.

**pgvector** is a lightweight, open-source vector extension perfect for simple vector search use cases with broader PostgreSQL version compatibility.

Both extensions use compatible `vector` types, making migration straightforward. Choose based on your specific requirements for features, performance, and licensing.

## Related Documentation

- [NeuronDB Installation Guide](../../NeuronDB/INSTALL.md)
- [Benchmark Results](../benchmarks/BENCHMARK_RESULTS.md)
- [Vector Search Guide](../../NeuronDB/docs/vector-search/indexing.md)
- [GPU Support](../../NeuronDB/docs/gpu/cuda-support.md)

