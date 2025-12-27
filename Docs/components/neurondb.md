# NeuronDB Component

PostgreSQL extension that adds vector search, machine learning algorithms, and embedding generation capabilities directly to PostgreSQL.

## Overview

NeuronDB extends PostgreSQL with native vector types, approximate nearest neighbor (ANN) search, GPU acceleration, ML analytics, hybrid semantic+lexical search, background workers, observability, and auto-tuning.

## Key Capabilities

| Feature | Description |
|---------|-------------|
| **Vector Search** | HNSW and IVF indexing for similarity search |
| **Machine Learning** | 52+ algorithms (classification, regression, clustering) |
| **GPU Acceleration** | CUDA, ROCm, and Metal support |
| **Hybrid Search** | Vector and full-text search combination |
| **RAG Pipeline** | Document retrieval and context generation |
| **Embeddings** | Text, image, and multimodal embedding generation |
| **Background Workers** | Async job processing and index maintenance |

## Documentation

### Local Documentation

- **[Component README](../../NeuronDB/README.md)** - Complete feature documentation
- **[Installation Guide](../../NeuronDB/INSTALL.md)** - Build and installation instructions
- **[Docker Guide](../../NeuronDB/docker/README.md)** - Container deployment
- **[SQL API Reference](../../NeuronDB/docs/sql-api.md)** - Function reference
- **[Vector Search Guide](../../NeuronDB/docs/vector-search/)** - Indexing and search
- **[ML Algorithms Guide](../../NeuronDB/docs/ml-algorithms/)** - Machine learning features
- **[RAG Pipeline Guide](../../NeuronDB/docs/rag/)** - Retrieval-augmented generation

### Official Documentation

- **[NeuronDB Guide](https://www.neurondb.ai/docs/neurondb)** - Complete PostgreSQL extension guide
- **[Vector Search Guide](https://www.neurondb.ai/docs/vector-search)** - HNSW indexing, distance metrics, quantization
- **[ML Algorithms Guide](https://www.neurondb.ai/docs/ml-algorithms)** - 52 ML algorithms with examples
- **[RAG Pipeline Guide](https://www.neurondb.ai/docs/rag)** - Complete RAG implementation
- **[GPU Acceleration Guide](https://www.neurondb.ai/docs/gpu)** - CUDA, ROCm, Metal support
- **[Hybrid Search Guide](https://www.neurondb.ai/docs/hybrid-search)** - Vector + full-text search
- **[Performance Guide](https://www.neurondb.ai/docs/performance)** - Optimization strategies
- **[Security Guide](https://www.neurondb.ai/docs/security)** - Encryption, privacy, RLS

## Installation

### Docker (Recommended)

```bash
cd NeuronDB/docker
docker compose up -d neurondb
```

### Source Build

```bash
cd NeuronDB
make install PG_CONFIG=/usr/local/pgsql/bin/pg_config
```

See [Installation Guide](../../NeuronDB/INSTALL.md) for detailed instructions.

## Quick Start

### Create Extension

```sql
CREATE EXTENSION neurondb;
```

### Create Vector Table

```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    embedding vector(384)
);
```

### Generate Embeddings

```sql
INSERT INTO documents (title, content, embedding)
VALUES (
    'Machine Learning',
    'Machine learning is a subset of AI',
    neurondb.embed_text('Machine learning is a subset of AI', 'model_name')
);
```

### Create Index

```sql
SELECT neurondb.hnsw_create_index('documents', 'embedding', 'documents_idx', 16, 200);
```

### Vector Search

```sql
SELECT id, title,
       embedding <-> neurondb.embed_text('artificial intelligence', 'model_name') AS distance
FROM documents
ORDER BY distance
LIMIT 5;
```

## Features

### Vector Search
- HNSW and IVF indexing
- Multiple distance metrics (L2, Cosine, Inner Product, etc.)
- Product Quantization (PQ) and Optimized PQ (OPQ)
- Support for multiple vector types

### Machine Learning
- Classification: Random Forest, XGBoost, SVM, Logistic Regression
- Regression: Linear, Ridge, Lasso, Deep Learning
- Clustering: K-Means, DBSCAN, GMM, Hierarchical
- Dimensionality Reduction: PCA, PCA Whitening
- Outlier Detection, Drift Detection, Quality Metrics

### GPU Acceleration
- CUDA support for NVIDIA GPUs
- ROCm support for AMD GPUs
- Metal support for Apple Silicon
- Automatic GPU detection and fallback

### Background Workers
- **neuranq**: Async job queue executor
- **neuranmon**: Live query auto-tuner
- **neurandefrag**: Index maintenance and defragmentation
- **neuranllm**: LLM job processor

## Configuration

See [Configuration Guide](../../NeuronDB/docs/configuration.md) for GUC parameters and settings.

## System Requirements

- PostgreSQL 16, 17, or 18
- Build tools: C compiler (GCC or Clang), Make
- Optional: CUDA, ROCm, or Metal for GPU acceleration

## Location

**Component Directory**: [`NeuronDB/`](../../NeuronDB/)

## Support

- **Documentation**: [NeuronDB/docs/](../../NeuronDB/docs/)
- **Official Docs**: [https://www.neurondb.ai/docs/neurondb](https://www.neurondb.ai/docs/neurondb)
- **Issues**: [GitHub Issues](https://github.com/neurondb/NeurondB/issues)

