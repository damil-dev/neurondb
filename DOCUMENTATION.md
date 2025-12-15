# NeuronDB Ecosystem - Complete Documentation

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://www.neurondb.ai/docs)
[![Website](https://img.shields.io/badge/website-www.neurondb.ai-blue.svg)](https://www.neurondb.ai/)

## Official Documentation Site

**🌐 [https://www.neurondb.ai/docs](https://www.neurondb.ai/docs)**

The official documentation site provides comprehensive guides, tutorials, API references, and best practices for the entire NeuronDB ecosystem.

## Quick Navigation

### Getting Started
- **[Official Getting Started Guide](https://www.neurondb.ai/docs/getting-started)** - Complete installation and setup
- **[Quick Start Tutorial](https://www.neurondb.ai/docs/getting-started/quickstart)** - Get up and running in minutes
- **[Installation Guide](https://www.neurondb.ai/docs/installation)** - Platform-specific installation instructions

### Core Components

#### NeuronDB
- **[NeuronDB Documentation](https://www.neurondb.ai/docs/neurondb)** - Complete PostgreSQL extension guide
- **[Vector Search Guide](https://www.neurondb.ai/docs/vector-search)** - HNSW indexing, distance metrics, quantization
- **[ML Algorithms Guide](https://www.neurondb.ai/docs/ml-algorithms)** - 52 ML algorithms with examples
- **[RAG Pipeline Guide](https://www.neurondb.ai/docs/rag)** - Complete RAG implementation
- **[GPU Acceleration Guide](https://www.neurondb.ai/docs/gpu)** - CUDA, ROCm, Metal support
- **[Hybrid Search Guide](https://www.neurondb.ai/docs/hybrid-search)** - Vector + full-text search
- **[Performance Guide](https://www.neurondb.ai/docs/performance)** - Optimization strategies
- **[Security Guide](https://www.neurondb.ai/docs/security)** - Encryption, privacy, RLS

#### NeuronAgent
- **[NeuronAgent Documentation](https://www.neurondb.ai/docs/neuronagent)** - Agent runtime system
- **[API Reference](https://www.neurondb.ai/docs/neuronagent/api)** - Complete REST API documentation
- **[Architecture Guide](https://www.neurondb.ai/docs/neuronagent/architecture)** - System design
- **[Deployment Guide](https://www.neurondb.ai/docs/neuronagent/deployment)** - Production deployment

#### NeuronMCP
- **[NeuronMCP Documentation](https://www.neurondb.ai/docs/neuronmcp)** - Model Context Protocol server
- **[MCP Integration Guide](https://www.neurondb.ai/docs/neuronmcp/integration)** - Claude Desktop setup
- **[Tool Reference](https://www.neurondb.ai/docs/neuronmcp/tools)** - Available MCP tools

### Advanced Topics

#### Vector Operations
- **[Vector Types](https://www.neurondb.ai/docs/vector-search/vector-types)** - vector, vectorp, vecmap, vgraph, rtext
- **[Indexing Strategies](https://www.neurondb.ai/docs/vector-search/indexing)** - HNSW, IVF, optimization
- **[Distance Metrics](https://www.neurondb.ai/docs/vector-search/distance-metrics)** - L2, Cosine, Inner Product, and more
- **[Quantization](https://www.neurondb.ai/docs/vector-search/quantization)** - FP16, INT8, Binary compression

#### Machine Learning
- **[Classification](https://www.neurondb.ai/docs/ml-algorithms/classification)** - Random Forest, XGBoost, SVM, KNN
- **[Regression](https://www.neurondb.ai/docs/ml-algorithms/regression)** - Linear, Ridge, Lasso, Deep Learning
- **[Clustering](https://www.neurondb.ai/docs/ml-algorithms/clustering)** - K-Means, DBSCAN, GMM
- **[Dimensionality Reduction](https://www.neurondb.ai/docs/ml-algorithms/dimensionality-reduction)** - PCA, PCA Whitening
- **[Embedding Generation](https://www.neurondb.ai/docs/ml-embeddings/embedding-generation)** - Text, image, multimodal
- **[Model Inference](https://www.neurondb.ai/docs/ml-embeddings/model-inference)** - ONNX runtime, batch processing

#### Retrieval & Reranking
- **[RAG Pipeline](https://www.neurondb.ai/docs/rag)** - Complete in-database RAG
- **[Hybrid Search](https://www.neurondb.ai/docs/hybrid-search)** - Vector + full-text combination
- **[Reranking](https://www.neurondb.ai/docs/reranking)** - Cross-encoder, LLM, ColBERT
- **[Multi-Vector Search](https://www.neurondb.ai/docs/hybrid-search/multi-vector)** - Multiple embeddings per document

#### Performance & Operations
- **[Performance Optimization](https://www.neurondb.ai/docs/performance)** - SIMD, query planning, caching
- **[GPU Acceleration](https://www.neurondb.ai/docs/gpu)** - CUDA, ROCm, Metal setup
- **[Monitoring](https://www.neurondb.ai/docs/performance/monitoring)** - Metrics, Prometheus, observability
- **[Background Workers](https://www.neurondb.ai/docs/background-workers)** - Async job processing

#### Deployment & Operations
- **[Docker Deployment](https://www.neurondb.ai/docs/docker)** - Container deployment guide
- **[Ecosystem Setup](https://www.neurondb.ai/docs/ecosystem)** - Running all components together
- **[Configuration](https://www.neurondb.ai/docs/configuration)** - Complete configuration reference
- **[Troubleshooting](https://www.neurondb.ai/docs/troubleshooting)** - Common issues and solutions

## Local Documentation

This repository contains local documentation files for quick reference. For the most up-to-date and comprehensive documentation, always refer to the official site.

### Component READMEs
- [NeuronDB README](NeuronDB/README.md) - Local feature overview
- [NeuronAgent README](NeuronAgent/README.md) - Local component guide
- [NeuronMCP README](NeuronMCP/README.md) - Local component guide

### Installation Guides
- [NeuronDB Installation](NeuronDB/INSTALL.md) - Build from source
- [Docker Guides](NeuronDB/docker/README.md) - Container deployment

### API References
- [SQL API Reference](NeuronDB/docs/sql-api.md) - Function documentation
- [NeuronAgent API](NeuronAgent/docs/API.md) - REST API reference

## Documentation Structure

### Official Site Organization

The official documentation at [www.neurondb.ai/docs](https://www.neurondb.ai/docs) is organized into:

1. **Getting Started** - Installation, quick start, first steps
2. **NeuronDB** - Core extension features and capabilities
3. **NeuronAgent** - Agent runtime system
4. **NeuronMCP** - Model Context Protocol server
5. **Ecosystem** - Running all components together
6. **API Reference** - Complete function and API documentation
7. **Guides** - Detailed tutorials and how-tos
8. **Deployment** - Production deployment guides
9. **Troubleshooting** - Common issues and solutions

### Feature Categories

#### Vector Search
- Vector types and operations
- Indexing (HNSW, IVF)
- Distance metrics
- Quantization
- Performance optimization

#### Machine Learning
- 52 ML algorithms
- Model training and inference
- Embedding generation
- AutoML and feature store
- Analytics and clustering

#### Retrieval & RAG
- Hybrid search
- Reranking strategies
- Document processing
- LLM integration
- Multi-vector search

#### Performance
- GPU acceleration
- SIMD optimization
- Query planning
- Caching strategies
- Monitoring and metrics

#### Security
- Vector encryption
- Differential privacy
- Row-level security
- Multi-tenant isolation
- Authentication and authorization

## Key Resources

### Official Links
- **Main Website**: [https://www.neurondb.ai/](https://www.neurondb.ai/)
- **Documentation**: [https://www.neurondb.ai/docs](https://www.neurondb.ai/docs)
- **Blog**: [https://www.neurondb.ai/blog](https://www.neurondb.ai/blog)
- **Tutorials**: [https://www.neurondb.ai/tutorials](https://www.neurondb.ai/tutorials)

### Community
- **GitHub**: [https://github.com/neurondb/NeurondB](https://github.com/neurondb/NeurondB)
- **Issues**: [https://github.com/neurondb/NeurondB/issues](https://github.com/neurondb/NeurondB/issues)
- **Support**: support@neurondb.ai

## Documentation Best Practices

### When to Use Official Docs
- **Always** refer to [www.neurondb.ai/docs](https://www.neurondb.ai/docs) for:
  - Latest API changes and new features
  - Production deployment best practices
  - Performance optimization strategies
  - Troubleshooting specific issues
  - Complete function signatures and examples

### When to Use Local Docs
- Quick reference during development
- Component-specific setup instructions
- Local build and installation
- Repository structure understanding

## Contributing to Documentation

Documentation improvements are welcome! Please:
1. Check the official site first to ensure consistency
2. Update local docs to match official documentation
3. Submit improvements via GitHub pull requests
4. Reference official documentation links where appropriate

## Version Information

This documentation corresponds to the NeuronDB ecosystem version in this repository. For the latest version information and release notes, visit:
- [Release Notes](https://www.neurondb.ai/docs/whats-new)
- [Changelog](https://www.neurondb.ai/docs/changelog)

---

**For the most comprehensive, up-to-date documentation, always visit [https://www.neurondb.ai/docs](https://www.neurondb.ai/docs)**


