# NeuronDB Complete Documentation Index

**Complete index of all documentation in the NeuronDB ecosystem.**

> **Version:** 1.0  
> **Last Updated:** 2025-01-01

## Quick Navigation

- [Getting Started](#getting-started)
- [Reference Documentation](#reference-documentation)
- [Internals Documentation](#internals-documentation)
- [Advanced Features](#advanced-features)
- [Development](#development)
- [Deployment](#deployment)
- [Ecosystem Integration](#ecosystem-integration)

---

## Getting Started

### Quick Start Guides

- **[QUICKSTART.md](../../QUICKSTART.md)** - Get all services running in minutes
- **[Simple Start Guide](getting-started/simple-start.md)** - Beginner-friendly setup
- **[Architecture Overview](getting-started/architecture.md)** - Understand the architecture
- **[Troubleshooting](getting-started/troubleshooting.md)** - Common issues and solutions

---

## Reference Documentation

### SQL API

- **[SQL API Complete Reference](reference/sql-api-complete.md)** - All 654+ SQL functions, types, operators, and aggregates
  - Vector operations
  - Distance metrics
  - Quantization functions
  - Indexing functions
  - Embedding generation
  - Hybrid search
  - Reranking
  - Machine learning
  - RAG functions
  - LLM functions
  - Utility functions

### Data Types

- **[Data Types Complete Reference](reference/data-types.md)** - All data types with C structures
  - Vector types (vector, halfvec, sparsevec, binaryvec, etc.)
  - Internal C structures
  - Type storage formats
  - Type casting rules
  - Memory layout
  - Quantization formats

### Configuration

- **[Configuration Complete Reference](reference/configuration-complete.md)** - All GUC variables
  - Core/index settings
  - GPU settings
  - LLM settings
  - Worker settings
  - ONNX Runtime settings
  - Quota settings
  - AutoML settings

### Component APIs

- **[NeuronAgent API Complete Reference](reference/neuronagent-api-complete.md)** - REST and WebSocket API
  - Agents
  - Sessions
  - Messages
  - Tools
  - Memory
  - Webhooks

- **[NeuronMCP Tools Complete Reference](reference/neuronmcp-tools-complete.md)** - All 100+ MCP tools
  - Vector operations
  - Embedding functions
  - Hybrid search
  - Reranking
  - Machine learning
  - RAG operations
  - PostgreSQL administration (27 tools)
  - Analytics tools

- **[NeuronDesktop API Complete Reference](reference/neurondesktop-api-complete.md)** - REST and WebSocket API
  - Profiles
  - NeuronDB operations
  - Agent integration
  - MCP integration
  - Model management
  - Database management

---

## Internals Documentation

### Architecture

- **[NeuronDB Internal Architecture](internals/architecture-complete.md)** - Complete internal architecture
  - Source code organization
  - Module breakdown
  - Data flow
  - Memory management
  - Threading model
  - Index structures

- **[NeuronAgent Internal Architecture](internals/neuronagent-architecture.md)** - Agent runtime architecture
  - Component breakdown
  - State machine
  - Data flow
  - Memory management
  - Tool execution flow

- **[NeuronDesktop Frontend Architecture](internals/neurondesktop-frontend.md)** - Frontend architecture
  - Technology stack
  - Project structure
  - Component architecture
  - State management
  - API client
  - WebSocket integration

### Index Methods

- **[Index Methods Complete Reference](internals/index-methods.md)** - All index types
  - HNSW index
  - IVF index
  - Hybrid index
  - Temporal index
  - Sparse index
  - Index tuning
  - Index maintenance

---

## Advanced Features

### GPU Acceleration

- **[GPU Acceleration Complete Reference](advanced/gpu-acceleration-complete.md)** - CUDA, ROCm, Metal
  - GPU backend interface
  - CUDA implementation
  - ROCm implementation
  - Metal implementation
  - Memory management
  - Kernel implementations
  - Performance tuning

### Machine Learning

- **[ML Algorithms Complete Reference](advanced/ml-algorithms-complete.md)** - All 19 ML algorithms
  - Clustering (K-Means, DBSCAN, GMM, Hierarchical)
  - Classification (Random Forest, Logistic Regression, SVM, etc.)
  - Regression (Linear, Ridge, Lasso)
  - Dimensionality reduction (PCA)
  - Quantization (PQ, OPQ)
  - Outlier detection
  - Time series (ARIMA)
  - Recommendation systems

### RAG Pipeline

- **[RAG Pipeline Complete Reference](advanced/rag-pipeline-complete.md)** - Complete RAG system
  - Document processing
  - Chunking strategies
  - Embedding generation
  - Retrieval methods
  - Reranking strategies
  - LLM integration
  - End-to-end examples

---

## Development

### Build System

- **[Build System Documentation](development/build-system.md)** - Complete build system
  - Makefile structure
  - Build targets
  - Platform-specific builds
  - GPU backend compilation
  - Dependency management
  - Testing infrastructure

### Development Guide

- **[Development Guide](development/development-guide.md)** - Development procedures
  - Code organization
  - Adding new SQL functions
  - Adding new ML algorithms
  - Adding new tools
  - Testing procedures
  - Debugging guides

---

## Deployment

- **[Deployment Complete Guide](deployment/deployment-complete.md)** - Complete deployment guide
  - Docker deployment (all profiles)
  - Native installation
  - Production considerations
  - Scaling strategies
  - Monitoring setup
  - Backup and recovery

---

## Ecosystem Integration

- **[Ecosystem Integration Complete Guide](ecosystem/integration-complete.md)** - Integration guide
  - Component communication
  - Data flow
  - Authentication
  - Configuration sharing
  - Deployment coordination
  - Integration examples

---

## Documentation Statistics

### Coverage

- **SQL Functions:** 654+ functions documented
- **Data Types:** 8+ types documented
- **Configuration Options:** 30+ GUC variables documented
- **API Endpoints:** 50+ endpoints documented
- **MCP Tools:** 100+ tools documented
- **ML Algorithms:** 19 algorithms documented
- **Index Methods:** 5 index types documented

### Documentation Files

- **Reference:** 6 files
- **Internals:** 4 files
- **Advanced:** 3 files
- **Development:** 2 files
- **Deployment:** 1 file
- **Ecosystem:** 1 file

**Total:** 17 comprehensive documentation files

---

## Related Documentation

- **[Main Documentation Index](../../DOCUMENTATION.md)** - Original documentation index
- **[Contributing Guide](../../CONTRIBUTING.md)** - Contribution guidelines
- **[README](../../README.md)** - Project overview

---

**Last Updated:** 2025-01-01  
**Documentation Version:** 1.0.0



