# Changelog

All notable changes to NeuronDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release preparation
- GitHub Container Registry (GHCR) image publishing
- DEB and RPM package builds
- Comprehensive documentation

## [1.0.0] - 2026-01-10

### Added
- PostgreSQL extension for vector search with HNSW and IVF indexes
- 473+ SQL functions for vector operations, ML, and embeddings
- 52+ ML algorithms (classification, regression, clustering, dimensionality reduction)
- GPU acceleration support for CUDA, ROCm, and Metal backends
- Embedding generation with multiple provider support (OpenAI, HuggingFace, local models)
- RAG (Retrieval-Augmented Generation) pipelines with document processing
- Hybrid search combining vector similarity with full-text search
- NeuronAgent: REST/WebSocket API for agent runtime with workflow engine
- NeuronMCP: MCP protocol server with 100+ tools and resources
- NeuronDesktop: Web UI for ecosystem management and visualization
- Comprehensive benchmark suite (Vector, Hybrid, RAG benchmarks)
- Support for multiple vector types: vector, vectorp, vecmap, vgraph, rtext
- Product Quantization (PQ) and Optimized PQ (OPQ) for vector compression
- Multi-vector and faceted search capabilities
- Background workers for index maintenance and monitoring
- Comprehensive documentation and examples

### Changed
- Improved index build performance for large datasets
- Enhanced GPU memory management
- Optimized distance calculation performance

### Fixed
- Memory leaks in long-running queries
- Index corruption issues under high concurrency
- GPU initialization errors on some systems

### Supported Platforms
- PostgreSQL: 16, 17, 18
- GPU Backends: CPU, CUDA, ROCm, Metal
- Operating Systems: Ubuntu 20.04+, RHEL/CentOS 8+, macOS 13+
- Architectures: linux/amd64, linux/arm64

---

## Release Notes Format

Each release includes:

- **Version**: Semantic version (MAJOR.MINOR.PATCH)
- **Date**: Release date
- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security fixes

See [RELEASE.md](RELEASE.md) for release process documentation.

