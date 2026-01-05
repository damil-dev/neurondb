# NeuronDB Roadmap

This document outlines the planned features and improvements for NeuronDB.

## 2025 Q1-Q2

### High Priority

1. **Release Readiness** (Q1)
   - v1.0.0 release with GHCR images and packages
   - Comprehensive test coverage for all components
   - Performance benchmarks and optimization

2. **Enhanced GPU Support** (Q1)
   - Improved CUDA performance
   - Better ROCm compatibility
   - Metal optimization for Apple Silicon

3. **Documentation Improvements** (Q1)
   - SQL API reference documentation
   - Operational runbooks
   - Migration guides

4. **Performance Optimization** (Q2)
   - Index build time improvements
   - Query latency reduction
   - Memory usage optimization

### Medium Priority

5. **Additional Index Types** (Q2)
   - ScaNN index support
   - DiskANN index (for very large datasets)
   - Auto-index selection

6. **Enhanced RAG Features** (Q2)
   - Improved reranking
   - Multi-modal support
   - Better context window management

7. **Monitoring and Observability** (Q2)
   - Prometheus metrics export
   - Grafana dashboards
   - Performance insights

### Low Priority

8. **Community Features** (Q2-Q3)
   - Plugin system for custom functions
   - Community-contributed algorithms
   - Example applications gallery

## 2025 Q3-Q4

### Planned Features

9. **Distributed Vector Search** (Q3)
   - Multi-node query distribution
   - Replication strategies
   - Load balancing

10. **Advanced ML Features** (Q3)
    - Online learning
    - Model versioning
    - A/B testing framework

11. **Deployment Features** (Q4)
    - High availability setup
    - Backup and restore automation
    - Multi-tenant support

12. **Developer Experience** (Q4)
    - Python client library
    - TypeScript/JavaScript SDK
    - CLI improvements

## Future Considerations

- Cloud-native deployment (Kubernetes operators)
- Serverless support
- Integration with more LLM providers
- Additional database backends
- Real-time streaming updates

## How to Contribute

We welcome community input on the roadmap:

1. **GitHub Discussions**: Share feature requests and ideas
2. **GitHub Issues**: Report bugs or suggest enhancements
3. **Pull Requests**: Contribute code improvements

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Milestone Tracking

Milestones are tracked in GitHub:
- [Milestones](https://github.com/neurondb/neurondb/milestones)

Each milestone links to relevant issues and pull requests.

## Version Planning

- **v1.0.0**: Initial stable release (Q1 2025)
- **v1.1.0**: Performance improvements and bug fixes (Q2 2025)
- **v1.2.0**: Enhanced GPU support and new index types (Q2 2025)
- **v2.0.0**: Major features (distributed search, deployment features) (Q3-Q4 2025)

> [!NOTE]
> This roadmap is subject to change based on community feedback and priorities.

