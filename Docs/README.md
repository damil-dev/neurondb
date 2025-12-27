# NeuronDB Ecosystem Documentation

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://www.neurondb.ai/docs)
[![Website](https://img.shields.io/badge/website-www.neurondb.ai-blue.svg)](https://www.neurondb.ai/)

## Overview

The NeuronDB ecosystem consists of four integrated components that work together to provide a complete AI database and application platform:

- **NeuronDB** - PostgreSQL extension for vector search, ML algorithms, and embeddings
- **NeuronAgent** - AI agent runtime with REST API and WebSocket support
- **NeuronMCP** - Model Context Protocol server for MCP-compatible clients
- **NeuronDesktop** - Unified web interface for managing all ecosystem components

## Official Documentation

**🌐 [https://www.neurondb.ai/docs](https://www.neurondb.ai/docs)**

For comprehensive documentation, tutorials, API references, and best practices, visit the official documentation site.

## Quick Navigation

### Getting Started
- **[Getting Started Guide](getting-started/README.md)** - Overview of getting started
- **[Installation](getting-started/installation.md)** - Install the ecosystem
- **[Quick Start](getting-started/quickstart.md)** - Get up and running quickly

### Components
- **[Components Overview](components/README.md)** - All ecosystem components
- **[NeuronDB](components/neurondb.md)** - PostgreSQL extension documentation
- **[NeuronAgent](components/neuronagent.md)** - Agent runtime documentation
- **[NeuronMCP](components/neuronmcp.md)** - MCP server documentation
- **[NeuronDesktop](components/neurondesktop.md)** - Web interface documentation

### Deployment
- **[Deployment Overview](deployment/README.md)** - Deployment options
- **[Docker Deployment](deployment/docker.md)** - Container deployment guide

### Ecosystem
- **[Ecosystem Overview](ecosystem/README.md)** - Running all components together
- **[Integration Guide](ecosystem/integration.md)** - Component integration

## Component Documentation

Each component has its own detailed documentation:

| Component | Local Documentation | Official Documentation |
|-----------|---------------------|----------------------|
| **NeuronDB** | [NeuronDB/docs/](../NeuronDB/docs/) | [NeuronDB Guide](https://www.neurondb.ai/docs/neurondb) |
| **NeuronAgent** | [NeuronAgent/docs/](../NeuronAgent/docs/) | [NeuronAgent Guide](https://www.neurondb.ai/docs/neuronagent) |
| **NeuronMCP** | [NeuronMCP/docs/](../NeuronMCP/docs/) | [NeuronMCP Guide](https://www.neurondb.ai/docs/neuronmcp) |
| **NeuronDesktop** | [NeuronDesktop/docs/](../NeuronDesktop/docs/) | [NeuronDesktop Guide](https://www.neurondb.ai/docs/neurondesktop) |

## Key Features

### NeuronDB
- Vector search with HNSW and IVF indexing
- 52+ ML algorithms (classification, regression, clustering)
- GPU acceleration (CUDA, ROCm, Metal)
- Hybrid search combining vector and full-text
- RAG pipeline with LLM integration
- Background workers for async operations

### NeuronAgent
- Agent state machine for autonomous task execution
- Long-term memory with HNSW vector search
- Tool registry (SQL, HTTP, code, shell)
- REST API and WebSocket support
- API key authentication with rate limiting

### NeuronMCP
- Model Context Protocol (JSON-RPC 2.0)
- Stdio transport for MCP clients
- Vector operations and ML tools
- Resource management (schema, models, indexes)

### NeuronDesktop
- Unified web interface for all components
- Real-time communication via WebSocket
- Secure authentication with API keys
- Professional UI with metrics and monitoring
- MCP server integration and testing

## Support & Resources

- **Official Documentation**: [https://www.neurondb.ai/docs](https://www.neurondb.ai/docs)
- **GitHub**: [https://github.com/neurondb/NeurondB](https://github.com/neurondb/NeurondB)
- **Issues**: [Report Issues](https://github.com/neurondb/NeurondB/issues)
- **Email**: support@neurondb.ai

---

**For the most comprehensive, up-to-date documentation, always visit [https://www.neurondb.ai/docs](https://www.neurondb.ai/docs)**

