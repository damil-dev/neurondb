# NeuronDB Ecosystem - Component Documentation

**Quick reference guide for all NeuronDB ecosystem components**

This directory provides component-specific documentation and ecosystem integration guides.

---

## 📚 Documentation Structure

```
Docs/
├── README.md                     # This file
├── components/                   # Individual component docs
│   ├── README.md                # Components overview
│   ├── neurondb.md              # NeuronDB extension
│   ├── neuronagent.md           # NeuronAgent runtime
│   ├── neuronmcp.md             # NeuronMCP server
│   └── neurondesktop.md         # NeuronDesktop interface
├── deployment/                   # Deployment guides
│   ├── README.md                # Deployment overview
│   └── docker.md                # Docker deployment
├── ecosystem/                    # Ecosystem integration
│   ├── README.md                # Ecosystem overview
│   └── integration.md           # Integration patterns
├── getting-started/             # Getting started guides
│   ├── installation.md          # Installation guide
│   └── quickstart.md            # Quick start tutorial
├── DOCKER.md                    # Docker reference
├── DOCUMENTATION.md             # Documentation index
└── PACKEGE.md                   # Package information
```

---

## 🎯 Quick Navigation

### By Role

**I'm a Developer** → Start with:
- [Quick Start](getting-started/quickstart.md)
- [Components Overview](components/README.md)
- [Integration Guide](ecosystem/integration.md)

**I'm a DevOps Engineer** → Start with:
- [Docker Deployment](deployment/docker.md)
- [Deployment Overview](deployment/README.md)
- [Ecosystem Guide](ecosystem/README.md)

**I'm an Architect** → Start with:
- [Ecosystem Overview](ecosystem/README.md)
- [Components Architecture](components/README.md)
- Main [README.md](../README.md)

---

## 🚀 Getting Started

### For New Users

1. **[Quick Start](getting-started/quickstart.md)** (5 minutes)
   - Get everything running fast
   - Run basic examples
   - Verify installation

2. **[Installation Guide](getting-started/installation.md)** (15 minutes)
   - Detailed installation steps
   - Platform-specific instructions
   - Troubleshooting tips

3. **[Components Overview](components/README.md)** (10 minutes)
   - Understand each component
   - Choose what you need
   - See how they integrate

---

## 📦 Component Documentation

### NeuronDB Extension

**[Component Guide](components/neurondb.md)**

PostgreSQL extension providing vector search, ML algorithms, and embeddings.

**Key Features:**
- 473 SQL functions
- 52+ ML algorithms
- Vector search (HNSW, IVF)
- GPU acceleration (CUDA, ROCm, Metal)
- RAG pipeline support

**Quick Links:**
- [README](../NeuronDB/README.md)
- [Installation](../NeuronDB/INSTALL.md)
- [SQL API Reference](../NeuronDB/docs/sql-api.md)
- [Full Documentation](../NeuronDB/docs/)

---

### NeuronAgent

**[Component Guide](components/neuronagent.md)**

REST/WebSocket agent runtime for autonomous AI agents.

**Key Features:**
- Agent state machine
- Long-term memory (vector-based)
- Tool registry (SQL, HTTP, Code, Shell)
- WebSocket streaming
- API key authentication

**Quick Links:**
- [README](../NeuronAgent/README.md)
- [API Reference](../NeuronAgent/docs/API.md)
- [Architecture](../NeuronAgent/docs/ARCHITECTURE.md)
- [Deployment](../NeuronAgent/docs/DEPLOYMENT.md)

---

### NeuronMCP

**[Component Guide](components/neuronmcp.md)**

Model Context Protocol server for MCP clients (Claude Desktop, etc.).

**Key Features:**
- 100+ MCP tools
- JSON-RPC 2.0 protocol
- Stdio/HTTP/SSE transports
- Vector operations
- ML tools
- PostgreSQL admin tools (27 tools)

**Quick Links:**
- [README](../NeuronMCP/README.md)
- [Tools Reference](../NeuronMCP/TOOLS_REFERENCE.md)
- [Setup Guide](../NeuronMCP/docs/NEURONDB_MCP_SETUP.md)
- [PostgreSQL Tools](../NeuronMCP/POSTGRESQL_TOOLS.md)

---

### NeuronDesktop

**[Component Guide](components/neurondesktop.md)**

Unified web interface for managing all ecosystem components.

**Key Features:**
- Unified dashboard
- Real-time updates (WebSocket)
- MCP integration
- NeuronAgent management
- SQL console
- Metrics & monitoring

**Quick Links:**
- [README](../NeuronDesktop/README.md)
- [API Reference](../NeuronDesktop/docs/API.md)
- [Integration Guide](../NeuronDesktop/docs/INTEGRATION.md)
- [Deployment](../NeuronDesktop/docs/DEPLOYMENT.md)

---

## 🏗️ Ecosystem Integration

### Running All Components Together

**[Ecosystem Overview](ecosystem/README.md)**

Learn how all components work together as a unified system.

**Topics Covered:**
- Architecture overview
- Component communication
- Data flow
- Network configuration
- Shared resources

**[Integration Guide](ecosystem/integration.md)**

Practical patterns for integrating components.

**Topics Covered:**
- Database setup
- Service configuration
- API integration
- WebSocket connections
- Error handling

---

## 🐳 Deployment

### Docker Deployment

**[Docker Guide](deployment/docker.md)**

Complete Docker deployment guide for all services.

**Deployment Options:**
- **Development:** Docker Compose (all services)
- **Production:** Individual containers with orchestration
- **GPU:** CUDA, ROCm, Metal variants

**Quick Start:**
```bash
# Start all services
docker compose up -d

# Verify services
docker compose ps

# Run smoke tests
./scripts/smoke-test.sh
```

**Quick Links:**
- [Main Docker README](../dockers/README.md)
- [NeuronDB Docker](../NeuronDB/docker/README.md)
- [NeuronAgent Docker](../NeuronAgent/docker/README.md)
- [NeuronMCP Docker](../NeuronMCP/docker/README.md)

---

### Production Deployment

**[Deployment Overview](deployment/README.md)**

Strategies for production deployments.

**Deployment Patterns:**
- Single server (Docker Compose)
- Multi-server (separate services)
- Kubernetes (container orchestration)
- Bare metal (high performance)

**Topics Covered:**
- Infrastructure requirements
- Scaling strategies
- High availability
- Monitoring & logging
- Security best practices
- Backup & recovery

---

## 📖 Additional Documentation

### Official Documentation Site

**🌐 [https://www.neurondb.ai/docs](https://www.neurondb.ai/docs)**

The official documentation site provides:
- Complete API references
- Detailed tutorials
- Performance guides
- Production best practices
- Latest updates

### Main Repository Documentation

| Document | Description |
|----------|-------------|
| **[Main README](../README.md)** | Project overview and architecture |
| **[Quick Start](../QUICKSTART.md)** | Get started in minutes |
| **[Documentation Index](../DOCUMENTATION.md)** | Complete documentation reference |
| **[Compatibility Matrix](../COMPATIBILITY.md)** | Version and platform compatibility |
| **[Contributing Guide](../CONTRIBUTING.md)** | How to contribute |
| **[Security Policy](../SECURITY.md)** | Security and vulnerability reporting |

### Component-Specific Documentation

| Component | Main Docs | Additional |
|-----------|-----------|------------|
| **NeuronDB** | [README](../NeuronDB/README.md) | [60+ doc files](../NeuronDB/docs/) |
| **NeuronAgent** | [README](../NeuronAgent/README.md) | [4 doc files](../NeuronAgent/docs/) |
| **NeuronMCP** | [README](../NeuronMCP/README.md) | [7 doc files](../NeuronMCP/docs/) |
| **NeuronDesktop** | [README](../NeuronDesktop/README.md) | [5 doc files](../NeuronDesktop/docs/) |

---

## 🎓 Learning Resources

### Tutorials & Examples

| Resource | Description |
|----------|-------------|
| **[Examples](../examples/)** | 4 comprehensive examples |
| **[NeuronDB Demo](../NeuronDB/demo/)** | 60+ SQL examples |
| **[NeuronAgent Examples](../NeuronAgent/examples/)** | 38 Python/Go examples |
| **[NeuronMCP Examples](../NeuronMCP/docs/examples/)** | MCP client examples |

### By Use Case

| Use Case | Resources |
|----------|-----------|
| **Vector Search** | [Example](../examples/semantic-search-docs/), [Docs](../NeuronDB/docs/vector-search/) |
| **RAG Applications** | [Example](../examples/rag-chatbot-pdfs/), [Docs](../NeuronDB/docs/rag/) |
| **AI Agents** | [Example](../examples/agent-tools/), [Docs](../NeuronAgent/docs/API.md) |
| **MCP Integration** | [Example](../examples/mcp-integration/), [Docs](../NeuronMCP/README.md) |

---

## 🔧 Configuration Reference

### Database Configuration

```bash
# Environment variables (all components)
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=neurondb
export DB_USER=neurondb
export DB_PASSWORD=neurondb
```

### Service Configuration

| Component | Default Port | Config File |
|-----------|--------------|-------------|
| **NeuronDB** | 5432/5433 | N/A (PostgreSQL config) |
| **NeuronAgent** | 8080 | `config.yaml` |
| **NeuronMCP** | stdio | `mcp-config.json` |
| **NeuronDesktop API** | 8081 | Environment variables |
| **NeuronDesktop Frontend** | 3000 | Environment variables |

---

## 🛠️ Utilities & Scripts

### Setup Scripts

| Script | Description |
|--------|-------------|
| `scripts/neurondb-setup.sh` | Complete ecosystem setup |
| `scripts/setup_neurondb_ecosystem.sh` | Unified database setup |

### Testing Scripts

| Script | Description |
|--------|-------------|
| `scripts/smoke-test.sh` | Quick health checks |
| `scripts/verify_neurondb_integration.sh` | Comprehensive tests |

### Docker Scripts

| Script | Description |
|--------|-------------|
| `scripts/run_neurondb_docker.sh` | Run database container |
| `scripts/run_neuronagent_docker.sh` | Run agent container |
| `scripts/run_neuronmcp_docker.sh` | Run MCP container |

See [scripts/README.md](../scripts/README.md) for complete reference.

---

## 🆘 Support & Community

### Getting Help

| Resource | Description |
|----------|-------------|
| **[Official Documentation](https://www.neurondb.ai/docs)** | Complete documentation |
| **[Troubleshooting Guide](../NeuronDB/docs/troubleshooting.md)** | Common issues |
| **[GitHub Issues](https://github.com/neurondb/NeurondB/issues)** | Report bugs |
| **[GitHub Discussions](https://github.com/neurondb/NeurondB/discussions)** | Ask questions |
| **Email Support** | support@neurondb.ai |

### Contributing

We welcome contributions! See:
- **[Contributing Guide](../CONTRIBUTING.md)** - How to contribute
- **[Code of Conduct](../CODE_OF_CONDUCT.md)** - Community guidelines

---

## 📊 Documentation Statistics

| Metric | Count |
|--------|-------|
| **Total Documentation Files** | 100+ |
| **Component READMEs** | 4 |
| **NeuronDB Docs** | 60+ files |
| **API References** | 4 (SQL, REST, MCP, Web UI) |
| **Examples** | 4 main + 100+ code examples |
| **Setup Scripts** | 19 scripts |

---

## 🗺️ Documentation Roadmap

### Planned Documentation

- [ ] Advanced ML algorithms guide
- [ ] Performance tuning cookbook
- [ ] Security hardening guide
- [ ] Kubernetes deployment guide
- [ ] Multi-region deployment patterns
- [ ] Migration guide (from other vector DBs)
- [ ] Upgrade guide (version migration)
- [ ] Video tutorials
- [ ] Interactive examples

---

## 📝 Documentation Standards

This documentation follows these standards:

### Style Guide

- **Crisp & Clear:** Direct, actionable information
- **Consistent Structure:** Standard sections across documents
- **Code Examples:** Working, tested examples
- **Navigation:** Clear links and cross-references
- **Updates:** Regular updates with version changes

### File Organization

- **README files:** Overview and quick start for each directory
- **Topic files:** Detailed documentation for specific topics
- **API references:** Complete function/endpoint documentation
- **Examples:** Separate directory with working examples

---

**Last Updated:** 2025-01-30  
**Documentation Version:** 1.0.0

**📚 For the most comprehensive, up-to-date documentation, always visit [https://www.neurondb.ai/docs](https://www.neurondb.ai/docs)**
