# NeuronDB Ecosystem Components

The NeuronDB ecosystem consists of four integrated components that work together to provide a complete AI database and application platform.

## Components Overview

| Component | Purpose | Technology |
|-----------|---------|------------|
| **NeuronDB** | PostgreSQL extension for vector search and ML | C/CUDA extension |
| **NeuronAgent** | AI agent runtime system | Go |
| **NeuronMCP** | Model Context Protocol server | Go |
| **NeuronDesktop** | Unified web interface | Go + Next.js/React |

## Component Details

### [NeuronDB](neurondb.md)

PostgreSQL extension that adds:
- Vector search with HNSW and IVF indexing
- 52+ ML algorithms
- GPU acceleration (CUDA, ROCm, Metal)
- Hybrid search and RAG pipelines
- Background workers

**Documentation**: [NeuronDB Component Guide](neurondb.md) | [NeuronDB/docs/](../../NeuronDB/docs/) | [Official Docs](https://www.neurondb.ai/docs/neurondb)

### [NeuronAgent](neuronagent.md)

Agent runtime system providing:
- REST API and WebSocket endpoints
- Agent state machine
- Long-term memory with vector search
- Tool execution (SQL, HTTP, code, shell)
- Session and message management

**Documentation**: [NeuronAgent Component Guide](neuronagent.md) | [NeuronAgent/docs/](../../NeuronAgent/docs/) | [Official Docs](https://www.neurondb.ai/docs/neuronagent)

### [NeuronMCP](neuronmcp.md)

Model Context Protocol server enabling:
- MCP protocol (JSON-RPC 2.0)
- Stdio transport for MCP clients
- Vector operations and ML tools
- Resource management
- Claude Desktop integration

**Documentation**: [NeuronMCP Component Guide](neuronmcp.md) | [NeuronMCP/docs/](../../NeuronMCP/docs/) | [Official Docs](https://www.neurondb.ai/docs/neuronmcp)

### [NeuronDesktop](neurondesktop.md)

Unified web interface providing:
- Single dashboard for all components
- Real-time communication via WebSocket
- Secure authentication
- Professional UI with metrics
- MCP server integration

**Documentation**: [NeuronDesktop Component Guide](neurondesktop.md) | [NeuronDesktop/docs/](../../NeuronDesktop/docs/) | [Official Docs](https://www.neurondb.ai/docs/neurondesktop)

## Component Communication

```
┌─────────────┐
│ NeuronDesktop│ (Web UI)
└──────┬──────┘
       │
       ├──► NeuronAgent (HTTP/WebSocket)
       ├──► NeuronMCP (stdio)
       └──► NeuronDB (PostgreSQL)
              │
              ├──► NeuronAgent (PostgreSQL)
              └──► NeuronMCP (PostgreSQL)
```

All components connect to the same NeuronDB PostgreSQL instance. Services operate independently and can run separately.

## Getting Started

1. **[Install Components](../getting-started/installation.md)** - Installation guide
2. **[Component Guides](.)** - Detailed component documentation
3. **[Integration Guide](../ecosystem/integration.md)** - Connect components together

## Official Documentation

For comprehensive component documentation:
**🌐 [https://www.neurondb.ai/docs](https://www.neurondb.ai/docs)**

