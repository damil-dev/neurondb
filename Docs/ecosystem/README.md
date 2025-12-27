# NeuronDB Ecosystem Overview

The NeuronDB ecosystem consists of four integrated components that work together to provide a complete AI database and application platform.

## Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **NeuronDB** | PostgreSQL extension for vector search and ML | C/CUDA extension |
| **NeuronAgent** | AI agent runtime system | Go |
| **NeuronMCP** | Model Context Protocol server | Go |
| **NeuronDesktop** | Unified web interface | Go + Next.js/React |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    NeuronDesktop                         │
│              (Web Interface - Port 3000/8081)            │
└──────┬──────────────┬───────────────┬──────────────────┘
       │              │               │
       │ HTTP/WS      │ stdio         │ HTTP
       │              │               │
┌──────▼──────┐  ┌───▼────┐  ┌───────▼────────┐
│ NeuronAgent │  │NeuronMCP│  │   NeuronDB     │
│ (Port 8080)│  │ (stdio) │  │ (PostgreSQL)   │
└──────┬──────┘  └────┬────┘  │ (Port 5432)    │
       │              │       └───────┬────────┘
       │              │               │
       └──────────────┴───────────────┘
                    PostgreSQL
```

## Component Communication

All components connect to the same NeuronDB PostgreSQL instance:

- **NeuronDB**: Database server with extension
- **NeuronAgent**: Connects via PostgreSQL connection string
- **NeuronMCP**: Connects via PostgreSQL connection string
- **NeuronDesktop**: Connects to all three components

### Communication Matrix

| Component | Connection Method | Protocol | Port | Purpose |
|-----------|------------------|----------|------|---------|
| **NeuronDB** | PostgreSQL native | TCP | 5432/5433 | Database server with extension |
| **NeuronAgent** | Database connection | TCP | 5432/5433 | Agent runtime data access |
| **NeuronAgent** | HTTP REST API | HTTP | 8080 | Client API access |
| **NeuronAgent** | WebSocket | WS | 8080 | Streaming responses |
| **NeuronMCP** | Database connection | TCP | 5432/5433 | MCP server data access |
| **NeuronMCP** | Stdio | JSON-RPC 2.0 | - | MCP client communication |
| **NeuronDesktop** | HTTP/WebSocket | HTTP/WS | 3000/8081 | Web interface |
| **NeuronDesktop** | Database connection | TCP | 5432/5433 | NeuronDB access |
| **NeuronDesktop** | HTTP | HTTP | 8080 | NeuronAgent access |
| **NeuronDesktop** | Stdio | JSON-RPC 2.0 | - | NeuronMCP access |

## Data Flow

1. **Client Requests** → NeuronDesktop (HTTP/WebSocket) or NeuronAgent (HTTP/WebSocket) or NeuronMCP (stdio)
2. **Service Processing** → Agent runtime, MCP protocol handler, or web interface
3. **Database Queries** → NeuronDB PostgreSQL extension
4. **Vector/ML Operations** → Extension executes vector search, ML algorithms
5. **Results Return** → Through service layer back to clients

## Key Features by Layer

### Client Layer
- Web applications via NeuronDesktop
- REST API clients via NeuronAgent
- MCP clients (Claude Desktop) via NeuronMCP
- Mobile apps via REST/WebSocket

### Service Layer
- **NeuronAgent**: Agent state machine, tool execution, memory management
- **NeuronMCP**: MCP protocol, tool/resource handlers, middleware
- **NeuronDesktop**: Unified interface, real-time communication, metrics

### Database Layer
- Vector search with HNSW/IVF indexing
- 52+ ML algorithms (classification, regression, clustering, etc.)
- Embedding generation (text, image, multimodal)
- Hybrid search combining vector and full-text
- RAG pipeline with LLM integration
- GPU acceleration (CUDA, ROCm, Metal)
- Background workers for async operations

## Use Cases

### Vector Search Applications
- Semantic search
- Product recommendations
- Content similarity matching
- Image search

### RAG Pipelines
- Document Q&A systems
- Knowledge bases
- Chatbots with context
- Research assistants

### Agent Applications
- Autonomous task execution
- Multi-step workflows
- Tool orchestration
- Long-term memory systems

### MCP Integration
- Claude Desktop integration
- Custom MCP clients
- Tool discovery and execution
- Resource management

## Getting Started

1. **[Installation Guide](../getting-started/installation.md)** - Install all components
2. **[Quick Start Guide](../getting-started/quickstart.md)** - Get up and running quickly
3. **[Integration Guide](integration.md)** - Connect components together
4. **[Component Documentation](../components/README.md)** - Learn about each component

## Deployment

- **[Deployment Overview](../deployment/README.md)** - Deployment options
- **[Docker Deployment](../deployment/docker.md)** - Container deployment
- **[Component-Specific Guides](../components/)** - Individual component deployment

## Official Documentation

For comprehensive ecosystem documentation:
**🌐 [https://www.neurondb.ai/docs/ecosystem](https://www.neurondb.ai/docs/ecosystem)**

