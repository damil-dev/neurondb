# NeuronMCP Component

Model Context Protocol server enabling MCP clients to access NeuronDB through stdio communication.

## Overview

NeuronMCP implements the Model Context Protocol using JSON-RPC 2.0 over stdio. It provides tools and resources for MCP clients to interact with NeuronDB, including vector operations, ML model training, and database schema management.

## Key Capabilities

| Feature | Description |
|---------|-------------|
| **MCP Protocol** | JSON-RPC 2.0 implementation with stdio transport |
| **Vector Operations** | 50+ tools for search, embedding generation, indexing |
| **ML Tools** | Complete ML pipeline: training, prediction, evaluation |
| **Resource Management** | Schema, models, indexes, config, workers, stats |
| **Middleware** | Validation, logging, timeouts, error handling |
| **Security** | JWT, API keys, OAuth2 authentication with rate limiting |
| **Performance** | Caching layer with TTL, connection pooling |
| **Enterprise** | Metrics (Prometheus), webhooks, retry/resilience |

## Documentation

### Local Documentation

- **[Component README](../../NeuronMCP/README.md)** - Overview and usage
- **[Docker Guide](../../NeuronMCP/docker/README.md)** - Container deployment
- **[Setup Guide](../../NeuronMCP/SETUP_GUIDE.md)** - Setup instructions
- **[Tools Reference](../../NeuronMCP/TOOLS_REFERENCE.md)** - Available tools
- **[PostgreSQL Tools](../../NeuronMCP/POSTGRESQL_TOOLS.md)** - PostgreSQL-specific tools

### Official Documentation

- **[NeuronMCP Guide](https://www.neurondb.ai/docs/neuronmcp)** - Model Context Protocol server
- **[MCP Integration Guide](https://www.neurondb.ai/docs/neuronmcp/integration)** - Claude Desktop setup
- **[Tool Reference](https://www.neurondb.ai/docs/neuronmcp/tools)** - Available MCP tools

## Installation

### Docker (Recommended)

```bash
cd NeuronMCP/docker
docker compose up -d neurondb-mcp
```

### Source Build

```bash
cd NeuronMCP
go build ./cmd/neurondb-mcp
```

## Quick Start

### Database Setup

```bash
createdb neurondb
psql -d neurondb -c "CREATE EXTENSION neurondb;"
cd NeuronMCP
./scripts/setup_neurondb_mcp.sh
```

### Configuration

Set environment variables:

```bash
export NEURONDB_HOST=localhost
export NEURONDB_PORT=5432
export NEURONDB_DATABASE=neurondb
export NEURONDB_USER=neurondb
export NEURONDB_PASSWORD=neurondb
```

### Claude Desktop Configuration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "neurondb": {
      "command": "neurondb-mcp",
      "env": {
        "NEURONDB_HOST": "localhost",
        "NEURONDB_PORT": "5432",
        "NEURONDB_DATABASE": "neurondb",
        "NEURONDB_USER": "neurondb",
        "NEURONDB_PASSWORD": "neurondb"
      }
    }
  }
}
```

## Features

### MCP Protocol
- Full JSON-RPC 2.0 implementation
- Stdio transport for MCP clients
- Support for tools, resources, prompts, and sampling

### Vector Operations
- Vector search with multiple distance metrics
- Embedding generation (text, image, multimodal)
- Index creation and management (HNSW, IVF)
- Vector operations and transformations

### ML Tools
- Model training for all ML algorithms
- Prediction and inference
- Model evaluation and metrics
- AutoML and hyperparameter tuning

### Resource Management
- Schema inspection and management
- Model catalog and versioning
- Index configuration and status
- Worker configuration and monitoring
- System statistics and health

### Middleware
- Request validation
- Logging and observability
- Timeout handling
- Error recovery
- Authentication and authorization
- Rate limiting

## Architecture

```
┌─────────────────────────────────────────────┐
│          MCP Client                         │
│  (Claude Desktop, etc.)                     │
└──────────────┬──────────────────────────────┘
               │ stdio (JSON-RPC 2.0)
┌──────────────▼──────────────────────────────┐
│          NeuronMCP Server                   │
├─────────────────────────────────────────────┤
│  MCP Protocol Handler                       │
├─────────────────────────────────────────────┤
│  Tools │  Resources │  Middleware           │
├─────────────────────────────────────────────┤
│          NeuronDB PostgreSQL                │
│  (Vector Search │  ML │  Embeddings)        │
└─────────────────────────────────────────────┘
```

## Configuration

See [Setup Guide](../../NeuronMCP/SETUP_GUIDE.md) for detailed configuration options.

## System Requirements

- Go 1.23 or later
- PostgreSQL 16+ with NeuronDB extension
- MCP-compatible client (e.g., Claude Desktop)

## Location

**Component Directory**: [`NeuronMCP/`](../../NeuronMCP/)

## Support

- **Documentation**: [NeuronMCP/docs/](../../NeuronMCP/docs/)
- **Official Docs**: [https://www.neurondb.ai/docs/neuronmcp](https://www.neurondb.ai/docs/neuronmcp)
- **Issues**: [GitHub Issues](https://github.com/neurondb/NeurondB/issues)

