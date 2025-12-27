# NeuronAgent Component

AI agent runtime system providing REST API and WebSocket endpoints for building applications with long-term memory and tool execution.

## Overview

NeuronAgent integrates with NeuronDB PostgreSQL extension to provide agent runtime capabilities. Use it to build autonomous agent systems with persistent memory, tool execution, and streaming responses.

## Key Capabilities

| Feature | Description |
|---------|-------------|
| **Agent Runtime** | Complete state machine for autonomous task execution |
| **Long-term Memory** | HNSW-based vector search for context retrieval |
| **Tool System** | Extensible tool registry with SQL, HTTP, Code, and Shell tools |
| **REST API** | Full CRUD API for agents, sessions, and messages |
| **WebSocket Support** | Streaming agent responses in real-time |
| **Authentication** | API key-based authentication with rate limiting |
| **Background Jobs** | PostgreSQL-based job queue with worker pool |
| **NeuronDB Integration** | Direct integration with NeuronDB embedding and LLM functions |

## Documentation

### Local Documentation

- **[Component README](../../NeuronAgent/README.md)** - Overview and quick start
- **[API Reference](../../NeuronAgent/docs/API.md)** - Complete REST API documentation
- **[Architecture](../../NeuronAgent/docs/ARCHITECTURE.md)** - System design and structure
- **[Deployment Guide](../../NeuronAgent/docs/DEPLOYMENT.md)** - Production deployment
- **[Docker Guide](../../NeuronAgent/docker/README.md)** - Container deployment

### Official Documentation

- **[NeuronAgent Guide](https://www.neurondb.ai/docs/neuronagent)** - Agent runtime system
- **[API Reference](https://www.neurondb.ai/docs/neuronagent/api)** - Complete REST API documentation
- **[Architecture Guide](https://www.neurondb.ai/docs/neuronagent/architecture)** - System design
- **[Deployment Guide](https://www.neurondb.ai/docs/neuronagent/deployment)** - Production deployment
- **[Docker Guide](https://www.neurondb.ai/docs/neuronagent/docker)** - Container deployment

## Installation

### Docker (Recommended)

```bash
cd NeuronAgent/docker
docker compose up -d agent-server
```

### Source Build

```bash
cd NeuronAgent
go build ./cmd/agent-server
```

## Quick Start

### Database Setup

```bash
createdb neurondb
psql -d neurondb -c "CREATE EXTENSION neurondb;"
cd NeuronAgent
./scripts/run_migrations.sh
```

### Configuration

Set environment variables:

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=neurondb
export DB_USER=neurondb
export DB_PASSWORD=neurondb
export SERVER_PORT=8080
```

### Run Service

```bash
./agent-server
```

### Create an Agent

```bash
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "research_agent",
    "profile": "research",
    "tools": ["sql", "http"]
  }'
```

## Features

### Agent State Machine
- Autonomous task execution
- Session management
- Message handling
- Tool execution

### Long-term Memory
- HNSW vector search for context retrieval
- Persistent memory storage
- Semantic similarity search

### Tool Registry
- **SQL Tools**: Execute SQL queries
- **HTTP Tools**: Make HTTP requests
- **Code Tools**: Execute code snippets
- **Shell Tools**: Run shell commands

### REST API
- Agent management (create, list, update, delete)
- Session management
- Message handling
- Tool execution

### WebSocket Support
- Real-time streaming responses
- Live agent interactions
- Event-based communication

## Architecture

```
┌─────────────────────────────────────────────┐
│          NeuronAgent Service                │
├─────────────────────────────────────────────┤
│  REST API     │  WebSocket  │  Health      │
├─────────────────────────────────────────────┤
│  Agent State Machine │  Session Management  │
├─────────────────────────────────────────────┤
│  Tool Registry │  Memory Store │  Job Queue │
├─────────────────────────────────────────────┤
│          NeuronDB PostgreSQL                │
│  (Vector Search │  Embeddings │  LLM)       │
└─────────────────────────────────────────────┘
```

## Configuration

See [Deployment Guide](../../NeuronAgent/docs/DEPLOYMENT.md) for complete configuration options.

## System Requirements

- Go 1.23 or later
- PostgreSQL 16+ with NeuronDB extension
- Network: Port 8080 available (configurable)

## Location

**Component Directory**: [`NeuronAgent/`](../../NeuronAgent/)

## Support

- **Documentation**: [NeuronAgent/docs/](../../NeuronAgent/docs/)
- **Official Docs**: [https://www.neurondb.ai/docs/neuronagent](https://www.neurondb.ai/docs/neuronagent)
- **Issues**: [GitHub Issues](https://github.com/neurondb/NeurondB/issues)

