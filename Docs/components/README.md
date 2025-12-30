# NeuronDB Components Overview

**Comprehensive guide to all NeuronDB ecosystem components**

The NeuronDB ecosystem consists of four integrated components that work together seamlessly while maintaining independence. Each component can be deployed and scaled separately.

---

## 🎯 Component Summary

| Component | Purpose | Technology | Port | Protocol |
|-----------|---------|------------|------|----------|
| **[NeuronDB](#neurondb)** | Database extension with vector search & ML | C | 5432/5433 | PostgreSQL |
| **[NeuronAgent](#neuronagent)** | AI agent runtime with autonomous execution | Go | 8080 | REST/WebSocket |
| **[NeuronMCP](#neuronmcp)** | MCP server for AI assistants | Go | stdio | JSON-RPC 2.0 |
| **[NeuronDesktop](#neurondesktop)** | Unified web dashboard | Go + Next.js | 3000/8081 | HTTP/WebSocket |

---

## 🔧 NeuronDB

**PostgreSQL extension providing vector search, ML algorithms, and embeddings**

### Overview

NeuronDB transforms PostgreSQL into an AI-powered database with native support for vector operations, machine learning, and RAG pipelines. All functionality is accessible through SQL.

### Key Features

- **473 SQL Functions** - Complete API for vector ops, ML, RAG
- **52+ ML Algorithms** - Classification, regression, clustering, and more
- **Vector Search** - HNSW and IVF indexing with 7+ distance metrics
- **GPU Acceleration** - CUDA, ROCm, and Metal support
- **RAG Pipeline** - Document processing, embedding, retrieval
- **Background Workers** - Async jobs, auto-tuning, index maintenance

### Technology Stack

- **Language:** C
- **Platform:** PostgreSQL 16, 17, 18
- **Build:** Make, GCC/Clang
- **GPU:** CUDA 11.8+, ROCm 5.6+, Metal (macOS)
- **ML Libraries:** XGBoost, LightGBM, CatBoost (optional)

### Quick Start

```bash
# Install extension
cd NeuronDB
make install PG_CONFIG=/path/to/pg_config

# Enable in database
psql -d mydb -c "CREATE EXTENSION neurondb;"

# Test
psql -d mydb -c "SELECT neurondb.version();"
```

### Use Cases

| Use Case | Functions |
|----------|-----------|
| **Semantic Search** | `vector_search()`, `embed_text()`, HNSW indexes |
| **ML Training** | `train_xgboost()`, `train_random_forest()`, `train_kmeans()` |
| **RAG Applications** | `chunk_text()`, `retrieve_context()`, `llm_generate()` |
| **Analytics** | `detect_outliers()`, `quality_metrics()`, `detect_drift()` |

### Documentation

- **[README](../NeuronDB/README.md)** - Complete feature reference
- **[Installation Guide](../NeuronDB/INSTALL.md)** - Build and install instructions
- **[SQL API Reference](../NeuronDB/docs/sql-api.md)** - All 473 functions
- **[Full Documentation](../NeuronDB/docs/)** - 60+ doc files

### Architecture

```
NeuronDB Extension
├── Vector Operations (SIMD optimized)
│   ├── Distance calculations (L2, cosine, inner product)
│   ├── HNSW indexing (approximate nearest neighbor)
│   ├── IVF indexing (inverted file)
│   └── Quantization (PQ, OPQ, binary)
├── Machine Learning
│   ├── Built-in algorithms (40+)
│   ├── External libraries (XGBoost, LightGBM, CatBoost)
│   ├── AutoML (hyperparameter tuning)
│   └── Model management (save, load, export)
├── Embeddings & LLMs
│   ├── Text embeddings (OpenAI, HuggingFace, local)
│   ├── Image embeddings (multimodal)
│   ├── LLM integration (OpenAI, Anthropic, Ollama)
│   └── Caching layer
└── Background Workers
    ├── neuranq (async job queue)
    ├── neuranmon (live query auto-tuner)
    ├── neurandefrag (index maintenance)
    └── neuranllm (LLM job processor)
```

---

## 🤖 NeuronAgent

**REST/WebSocket agent runtime for autonomous AI agents**

### Overview

NeuronAgent provides a complete runtime environment for building autonomous AI agents with long-term memory, tool execution, and streaming responses. Perfect for building agent-based applications.

### Key Features

- **Agent State Machine** - Autonomous task execution with state management
- **Long-term Memory** - Vector-based semantic memory with retrieval
- **Tool Registry** - SQL, HTTP, code execution, shell commands
- **REST API** - Full CRUD for agents, sessions, messages
- **WebSocket Support** - Real-time streaming responses
- **Authentication** - API key-based with rate limiting
- **Background Jobs** - PostgreSQL-based job queue

### Technology Stack

- **Language:** Go 1.23+
- **Framework:** Gorilla Mux (HTTP), gorilla/websocket
- **Database:** PostgreSQL + NeuronDB
- **Protocol:** REST, WebSocket
- **Port:** 8080 (configurable)

### Quick Start

```bash
# Using Docker
cd NeuronAgent/docker
docker compose up -d

# From source
cd NeuronAgent
go run cmd/agent-server/main.go

# Test
curl http://localhost:8080/health
```

### Use Cases

| Use Case | API Endpoints |
|----------|---------------|
| **Customer Support Agents** | Agents, sessions, messages with tool access |
| **Research Assistants** | Long-term memory, SQL tools, HTTP tools |
| **Task Automation** | Tool execution, state management, job queue |
| **Chatbots** | WebSocket streaming, session management |

### Documentation

- **[README](../NeuronAgent/README.md)** - Overview and quick start
- **[API Reference](../NeuronAgent/docs/API.md)** - Complete REST API docs
- **[Architecture](../NeuronAgent/docs/ARCHITECTURE.md)** - System design
- **[Deployment Guide](../NeuronAgent/docs/DEPLOYMENT.md)** - Production deployment

### Architecture

```
NeuronAgent
├── HTTP Server
│   ├── REST API (/api/v1/*)
│   ├── WebSocket (/ws)
│   └── Health/Metrics (/health, /metrics)
├── Agent Engine
│   ├── State Machine (planning, executing, reflecting)
│   ├── Tool Executor (SQL, HTTP, code, shell)
│   ├── Memory Manager (vector-based semantic retrieval)
│   └── Session Manager (conversation state)
├── Middleware
│   ├── Authentication (API key validation)
│   ├── Rate Limiting (per-key limits)
│   ├── Logging (structured JSON logs)
│   └── Metrics (Prometheus format)
└── Database Layer
    ├── Agent storage (definitions, config)
    ├── Session storage (conversation history)
    ├── Memory storage (long-term memory with vectors)
    └── Job queue (background tasks)
```

### API Example

```bash
# Create agent
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "research-assistant",
    "system_prompt": "You are a helpful research assistant",
    "tools": ["sql", "http"],
    "config": {"temperature": 0.7}
  }'

# Send message
curl -X POST http://localhost:8080/api/v1/sessions/{id}/messages \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Find research papers about vector databases"
  }'
```

---

## 🔌 NeuronMCP

**Model Context Protocol server for MCP-compatible clients**

### Overview

NeuronMCP implements the Model Context Protocol, enabling AI assistants like Claude Desktop to access NeuronDB capabilities through a standardized protocol. Provides 100+ tools for vector operations, ML, RAG, and PostgreSQL administration.

### Key Features

- **100+ MCP Tools** - Comprehensive tool library
- **JSON-RPC 2.0** - Standard protocol implementation
- **Multiple Transports** - stdio, HTTP, Server-Sent Events
- **Vector Operations** - 50+ vector search and embedding tools
- **ML Tools** - Training, prediction, evaluation, AutoML
- **PostgreSQL Admin** - 27 database administration tools
- **Resource Management** - Schema, models, indexes

### Technology Stack

- **Language:** Go 1.23+ (server), TypeScript (client SDK)
- **Protocol:** JSON-RPC 2.0, MCP
- **Transport:** stdio, HTTP, SSE
- **Database:** PostgreSQL + NeuronDB
- **Clients:** Claude Desktop, custom MCP clients

### Quick Start

```bash
# Using Docker
cd NeuronMCP/docker
docker compose up -d

# From source
cd NeuronMCP
go build ./cmd/neurondb-mcp
./neurondb-mcp

# Configure Claude Desktop
# Edit: ~/.config/Claude/claude_desktop_config.json
```

### Use Cases

| Use Case | Tools |
|----------|-------|
| **AI Assistant Integration** | All 100+ tools via MCP protocol |
| **Database Administration** | 27 PostgreSQL admin tools |
| **Vector Search** | Search, embeddings, indexing tools |
| **ML Workflows** | Training, prediction, evaluation tools |

### Documentation

- **[README](../NeuronMCP/README.md)** - Overview and usage
- **[Tools Reference](../NeuronMCP/TOOLS_REFERENCE.md)** - All 100+ tools documented
- **[PostgreSQL Tools](../NeuronMCP/POSTGRESQL_TOOLS.md)** - Database admin tools
- **[Setup Guide](../NeuronMCP/docs/NEURONDB_MCP_SETUP.md)** - Complete setup instructions

### Architecture

```
NeuronMCP
├── Protocol Layer
│   ├── JSON-RPC 2.0 handler
│   ├── MCP protocol (2024-11-05)
│   ├── Transport (stdio/HTTP/SSE)
│   └── Request/response validation
├── Tool Registry (100+ tools)
│   ├── Vector Operations (50+)
│   │   ├── Search (8 distance metrics)
│   │   ├── Embeddings (batch, cached, multimodal)
│   │   ├── Quantization (6 types)
│   │   └── Hybrid search (7 variants)
│   ├── ML Tools (30+)
│   │   ├── Training (15 algorithms)
│   │   ├── Prediction (batch, streaming)
│   │   ├── Evaluation (metrics)
│   │   └── AutoML
│   ├── PostgreSQL Admin (27)
│   │   ├── Database ops (8)
│   │   ├── Performance (9)
│   │   ├── Security (5)
│   │   └── Monitoring (5)
│   └── RAG & Analytics (13+)
├── Resource Manager
│   ├── Schema introspection
│   ├── Model catalog
│   ├── Index management
│   └── Configuration
└── Middleware Stack
    ├── Authentication (JWT, API keys, OAuth2)
    ├── Rate Limiting (per-client)
    ├── Validation (input/output)
    ├── Logging (structured)
    └── Metrics (Prometheus)
```

### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "neurondb": {
      "command": "docker",
      "args": [
        "exec", "-i", "neurondb-mcp",
        "/app/neurondb-mcp"
      ],
      "env": {
        "NEURONDB_HOST": "localhost",
        "NEURONDB_PORT": "5433",
        "NEURONDB_DATABASE": "neurondb"
      }
    }
  }
}
```

---

## 🖥️ NeuronDesktop

**Unified web interface for managing the entire ecosystem**

### Overview

NeuronDesktop provides a professional web-based dashboard for managing NeuronDB, NeuronAgent, and NeuronMCP. Includes SQL console, agent management, MCP integration, and real-time monitoring.

### Key Features

- **Unified Dashboard** - Single interface for all components
- **Real-time Updates** - WebSocket for live data
- **SQL Console** - Interactive SQL query interface
- **Agent Management** - Create and manage AI agents through UI
- **MCP Integration** - Full MCP server integration and testing
- **Metrics & Monitoring** - Built-in metrics collection and visualization
- **Authentication** - Secure API key-based authentication
- **Professional UI** - Modern, responsive design with Tailwind CSS

### Technology Stack

- **Backend:** Go 1.23+
- **Frontend:** Next.js 14+, React 18+, TypeScript
- **Styling:** Tailwind CSS
- **Database:** PostgreSQL
- **Ports:** 3000 (frontend), 8081 (API)

### Quick Start

```bash
# Using Docker (from repository root)
docker compose --profile default up -d

# Access web interface
open http://localhost:3000

# API endpoint
curl http://localhost:8081/health
```

### Use Cases

| Use Case | Features |
|----------|----------|
| **Database Management** | SQL console, schema browser, query history |
| **Agent Development** | Agent creation, testing, session management |
| **MCP Testing** | Tool testing, resource inspection, prompt testing |
| **Monitoring** | Metrics dashboard, health checks, logs |

### Documentation

- **[README](../NeuronDesktop/README.md)** - Overview and features
- **[API Reference](../NeuronDesktop/docs/API.md)** - Complete API documentation
- **[Integration Guide](../NeuronDesktop/docs/INTEGRATION.md)** - Component integration
- **[Deployment Guide](../NeuronDesktop/docs/DEPLOYMENT.md)** - Production deployment

### Architecture

```
NeuronDesktop
├── Frontend (Port 3000)
│   ├── Next.js App Router
│   ├── React Components
│   │   ├── Dashboard
│   │   ├── SQL Console
│   │   ├── Agent Manager
│   │   ├── MCP Explorer
│   │   └── Metrics Dashboard
│   ├── State Management (React hooks)
│   └── API Client (fetch/WebSocket)
├── Backend API (Port 8081)
│   ├── HTTP Server (Gorilla Mux)
│   ├── WebSocket Server
│   ├── Authentication Middleware
│   └── Request/Response Logging
├── Integration Layer
│   ├── NeuronDB Client (PostgreSQL driver)
│   ├── NeuronAgent Client (HTTP client)
│   ├── NeuronMCP Proxy (stdio bridge)
│   └── Metrics Collector
└── Database Layer
    ├── Profile Management
    ├── Configuration Storage
    ├── Query History
    └── User Preferences
```

---

## 🔄 Component Interactions

### Communication Patterns

```
┌─────────────────────────────────────────────────────┐
│                 NeuronDesktop UI                     │
│              (Browser, Port 3000)                    │
└─────────────┬───────────────────────────────────────┘
              │ HTTP/WebSocket
┌─────────────▼───────────────────────────────────────┐
│          NeuronDesktop API (Port 8081)              │
├─────────────┬──────────────────┬─────────────────── │
│   MCP Proxy │  Agent Client    │  DB Client         │
└─────────┬───┴────────┬─────────┴──────┬─────────────┘
          │            │                │
          │ stdio      │ HTTP           │ PostgreSQL
          │            │                │
┌─────────▼──────┐ ┌──▼────────┐  ┌────▼──────┐
│  NeuronMCP     │ │NeuronAgent│  │ NeuronDB  │
│  (stdio)       │ │ (8080)    │  │(5432/5433)│
└─────────┬──────┘ └──┬────────┘  └────┬──────┘
          │            │                │
          └────────────┴────────────────┘
                       │
              All access PostgreSQL
```

### Data Flow Examples

**1. Vector Search via Agent:**
```
Client → NeuronAgent → NeuronDB
       (REST API)    (SQL query)
       ← Results ← (vector search)
```

**2. MCP Tool Call:**
```
Claude → NeuronMCP → NeuronDB
      (JSON-RPC)   (SQL exec)
      ← Result ← (query result)
```

**3. Web UI Query:**
```
Browser → Desktop API → NeuronDB
       (WebSocket)    (SQL)
       ← Stream ← (results)
```

---

## 🚀 Getting Started with Components

### Choose Your Starting Point

**For Database-Only Use:**
- Install **NeuronDB** extension only
- Use SQL directly from any PostgreSQL client
- Best for: embedding NeuronDB in existing applications

**For Agent Development:**
- Install **NeuronDB** + **NeuronAgent**
- Build autonomous agents with REST API
- Best for: building agent-based applications

**For AI Assistant Integration:**
- Install **NeuronDB** + **NeuronMCP**
- Connect Claude Desktop or other MCP clients
- Best for: augmenting AI assistants with database capabilities

**For Complete Solution:**
- Install all four components
- Use **NeuronDesktop** as unified interface
- Best for: comprehensive AI application development

---

## 📊 Feature Comparison

| Feature | NeuronDB | NeuronAgent | NeuronMCP | NeuronDesktop |
|---------|----------|-------------|-----------|---------------|
| **Vector Search** | ✅ Native | ✅ Via DB | ✅ Via DB | ✅ Via DB |
| **ML Training** | ✅ 52+ algorithms | ✅ Via DB | ✅ Via DB | ✅ Via DB |
| **Embeddings** | ✅ Native | ✅ Via DB | ✅ Via DB | ✅ Via DB |
| **Autonomous Agents** | ❌ | ✅ Core feature | ❌ | ✅ Management |
| **MCP Protocol** | ❌ | ❌ | ✅ Core feature | ✅ Integration |
| **Web Interface** | ❌ | ❌ | ❌ | ✅ Core feature |
| **REST API** | ❌ | ✅ Core feature | ⚠️ Optional | ✅ Core feature |
| **WebSocket** | ❌ | ✅ Streaming | ⚠️ Optional | ✅ Real-time |
| **SQL Access** | ✅ Direct | ✅ Tool | ✅ Tool | ✅ Console |

---

## 🛠️ Deployment Configurations

### Minimal (Database Only)

```yaml
services:
  neurondb:
    image: neurondb:cpu-pg17
    ports: ["5433:5432"]
```

### Standard (Database + Agent)

```yaml
services:
  neurondb:
    image: neurondb:cpu-pg17
  neuronagent:
    image: neuronagent:latest
    ports: ["8080:8080"]
```

### Full Stack (All Components)

```yaml
services:
  neurondb:
    image: neurondb:cpu-pg17
  neuronagent:
    image: neuronagent:latest
  neurondb-mcp:
    image: neurondb-mcp:latest
  neurondesk-api:
    image: neurondesk-api:latest
  neurondesk-frontend:
    image: neurondesk-frontend:latest
```

---

## 📚 Additional Resources

- **[Main README](../README.md)** - Project overview
- **[Quick Start](../QUICKSTART.md)** - Get started in minutes
- **[Documentation Index](../DOCUMENTATION.md)** - Complete documentation reference
- **[Project Overview](../PROJECT_OVERVIEW.md)** - Technical architecture
- **[Examples](../examples/)** - Code examples and tutorials

---

**Last Updated:** 2025-01-30  
**Components Version:** 1.0.0
