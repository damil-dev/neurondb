# NeuronDB Ecosystem

<div align="center">
  <img src="neurondb.png" alt="NeuronDB Logo" width="200">
</div>

[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%2C17%2C18-blue.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://www.neurondb.ai/docs)
[![Website](https://img.shields.io/badge/website-www.neurondb.ai-blue.svg)](https://www.neurondb.ai/)

> **📚 For comprehensive documentation, tutorials, API references, and best practices, visit [https://www.neurondb.ai/docs](https://www.neurondb.ai/docs)**

PostgreSQL extension with vector search, machine learning algorithms, and agent runtime capabilities. Three components operate independently while sharing the same database instance.

## Table of Contents

- [Overview](#overview)
- [Components](#components)
  - [NeuronDB](#neurondb)
  - [NeuronAgent](#neuronagent)
  - [NeuronMCP](#neuronmcp)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Service Management](#service-management)
- [Documentation](#documentation)
- [System Requirements](#system-requirements)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting](#troubleshooting)
- [Security](#security)
- [Performance](#performance)
- [Support](#support)

## Overview

NeuronDB provides vector search and machine learning within PostgreSQL. NeuronAgent adds REST API and WebSocket support for agent applications. NeuronMCP exposes NeuronDB through the Model Context Protocol for MCP-compatible clients.

Components connect via database connection strings. Each service configures independently. Services operate without requiring others to run.

### Component Communication Matrix

| Component | Connection Method | Protocol | Port | Purpose |
|-----------|------------------|----------|------|---------|
| **NeuronDB** | PostgreSQL native | TCP | 5432/5433 | Database server with extension |
| **NeuronAgent** | Database connection | TCP | 5432/5433 | Agent runtime data access |
| **NeuronAgent** | HTTP REST API | HTTP | 8080 | Client API access |
| **NeuronAgent** | WebSocket | WS | 8080 | Streaming responses |
| **NeuronMCP** | Database connection | TCP | 5432/5433 | MCP server data access |
| **NeuronMCP** | Stdio | JSON-RPC 2.0 | - | MCP client communication |

### Data Flow

1. **Client Requests** → NeuronAgent (HTTP/WebSocket) or NeuronMCP (stdio)
2. **Service Processing** → Agent runtime or MCP protocol handler
3. **Database Queries** → NeuronDB PostgreSQL extension
4. **Vector/ML Operations** → Extension executes vector search, ML algorithms
5. **Results Return** → Through service layer back to clients

### Key Features by Layer

**Client Layer:**
- Web applications via REST API
- Mobile apps via REST/WebSocket
- CLI tools via REST API
- MCP clients (Claude Desktop) via stdio

**Service Layer:**
- NeuronAgent: Agent state machine, tool execution, memory management
- NeuronMCP: MCP protocol, tool/resource handlers, middleware

**Database Layer:**
- Vector search with HNSW/IVF indexing
- 52+ ML algorithms (classification, regression, clustering, etc.)
- Embedding generation (text, image, multimodal)
- Hybrid search combining vector and full-text
- RAG pipeline with LLM integration
- GPU acceleration (CUDA, ROCm, Metal)
- Background workers for async operations

All components access the same PostgreSQL database instance. Services operate independently and can run separately.

## Components

### NeuronDB

PostgreSQL extension adding vector search, machine learning, and embedding generation.

#### Capabilities

| Feature | Description |
|---------|-------------|
| Vector Search | HNSW and IVF indexing for similarity search |
| Machine Learning | Classification, regression, clustering algorithms |
| GPU Acceleration | CUDA, ROCm, and Metal support |
| Hybrid Search | Vector and full-text search combination |
| RAG Pipeline | Document retrieval and context generation |
| Embeddings | Text, image, and multimodal embedding generation |

#### Documentation

| Document | Description |
|----------|-------------|
| [Component README](NeuronDB/README.md) | Complete feature documentation |
| [Installation Guide](NeuronDB/INSTALL.md) | Build and installation instructions |
| [Docker Guide](NeuronDB/docker/README.md) | Container deployment |
| [SQL API Reference](NeuronDB/docs/sql-api.md) | Function reference |

**Location:** [`NeuronDB/`](NeuronDB/)

### NeuronAgent

Agent runtime system providing REST API and WebSocket endpoints for building applications with memory and tool execution.

#### Capabilities

| Feature | Description |
|---------|-------------|
| Agent State Machine | Autonomous task execution |
| Long-term Memory | HNSW vector search for context retrieval |
| Tool Registry | SQL, HTTP, code execution, shell commands |
| REST API | Agent, session, and message management |
| WebSocket | Streaming agent responses |
| Authentication | API key with rate limiting |
| Background Jobs | PostgreSQL-based job queue |

#### Documentation

| Document | Description |
|----------|-------------|
| [Component README](NeuronAgent/README.md) | Overview and quick start |
| [API Reference](NeuronAgent/docs/API.md) | Complete REST API documentation |
| [Architecture](NeuronAgent/docs/ARCHITECTURE.md) | System design and structure |
| [Deployment Guide](NeuronAgent/docs/DEPLOYMENT.md) | Production deployment |
| [Docker Guide](NeuronAgent/docker/README.md) | Container deployment |

**Location:** [`NeuronAgent/`](NeuronAgent/)

### NeuronMCP

Model Context Protocol server enabling MCP clients to access NeuronDB through stdio communication.

#### Capabilities

| Feature | Description |
|---------|-------------|
| MCP Protocol | JSON-RPC 2.0 implementation |
| Stdio Transport | Standard input/output communication |
| Vector Operations | Search, embeddings, indexing |
| ML Tools | Training and prediction |
| Resource Management | Schema, models, indexes |
| Middleware | Validation, logging, timeouts |

#### Documentation

| Document | Description |
|----------|-------------|
| [Component README](NeuronMCP/README.md) | Overview and usage |
| [Docker Guide](NeuronMCP/docker/README.md) | Container deployment |

**Location:** [`NeuronMCP/`](NeuronMCP/)

## Quick Start

### Prerequisites

Install these components before starting:

1. **PostgreSQL 16, 17, or 18**
2. **Docker and Docker Compose**
3. **Go 1.23+** (for building from source)

See [NeuronDB installation guide](NeuronDB/INSTALL.md) for platform-specific requirements.

### Unified Setup (Recommended)

For a complete integrated setup of all three modules, use the unified setup script:

```bash
# Set database connection parameters (optional - defaults shown)
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=neurondb
export DB_USER=postgres
export DB_PASSWORD=your_password

# Run unified setup script
./scripts/setup_neurondb_ecosystem.sh
```

This script will:
1. Create the database if it doesn't exist
2. Install the NeuronDB extension
3. Set up NeuronMCP schema and functions
4. Run NeuronAgent migrations
5. Verify the installation

**Verify Integration:**

After setup, verify all components are properly integrated:

```bash
./scripts/verify_neurondb_integration.sh
```

This will test:
- NeuronDB extension functionality
- NeuronMCP schema and functions
- NeuronAgent schema and tables
- Cross-module integration
- Vector operations

### Installation Steps

#### Step 1: Start NeuronDB

```bash
cd NeuronDB/docker
docker compose up -d neurondb
```

Verify container status:

```bash
docker compose ps neurondb
```

Test database connection:

```bash
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" \
  -c "SELECT neurondb.version();"
```

#### Step 2: Start NeuronAgent

Configure environment (optional - docker-compose.yml has defaults):

```bash
cd ../../NeuronAgent/docker
# Optionally create .env file, or use environment variables directly
```

Example `.env` file (if creating one):

```env
DB_HOST=localhost
DB_PORT=5433
DB_NAME=neurondb
DB_USER=neurondb
DB_PASSWORD=neurondb
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
```

Start service:

```bash
docker compose build
docker compose up -d agent-server
```

Verify API:

```bash
curl http://localhost:8080/health
```

#### Step 3: Start NeuronMCP

Configure environment (optional - docker-compose.yml has defaults):

```bash
cd ../../NeuronMCP/docker
# Optionally create .env file, or use environment variables directly
```

Example `.env` file (if creating one):

```env
NEURONDB_HOST=localhost
NEURONDB_PORT=5433
NEURONDB_DATABASE=neurondb
NEURONDB_USER=neurondb
NEURONDB_PASSWORD=neurondb
```

Start service:

```bash
docker compose build
docker compose up -d neurondb-mcp
```

## Installation

### Database Setup and Integration

Before starting services, ensure the database is properly set up with all required schemas and extensions.

#### Option 1: Unified Setup Script (Recommended)

The unified setup script ensures proper execution order and handles all dependencies:

```bash
# Run from project root
./scripts/setup_neurondb_ecosystem.sh
```

**What it does:**
- Creates database if needed
- Installs NeuronDB extension
- Sets up NeuronMCP schema (13 tables, 30+ functions)
- Runs NeuronAgent migrations (4 migration files)
- Verifies installation

**Environment Variables:**
```bash
export DB_HOST=localhost          # Default: localhost
export DB_PORT=5432              # Default: 5432
export DB_NAME=neurondb          # Default: neurondb
export DB_USER=postgres          # Default: postgres
export DB_PASSWORD=your_password # Optional
```

#### Option 2: Manual Setup

If you prefer to set up components individually:

1. **Create database and install NeuronDB extension:**
   ```bash
   createdb neurondb
   psql -d neurondb -c "CREATE EXTENSION neurondb;"
   ```

2. **Setup NeuronMCP:**
   ```bash
   cd NeuronMCP
   ./scripts/setup_neurondb_mcp.sh
   ```

3. **Setup NeuronAgent:**
   ```bash
   cd NeuronAgent
   ./scripts/setup_neurondb_agent.sh
   ./scripts/run_migrations.sh
   ```

#### Verification

After setup, verify integration:

```bash
./scripts/verify_neurondb_integration.sh
```

This comprehensive test verifies:
- ✓ NeuronDB extension installed and functional
- ✓ NeuronMCP schema, tables, views, and functions
- ✓ NeuronAgent schema, tables, and vector columns
- ✓ HNSW indexes on vector columns
- ✓ Cross-module queries and operations
- ✓ Vector operations with different dimensions

### Installation Methods

| Method | Use Case | Requirements |
|--------|----------|--------------|
| Docker | Recommended for all deployments | Docker, Docker Compose |
| Source Build | Development or custom builds | PostgreSQL dev headers, C compiler |
| Package Manager | System integration | Platform-specific packages |

### Docker Installation

Docker installation provides isolation and consistent environments.

```bash
# NeuronDB
cd NeuronDB/docker
docker compose up -d neurondb

# NeuronAgent
cd ../../NeuronAgent/docker
docker compose up -d agent-server

# NeuronMCP
cd ../../NeuronMCP/docker
docker compose up -d neurondb-mcp
```

### Source Installation

Build from source for development or custom configurations.

```bash
# Install NeuronDB extension
cd NeuronDB
make install PG_CONFIG=/usr/local/pgsql/bin/pg_config

# Build NeuronAgent
cd ../NeuronAgent
go build ./cmd/agent-server

# Build NeuronMCP
cd ../NeuronMCP
go build ./cmd/neurondb-mcp
```

See [NeuronDB installation guide](NeuronDB/INSTALL.md) for detailed build instructions.

## Configuration

### Database Connection Parameters

All services connect to the same NeuronDB PostgreSQL instance.

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| Host | `localhost` | Database hostname or IP address |
| Port | `5432` (direct) / `5433` (Docker) | Database port number |
| Database | `neurondb` | Database name |
| User | `neurondb` | Database username |
| Password | `neurondb` | Database password |

### Environment Variables

#### NeuronAgent Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | `localhost` | Database hostname |
| `DB_PORT` | `5432` | Database port number |
| `DB_NAME` | `neurondb` | Database name |
| `DB_USER` | `neurondb` | Database username |
| `DB_PASSWORD` | `neurondb` | Database password |
| `SERVER_HOST` | `0.0.0.0` | API server host |
| `SERVER_PORT` | `8080` | API server port |
| `CONFIG_PATH` | - | Path to config.yaml file |

#### NeuronMCP Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEURONDB_HOST` | `localhost` | Database hostname |
| `NEURONDB_PORT` | `5432` | Database port number |
| `NEURONDB_DATABASE` | `neurondb` | Database name |
| `NEURONDB_USER` | `neurondb` | Database username |
| `NEURONDB_PASSWORD` | `neurondb` | Database password |
| `NEURONDB_MCP_CONFIG` | - | Path to mcp-config.json |

### Network Configuration

#### Option 1: Local Development

All services run on the same host.

- Use `localhost` as hostname
- Ports: 5433 (Docker) or 5432 (direct)
- No network configuration required

#### Option 2: Docker Network

Create shared network for container communication:

```bash
# Create network
docker network create neurondb-network

# Connect containers
docker network connect neurondb-network neurondb-cpu
docker network connect neurondb-network neuronagent
docker network connect neurondb-network neurondb-mcp
```

Use container names as hostnames in configuration.

#### Option 3: Production Deployment

Services run on separate hosts.

- Configure explicit hostnames in environment variables
- Ensure network connectivity between hosts
- Configure firewall rules for database port
- Use DNS names or IP addresses

For detailed network setup, see [ecosystem Docker guide](NeuronDB/docker/ECOSYSTEM.md#network-configuration).

## Usage Examples

### Vector Search Applications

Use NeuronDB for semantic search and similarity matching.

**Example: Document Search**

```sql
-- Create vector column
ALTER TABLE documents ADD COLUMN embedding vector(768);

-- Generate embeddings
UPDATE documents SET embedding = neurondb.embed_text(content, 'model_name');

-- Semantic search
SELECT id, content, embedding <=> query_embedding AS distance
FROM documents
ORDER BY embedding <=> query_embedding
LIMIT 10;
```

**Example: Product Recommendations**

```sql
-- Find similar products
SELECT product_id, name, product_embedding <=> user_embedding AS similarity
FROM products
WHERE category = 'electronics'
ORDER BY product_embedding <=> user_embedding
LIMIT 5;
```

### Retrieval-Augmented Generation

Build RAG pipelines with NeuronDB and NeuronAgent.

**Workflow:**

1. Ingest documents into PostgreSQL
2. Generate embeddings using NeuronDB functions
3. Store embeddings in vector columns
4. Retrieve context using vector search
5. Pass context to LLM through NeuronAgent

**Example:**

```sql
-- Store document with embedding
INSERT INTO documents (content, embedding)
VALUES ('Document text', neurondb.embed_text('Document text', 'model'));

-- Retrieve relevant context
SELECT content FROM documents
ORDER BY embedding <=> query_embedding
LIMIT 3;
```

### Agent Applications

Use NeuronAgent for autonomous agent systems.

**Example: Create Agent**

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

**Example: Send Message**

```bash
curl -X POST http://localhost:8080/api/v1/sessions/SESSION_ID/messages \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Find documents about machine learning"
  }'
```

### MCP Client Integration

Use NeuronMCP to connect MCP-compatible clients.

**Claude Desktop Configuration:**

```json
{
  "mcpServers": {
    "neurondb": {
      "command": "docker",
      "args": ["exec", "-i", "neurondb-mcp", "./neurondb-mcp"],
      "env": {
        "NEURONDB_HOST": "localhost",
        "NEURONDB_PORT": "5433",
        "NEURONDB_DATABASE": "neurondb",
        "NEURONDB_USER": "neurondb",
        "NEURONDB_PASSWORD": "neurondb"
      }
    }
  }
}
```

## Service Management

### Starting Services

Start services independently in any order:

```bash
# Start NeuronDB
cd NeuronDB/docker
docker compose up -d neurondb

# Start NeuronAgent
cd ../../NeuronAgent/docker
docker compose up -d agent-server

# Start NeuronMCP
cd ../../NeuronMCP/docker
docker compose up -d neurondb-mcp
```

### Stopping Services

Stop services without affecting others:

```bash
docker compose stop neurondb
docker compose stop agent-server
docker compose stop neurondb-mcp
```

### Health Checks

Verify service status:

```bash
# NeuronDB
docker compose ps neurondb

# NeuronAgent
curl http://localhost:8080/health

# NeuronMCP
docker compose ps neurondb-mcp
```

### Viewing Logs

Access service logs:

```bash
# Follow logs
docker compose logs -f neurondb
docker compose logs -f agent-server
docker compose logs -f neurondb-mcp

# View last 100 lines
docker compose logs --tail=100 neurondb
```

### Restarting Services

Restart individual services:

```bash
docker compose restart neurondb
docker compose restart agent-server
docker compose restart neurondb-mcp
```

## Documentation

### Official Documentation

**For comprehensive documentation, tutorials, API references, and best practices, visit:**

🌐 **[https://www.neurondb.ai/docs](https://www.neurondb.ai/docs)**

The official documentation site provides:
- Complete API reference for all 473 SQL functions
- Detailed tutorials and guides
- Performance optimization guides
- Production deployment best practices
- Troubleshooting and FAQ
- Latest updates and release notes

**📖 See [DOCUMENTATION.md](DOCUMENTATION.md) for a complete documentation index with quick links to all topics.**

### Component Documentation

#### NeuronDB

| Document | Purpose | Detailed Docs |
|----------|---------|---------------|
| [Main Documentation](NeuronDB/README.md) | Complete feature reference | [Vector Search](https://www.neurondb.ai/docs/vector-search) |
| [Installation Guide](NeuronDB/INSTALL.md) | Build and install instructions | [Installation Guide](https://www.neurondb.ai/docs/installation) |
| [Docker Guide](NeuronDB/docker/README.md) | Container deployment | [Docker Deployment](https://www.neurondb.ai/docs/docker) |
| [SQL API Reference](NeuronDB/docs/sql-api.md) | Function documentation | [Complete API Reference](https://www.neurondb.ai/docs/api) |
| [Vector Search](NeuronDB/docs/vector-search/) | Indexing and search guide | [Vector Search Guide](https://www.neurondb.ai/docs/vector-search) |
| [ML Algorithms](NeuronDB/docs/ml-algorithms/) | Machine learning features | [ML Algorithms](https://www.neurondb.ai/docs/ml-algorithms) |
| [RAG Pipeline](NeuronDB/docs/rag/) | Retrieval-augmented generation | [RAG Pipeline](https://www.neurondb.ai/docs/rag) |

#### NeuronAgent

| Document | Purpose | Detailed Docs |
|----------|---------|---------------|
| [Main Documentation](NeuronAgent/README.md) | Overview and features | [NeuronAgent Guide](https://www.neurondb.ai/docs/neuronagent) |
| [API Reference](NeuronAgent/docs/API.md) | Complete REST API docs | [API Reference](https://www.neurondb.ai/docs/neuronagent/api) |
| [Architecture](NeuronAgent/docs/ARCHITECTURE.md) | System design | [Architecture Guide](https://www.neurondb.ai/docs/neuronagent/architecture) |
| [Deployment Guide](NeuronAgent/docs/DEPLOYMENT.md) | Production setup | [Deployment Guide](https://www.neurondb.ai/docs/neuronagent/deployment) |
| [Docker Guide](NeuronAgent/docker/README.md) | Container deployment | [Docker Guide](https://www.neurondb.ai/docs/neuronagent/docker) |

#### NeuronMCP

| Document | Purpose | Detailed Docs |
|----------|---------|---------------|
| [Main Documentation](NeuronMCP/README.md) | Overview and usage | [NeuronMCP Guide](https://www.neurondb.ai/docs/neuronmcp) |
| [Docker Guide](NeuronMCP/docker/README.md) | Container deployment | [Docker Guide](https://www.neurondb.ai/docs/neuronmcp/docker) |

### Ecosystem Documentation

| Document | Purpose | Detailed Docs |
|----------|---------|---------------|
| [Ecosystem Docker Guide](NeuronDB/docker/ECOSYSTEM.md) | Running all services together | [Ecosystem Guide](https://www.neurondb.ai/docs/ecosystem) |
| [Configuration Reference](NeuronDB/docker/ECOSYSTEM.md#configuration) | Service configuration | [Configuration](https://www.neurondb.ai/docs/configuration) |
| [Network Setup](NeuronDB/docker/ECOSYSTEM.md#network-configuration) | Networking options | [Network Setup](https://www.neurondb.ai/docs/network) |
| [Troubleshooting](NeuronDB/docker/ECOSYSTEM.md#troubleshooting) | Common issues | [Troubleshooting](https://www.neurondb.ai/docs/troubleshooting) |

## System Requirements

### NeuronDB

| Component | Requirement |
|-----------|-------------|
| PostgreSQL | 16, 17, or 18 |
| Build Tools | C compiler (GCC or Clang), Make |
| Optional GPU | CUDA (NVIDIA), ROCm (AMD), Metal (macOS) |

See [installation guide](NeuronDB/INSTALL.md) for platform-specific requirements.

### NeuronAgent

| Component | Requirement |
|-----------|-------------|
| Go | 1.23 or later |
| Database | PostgreSQL 16+ with NeuronDB extension |
| Network | Port 8080 available (configurable) |

### NeuronMCP

| Component | Requirement |
|-----------|-------------|
| Go | 1.23 or later |
| Database | PostgreSQL 16+ with NeuronDB extension |
| Client | MCP-compatible client (e.g., Claude Desktop) |

## Docker Deployment

### Available Docker Images

| Component | Variants | Description |
|-----------|----------|-------------|
| NeuronDB | CPU, CUDA, ROCm, Metal | PostgreSQL with NeuronDB extension |
| NeuronAgent | CPU | Standalone service container |
| NeuronMCP | CPU | Standalone service container |

### Docker Configuration

Each component includes Docker configurations:

- `Dockerfile` - Multi-stage build for optimized images
- `docker-compose.yml` - Service definition and networking with default environment variables
- `.dockerignore` - Exclude unnecessary files from builds
- Optional `.env` file - Override environment variables (create manually if needed)

### Deployment Options

| Option | Description | Use Case |
|--------|-------------|----------|
| Individual Containers | Run each service separately | Production deployment |
| Docker Compose | Orchestrate all services | Development and testing |
| Kubernetes | Container orchestration | Large-scale production |

See component-specific Docker guides:
- [NeuronDB Docker](NeuronDB/docker/README.md)
- [NeuronAgent Docker](NeuronAgent/docker/README.md)
- [NeuronMCP Docker](NeuronMCP/docker/README.md)

## Troubleshooting

### Common Issues

#### Connection Problems

**Symptom:** Services cannot connect to NeuronDB

**Solutions:**

1. Verify NeuronDB container status:
   ```bash
   docker compose ps neurondb
   ```

2. Check database connection parameters:
   ```bash
   echo $DB_HOST $DB_PORT $DB_NAME
   ```

3. Test network connectivity:
   ```bash
   docker exec neuronagent ping neurondb-cpu
   ```

4. Verify firewall rules:
   ```bash
   sudo ufw status
   ```

#### Service Startup Issues

**Symptom:** Service fails to start

**Solutions:**

1. Check service logs:
   ```bash
   docker compose logs agent-server
   ```

2. Verify environment variables:
   ```bash
   docker compose config
   ```

3. Check database migrations:
   ```bash
   psql -d neurondb -c "\dt" | grep agent
   ```

4. Verify port availability:
   ```bash
   netstat -tulpn | grep 8080
   ```

#### Performance Issues

**Symptom:** Slow query performance

**Solutions:**

1. Verify indexes exist:
   ```sql
   SELECT indexname FROM pg_indexes WHERE tablename = 'your_table';
   ```

2. Check GPU acceleration:
   ```sql
   SELECT neurondb.gpu_enabled();
   ```

3. Review connection pool settings:
   ```bash
   grep DB_MAX_OPEN_CONNS .env
   ```

4. Monitor query performance:
   ```sql
   SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;
   ```

### Detailed Troubleshooting

- [NeuronDB Troubleshooting](NeuronDB/docs/troubleshooting.md)
- [Ecosystem Troubleshooting](NeuronDB/docker/ECOSYSTEM.md#troubleshooting)

## Security

### Security Practices

| Practice | Implementation |
|----------|----------------|
| Change Default Passwords | Update database passwords in production |
| Use Secrets Management | Store credentials in environment variables or secrets |
| Enable SSL/TLS | Configure encrypted database connections |
| API Key Management | Implement proper key rotation for NeuronAgent |
| Network Restrictions | Use Docker networks or firewalls to limit access |
| Authentication | Configure strong authentication methods |
| Authorization | Implement role-based access control |

### Configuration Checklist

- [ ] Changed default database password
- [ ] Enabled SSL/TLS for database connections
- [ ] Configured API key authentication for NeuronAgent
- [ ] Set up firewall rules
- [ ] Restricted network access
- [ ] Implemented secrets management
- [ ] Enabled audit logging

## Performance

### Optimization Strategies

| Strategy | Implementation |
|----------|----------------|
| Create Indexes | Add indexes on vector columns for search operations |
| Tune Connection Pools | Configure connection pool settings per service |
| Enable GPU Acceleration | Use GPU hardware when available |
| Monitor Performance | Track query performance and optimize |
| Connection Pooling | Use effective connection pooling strategies |

### Performance Monitoring

Monitor these metrics:

- Database query execution time
- Vector search latency
- Connection pool utilization
- GPU utilization (if enabled)
- API response times
- Memory usage

### Tuning Parameters

**Database:**

- `shared_buffers` - Increase for better cache performance
- `work_mem` - Adjust for complex queries
- `max_connections` - Set based on workload

**NeuronAgent:**

- `DB_MAX_OPEN_CONNS` - Connection pool size
- `DB_MAX_IDLE_CONNS` - Idle connections
- `DB_CONN_MAX_LIFETIME` - Connection lifetime

## Deployment

### Deployment Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Docker Compose | Single host deployment | Development, small production |
| Individual Containers | Separate container deployment | Medium-scale production |
| Kubernetes | Container orchestration | Large-scale production |
| Bare Metal | Direct installation | High-performance requirements |

### Deployment Checklist

- [ ] Configured environment variables
- [ ] Set up database backups
- [ ] Implemented health checks
- [ ] Configured monitoring
- [ ] Set up log aggregation
- [ ] Configured auto-scaling (if needed)
- [ ] Tested disaster recovery

### Production Guidelines

See component-specific deployment documentation:

- [NeuronDB Installation](NeuronDB/INSTALL.md)
- [NeuronAgent Deployment](NeuronAgent/docs/DEPLOYMENT.md)
- [Ecosystem Docker Guide](NeuronDB/docker/ECOSYSTEM.md)

## Support

### Getting Help

| Resource | Purpose |
|----------|---------|
| [Official Documentation](https://www.neurondb.ai/docs) | Complete documentation, tutorials, and guides |
| [GitHub Issues](https://github.com/neurondb/NeurondB/issues) | Report bugs and request features |
| [Website](https://www.neurondb.ai/) | Product information and resources |
| Email Support | support@neurondb.ai |

### Reporting Issues

Include this information when reporting issues:

1. Component version
2. PostgreSQL version
3. Operating system
4. Error messages
5. Steps to reproduce
6. Configuration details

## License

This software is licensed under a proprietary Software License Agreement with
strict commercial use prohibitions.

**Key Terms:**
- ✓ Personal use of binaries ONLY is permitted
- ✗ Commercial use is STRICTLY PROHIBITED
- ✗ Creating companies based on this code is STRICTLY PROHIBITED
- ✗ Using source code for commercial purposes is STRICTLY PROHIBITED

These terms apply RETROACTIVELY to ALL versions from 2024 onwards.

See [LICENSE](LICENSE) file for complete license terms.

## Quick Reference

### Getting Started Links

| Link | Description |
|------|-------------|
| [NeuronDB Installation](NeuronDB/INSTALL.md) | Install NeuronDB extension |
| [NeuronDB Quick Start](NeuronDB/docs/getting-started/quickstart.md) | Get started quickly |
| [Ecosystem Docker Setup](NeuronDB/docker/ECOSYSTEM.md) | Docker deployment guide |

### Component Links

| Component | Documentation |
|-----------|---------------|
| [NeuronDB](NeuronDB/README.md) | PostgreSQL extension documentation |
| [NeuronAgent](NeuronAgent/README.md) | Agent runtime documentation |
| [NeuronMCP](NeuronMCP/README.md) | MCP server documentation |

### Docker Links

| Component | Docker Guide |
|-----------|--------------|
| [NeuronDB Docker](NeuronDB/docker/README.md) | Container deployment |
| [NeuronAgent Docker](NeuronAgent/docker/README.md) | Container deployment |
| [NeuronMCP Docker](NeuronMCP/docker/README.md) | Container deployment |

### Feature Links

| Feature | Local Documentation | Official Documentation |
|---------|-------------------|----------------------|
| [Vector Search](NeuronDB/docs/vector-search/) | Vector indexing and search | [Vector Search Guide](https://www.neurondb.ai/docs/vector-search) |
| [ML Algorithms](NeuronDB/docs/ml-algorithms/) | Machine learning features | [ML Algorithms Guide](https://www.neurondb.ai/docs/ml-algorithms) |
| [RAG Pipeline](NeuronDB/docs/rag/) | Retrieval-augmented generation | [RAG Pipeline Guide](https://www.neurondb.ai/docs/rag) |
| [GPU Acceleration](NeuronDB/docs/gpu/) | GPU support and configuration | [GPU Acceleration Guide](https://www.neurondb.ai/docs/gpu) |
| Hybrid Search | - | [Hybrid Search Guide](https://www.neurondb.ai/docs/hybrid-search) |
| Performance Optimization | - | [Performance Guide](https://www.neurondb.ai/docs/performance) |
| Security Features | - | [Security Guide](https://www.neurondb.ai/docs/security) |
