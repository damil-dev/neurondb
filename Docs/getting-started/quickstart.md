# Quick Start Guide

Get up and running with the NeuronDB ecosystem in minutes.

## Prerequisites

- Docker and Docker Compose installed
- Or PostgreSQL 16+ with NeuronDB extension installed

## Quick Start with Docker

### 1. Start NeuronDB

```bash
cd NeuronDB/docker
docker compose up -d neurondb
```

### 2. Verify Installation

```bash
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" \
  -c "SELECT neurondb.version();"
```

### 3. Create Your First Vector Table

```sql
-- Connect to database
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb"

-- Create table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    embedding vector(384)
);

-- Insert sample data
INSERT INTO documents (title, content, embedding)
VALUES (
    'Machine Learning',
    'Machine learning is a subset of AI',
    neurondb.embed_text('Machine learning is a subset of AI', 'model_name')
);

-- Create HNSW index
SELECT neurondb.hnsw_create_index('documents', 'embedding', 'documents_idx', 16, 200);
```

### 4. Perform Vector Search

```sql
-- Search for similar documents
SELECT id, title,
       embedding <-> neurondb.embed_text('artificial intelligence', 'model_name') AS distance
FROM documents
ORDER BY distance
LIMIT 5;
```

## Quick Start with NeuronAgent

### 1. Start NeuronAgent

```bash
cd NeuronAgent/docker
docker compose up -d agent-server
```

### 2. Create an Agent

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

### 3. Send a Message

```bash
curl -X POST http://localhost:8080/api/v1/sessions/SESSION_ID/messages \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Find documents about machine learning"
  }'
```

## Quick Start with NeuronMCP

### 1. Configure Claude Desktop

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

### 2. Use in Claude Desktop

NeuronMCP tools will be available in Claude Desktop for vector search, embeddings, and ML operations.

## Quick Start with NeuronDesktop

### 1. Start NeuronDesktop

```bash
cd NeuronDesktop
docker-compose up -d
```

### 2. Access the Interface

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8081

### 3. Configure Profiles

1. Navigate to Settings
2. Create a new profile
3. Configure connections to:
   - NeuronDB (PostgreSQL)
   - NeuronAgent (HTTP API)
   - NeuronMCP (MCP server)

### 4. Use the Interface

- **MCP Console**: Test and use MCP tools
- **NeuronDB Search**: Perform vector searches
- **Agent Management**: Create and manage agents

## Next Steps

1. **[Installation Guide](installation.md)** - Complete installation instructions
2. **[Component Documentation](../components/README.md)** - Detailed component guides
3. **[Integration Guide](../ecosystem/integration.md)** - Connect all components
4. **[Official Documentation](https://www.neurondb.ai/docs/getting-started/quickstart)** - Comprehensive quick start

## Example Workflows

### Vector Search Workflow

1. Create vector table
2. Generate embeddings
3. Create HNSW index
4. Perform similarity search

### RAG Workflow

1. Ingest documents into PostgreSQL
2. Generate embeddings using NeuronDB
3. Retrieve context using vector search
4. Pass to LLM through NeuronAgent

### Agent Workflow

1. Create agent with tools
2. Start session
3. Send messages
4. Agent executes tools and responds

## Official Documentation

For detailed tutorials and examples:
**🌐 [https://www.neurondb.ai/docs/getting-started](https://www.neurondb.ai/docs/getting-started)**

