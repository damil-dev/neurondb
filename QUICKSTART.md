# Quick Start Guide

Get NeuronDB up and running in minutes with this step-by-step guide. No tribal knowledge required.

## Prerequisites

Before starting, ensure you have:

- **Docker** 20.10+ and **Docker Compose** 2.0+
- **5-10 minutes** for setup and verification
- **4GB RAM** minimum (8GB recommended)

Verify Docker installation:

```bash
docker --version
docker compose version
```

## Step 1: Start All Services

Start the complete NeuronDB ecosystem with a single command:

```bash
# From the repository root
docker compose up -d
```

This command will:
1. Build all Docker images (first time only, takes a few minutes)
2. Start PostgreSQL with NeuronDB extension
3. Start NeuronAgent (REST API server)
4. Start NeuronMCP (MCP protocol server)
5. Configure networking between all components

**What to expect:**
- First run: 5-10 minutes (building images)
- Subsequent runs: 30-60 seconds (containers already built)

**Check service status:**

```bash
docker compose ps
```

You should see three services running:
- `neurondb-cpu` (PostgreSQL with NeuronDB extension)
- `neuronagent` (REST API server)
- `neurondb-mcp` (MCP protocol server)

Wait for all services to show "healthy" status (may take 30-60 seconds).

## Step 2: Run Smoke Tests

Verify everything works with these quick smoke tests.

### Test 1: SQL Query (NeuronDB Extension)

Test that NeuronDB extension is loaded and functional:

```bash
docker compose exec neurondb psql -U neurondb -d neurondb -c "SELECT neurondb.version();"
```

**Expected output:**
```
version
--------
1.0.0
(1 row)
```

### Test 2: REST API Call (NeuronAgent)

Test that NeuronAgent REST API is responding:

```bash
curl http://localhost:8080/health
```

**Expected output:**
```json
{"status":"ok"}
```

### Test 3: MCP Protocol Call (NeuronMCP)

Test that NeuronMCP server responds to MCP protocol:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}' | \
docker compose exec -T neurondb-mcp /app/neurondb-mcp | head -20
```

**Expected output:** JSON-RPC response with server information.

**Alternative: Use the Python MCP client:**

```bash
cd NeuronMCP/client
pip install -r requirements.txt
./neurondb_mcp_client.py -c ../../neuronmcp_server.json -e "list_tools" 2>/dev/null | head -30
```

## Step 3: Quick Verification Script

Run the automated smoke test script:

```bash
./scripts/smoke-test.sh
```

This script runs all three tests above and reports success or failure.

**Expected output:**
```
✓ NeuronDB SQL query successful
✓ NeuronAgent REST API responding
✓ NeuronMCP server responding
All smoke tests passed!
```

## Next Steps

Now that everything is running, try these examples:

### Example 1: Create a Vector Table

```bash
docker compose exec neurondb psql -U neurondb -d neurondb <<EOF
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding vector(1536)
);

INSERT INTO documents (content, embedding)
VALUES ('Hello, world!', '[0.1, 0.2, 0.3]'::vector);

SELECT id, content FROM documents;
EOF
```

### Example 2: Create an Agent via REST API

```bash
# First, create an API key (check NeuronAgent documentation)
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-agent",
    "system_prompt": "You are a helpful assistant",
    "model_name": "gpt-4",
    "enabled_tools": [],
    "config": {}
  }'
```

## Troubleshooting

### Services Won't Start

**Check logs:**
```bash
docker compose logs neurondb
docker compose logs neuronagent
docker compose logs neurondb-mcp
```

**Common issues:**

1. **Port already in use:**
   - Change ports in `docker-compose.yml` or stop conflicting services
   - Default ports: 5433 (PostgreSQL), 8080 (NeuronAgent)

2. **Out of memory:**
   - Ensure Docker has at least 4GB RAM allocated
   - Check: Docker Desktop → Settings → Resources

3. **Build failures:**
   - Ensure Docker has sufficient disk space (10GB+ recommended)
   - Try: `docker compose build --no-cache`

### Services Start But Tests Fail

**Check service health:**
```bash
docker compose ps
```

All services should show "healthy" status. If not, check logs:

```bash
docker compose logs --tail=50 neurondb
docker compose logs --tail=50 neuronagent
docker compose logs --tail=50 neurondb-mcp
```

**Wait for initialization:**
- NeuronDB may take 30-60 seconds to initialize
- NeuronAgent waits for NeuronDB to be ready
- NeuronMCP waits for NeuronDB to be ready

## Uninstall and Cleanup

When you're done, clean up all resources:

### Stop Services

```bash
docker compose down
```

This stops all containers but preserves data volumes.

### Remove All Resources (Full Cleanup)

**Warning: This will delete all data!**

```bash
# Stop and remove containers, networks, and volumes
docker compose down -v

# Remove Docker images (optional)
docker rmi neurondb:cpu-pg17 neuronagent:latest neurondb-mcp:latest

# Remove Docker network (if it persists)
docker network rm neurondb-network 2>/dev/null || true
```

### Partial Cleanup Options

**Keep data, remove containers:**
```bash
docker compose down
```

**Remove data volumes but keep images:**
```bash
docker compose down -v
```

**Remove specific service only:**
```bash
docker compose stop neuronagent
docker compose rm neuronagent
```

## What's Running?

After `docker compose up -d`, you have:

| Service | Container Name | Port | Description |
|---------|---------------|------|-------------|
| **NeuronDB** | `neurondb-cpu` | 5433 | PostgreSQL with NeuronDB extension |
| **NeuronAgent** | `neuronagent` | 8080 | REST API server for agent runtime |
| **NeuronMCP** | `neurondb-mcp` | - | MCP protocol server (stdio) |

**Network:** All services communicate via `neurondb-network` Docker network.

**Data:** Persistent data stored in Docker volumes:
- `neurondb-data` (PostgreSQL data)

## Accessing Services

### PostgreSQL (NeuronDB)

```bash
# Connect via psql
docker compose exec neurondb psql -U neurondb -d neurondb

# Or from host
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb"
```

### NeuronAgent REST API

```bash
# Health check (no auth required)
curl http://localhost:8080/health

# API endpoints (auth required)
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8080/api/v1/agents
```

### NeuronMCP Server

MCP server communicates via stdio (JSON-RPC 2.0). Use MCP clients or the Python client:

```bash
cd NeuronMCP/client
./neurondb_mcp_client.py -c ../../neuronmcp_server.json -e "list_tools"
```

## Configuration

All services use default configuration suitable for development. To customize:

1. **Edit `docker-compose.yml`** for service-level changes
2. **Set environment variables** for runtime configuration
3. **Mount configuration files** for advanced setups

See component-specific documentation for detailed configuration options.

## Getting Help

- **Documentation:** See [README.md](README.md) for detailed documentation
- **Issues:** Check service logs: `docker compose logs [service-name]`
- **Support:** Contact support@neurondb.ai

## Next Steps

- Read the [full documentation](README.md)
- Explore [NeuronDB examples](NeuronDB/demo/)
- Try [NeuronAgent examples](NeuronAgent/examples/)
- Check out [NeuronMCP documentation](NeuronMCP/README.md)

