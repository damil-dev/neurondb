# Quick Start Guide

Get NeuronDB up and running in minutes with this step-by-step guide.

> **New here?** Start with **[Simple Start Guide](Docs/getting-started/simple-start.md)** instead - it explains everything in plain English!

> **TECHNICAL USER?** Continue below for a streamlined technical setup.

---

## Prerequisites

**Before starting, verify you have:**

- [ ] **Docker** 20.10+ and **Docker Compose** 2.0+
- [ ] **5-10 minutes** for setup and verification
- [ ] **4GB RAM** minimum (8GB recommended)
- [ ] Ports **5433, 8080, 8081, 3000** available

<details>
<summary><strong>Verify Docker installation</strong></summary>

```bash
docker --version
docker compose version
```

**Expected output:**
```
Docker version 20.10.0 or higher
Docker Compose version v2.0.0 or higher
```

</details>

## Step 1: Start All Services

Start the complete NeuronDB ecosystem with a single command:

```bash
# From the repository root
# Option 1: Use published images (recommended if available)
docker compose pull
docker compose up -d

# Option 2: Build from source
docker compose up -d --build
```

> [!NOTE]
> Published images from GitHub Container Registry (GHCR) are available starting with v1.0.0. See [Container Images documentation](Docs/deployment/container-images.md) for image names and tags.

This command will:

- [x] Build all Docker images (first time only, takes a few minutes)
- [x] Start PostgreSQL with NeuronDB extension
- [x] Start NeuronAgent (REST API server)
- [x] Start NeuronMCP (MCP protocol server)
- [x] Start NeuronDesktop (web interface with API and frontend)
- [x] Configure networking between all components

**What to expect:**
- First run: 5-10 minutes (building images)
- Subsequent runs: 30-60 seconds (containers already built)

**Check service status:**

```bash
docker compose ps
```

You should see five services running:

| Service | Status | Description |
|---------|--------|-------------|
| `neurondb-cpu` | healthy | PostgreSQL with NeuronDB extension |
| `neuronagent` | healthy | REST API server |
| `neurondb-mcp` | healthy | MCP protocol server |
| `neurondesk-api` | healthy | NeuronDesktop API server |
| `neurondesk-frontend` | healthy | NeuronDesktop web interface |

**Wait for all services to show "healthy" status** (may take 30-60 seconds)

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

**Expected output:**JSON-RPC response with server information.

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
NeuronDB SQL query successful
NeuronAgent REST API responding
NeuronMCP server responding
All smoke tests passed!
```

## Next Steps

**Congratulations!** Your NeuronDB ecosystem is running. Try these examples:

<details>
<summary><strong>Example 1: Create a Vector Table</strong></summary>

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

</details>

<details>
<summary><strong>Example 2: Create an Agent via REST API</strong></summary>

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

**Note:** Replace `YOUR_API_KEY` with an actual API key from NeuronAgent

</details>

## Troubleshooting

**Having issues?** Check these common problems:

### Services Won't Start

**Check logs:**
```bash
docker compose logs neurondb
docker compose logs neuronagent
docker compose logs neurondb-mcp
docker compose logs neurondesk-api
docker compose logs neurondesk-frontend
```

**Common issues:**

<details>
<summary><strong>Port already in use</strong></summary>

- Change ports in `docker-compose.yml` or stop conflicting services
- Default ports: **5433** (PostgreSQL), **8080** (NeuronAgent), **8081** (NeuronDesktop API), **3000** (NeuronDesktop UI)

</details>

<details>
<summary><strong>Out of memory</strong></summary>

- Ensure Docker has at least **4GB RAM** allocated
- Check: Docker Desktop → Settings → Resources
- Recommended: **8GB+** for optimal performance

</details>

<details>
<summary><strong>Build failures</strong></summary>

- Ensure Docker has sufficient disk space (**10GB+** recommended)
- Try: `docker compose build --no-cache`
- Check Docker logs: `docker compose logs --tail=50`

</details>

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
docker compose logs --tail=50 neurondesk-api
docker compose logs --tail=50 neurondesk-frontend
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
docker rmi neurondb:cpu-pg17 neuronagent:latest neurondb-mcp:latest neurondesk-api:latest neurondesk-frontend:latest

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
| **NeuronDesktop API** | `neurondesk-api` | 8081 | NeuronDesktop backend API |
| **NeuronDesktop Frontend** | `neurondesk-frontend` | 3000 | NeuronDesktop web interface |

**Network:**All services communicate via `neurondb-network` Docker network.

**Data:**Persistent data stored in Docker volumes:
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

### NeuronDesktop Web Interface

Access the unified web interface:

```bash
# Web UI (browser)
http://localhost:3000

# API endpoint
curl http://localhost:8081/health
```

## Configuration

All services use default configuration suitable for development. To customize:

1. **Edit `docker-compose.yml`**for service-level changes
2. **Set environment variables**for runtime configuration
3. **Mount configuration files**for advanced setups

See component-specific documentation for detailed configuration options.

## Getting Help

- **Documentation:**See [readme.md](readme.md) for detailed documentation
- **Issues:**Check service logs: `docker compose logs [service-name]`
- **Support:**Contact support@neurondb.ai

## Next Steps

- Read the [full documentation](readme.md)
- Explore [NeuronDB examples](NeuronDB/demo/)
- Try [NeuronAgent examples](NeuronAgent/examples/)
- Check out [NeuronMCP documentation](NeuronMCP/readme.md)
- Access [NeuronDesktop web interface](http://localhost:3000) and see [NeuronDesktop documentation](NeuronDesktop/readme.md)

