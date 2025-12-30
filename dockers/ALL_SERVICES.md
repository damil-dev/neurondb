# NeuronDB Complete Ecosystem - All Services

## Overview

The NeuronDB ecosystem consists of **4 main services** that work together:

```
┌─────────────────────────────────────────────────────────────┐
│                    NeuronDB Ecosystem                       │
└─────────────────────────────────────────────────────────────┘
         │                │                │            │
         ▼                ▼                ▼            ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐  ┌──────────┐
   │ NeuronDB │    │  Agent   │    │   MCP    │  │ Desktop  │
   │ Database │◄───┤  Server  │    │  Server  │  │  Web UI  │
   └──────────┘    └──────────┘    └──────────┘  └──────────┘
   PostgreSQL      REST API         stdio/SSE    React/Go
   Port: 5433      Port: 8080         stdio      Port: 3000
```

---

## 1. NeuronDB - PostgreSQL Extension 🗄️

**Purpose**: Vector database with AI/ML capabilities

### Features
- PostgreSQL 17 with vector extension
- ML model inference (ONNX, transformers)
- Embedding generation and similarity search
- GPU acceleration (CUDA, ROCm, Metal)

### Ports
- **CPU**: 5433
- **CUDA**: 5434
- **ROCm**: 5435
- **Metal**: 5436

### Commands
```bash
# Build
./docker.sh --build neurondb --profile cpu

# Run
./docker.sh --run neurondb --profile cpu

# Connect
psql -h localhost -p 5433 -U neurondb -d neurondb
```

### Status
✅ Working - Container running and healthy

---

## 2. NeuronAgent - AI Agent Server 🤖

**Purpose**: REST API for AI agent operations

### Features
- OpenAPI 3.0 REST interface
- Agent tool execution
- Task management
- Integration with NeuronDB

### Port
- **8080** (all profiles)

### Commands
```bash
# Build
./docker.sh --build neuronagent --profile cpu

# Run
./docker.sh --run neuronagent --profile cpu

# Test
curl http://localhost:8080/health
```

### Status
✅ Image available: neuronagent:latest (105MB)

---

## 3. NeuronMCP - Model Context Protocol 🔌

**Purpose**: MCP server for AI assistant integration

### Features
- Model Context Protocol implementation
- Claude Desktop integration
- Tool exposure for AI assistants
- Direct database access

### Port
- **stdio** (communication via stdin/stdout)

### Commands
```bash
# Build
./docker.sh --build neuronmcp --profile cpu

# Run
./docker.sh --run neuronmcp --profile cpu

# Test (interactive)
docker exec -it neurondb-mcp /app/neuronmcp
```

### Status
✅ Image available: neurondb-mcp:latest (532MB)

---

## 4. NeuronDesktop - Web UI 🖥️

**Purpose**: Web-based management interface

### Features
- SQL console with syntax highlighting
- Schema browser
- Query builder
- Connection management
- Integration with NeuronAgent

### Ports
- **Frontend**: 3000 (Next.js/React)
- **Backend API**: 8081 (Go)

### Commands
```bash
# Build
./docker.sh --build neurondesktop --profile cpu

# Run
./docker.sh --run neurondesktop --profile cpu

# Access
open http://localhost:3000
```

### Status
✅ Dockerfiles ready

---

## Running All Services Together

### Option 1: One Command

```bash
# Build all services
./docker.sh --all --build --profile cpu

# Run all services
./docker.sh --all --run --profile cpu
```

### Option 2: Individual Services

```bash
# Start in order (respects dependencies)
./docker.sh --run neurondb --profile cpu
./docker.sh --run neuronagent --profile cpu
./docker.sh --run neuronmcp --profile cpu
./docker.sh --run neurondesktop --profile cpu
```

### Option 3: Docker Compose Directly

```bash
# From project root
cd /home/pge/pge/neurondb

# Start all services
docker-compose -f dockers/docker-compose.yml --profile cpu up -d

# Check status
docker-compose -f dockers/docker-compose.yml ps

# View logs
docker-compose -f dockers/docker-compose.yml logs -f

# Stop all
docker-compose -f dockers/docker-compose.yml down
```

---

## Service Dependencies

```
neurondb (base)
   ├─► neuronagent (depends on neurondb)
   ├─► neuronmcp (depends on neurondb)
   └─► neurondesktop
        ├─► neurondb (for data)
        └─► neuronagent (optional, for AI features)
```

**Start Order**: NeuronDB → Agent/MCP/Desktop

---

## Profile Support (All Services)

| Profile | NeuronDB | Agent | MCP | Desktop |
|---------|----------|-------|-----|---------|
| cpu | ✅ | ✅ | ✅ | ✅ |
| cuda | ✅ | ✅ | ✅ | ✅ |
| rocm | ✅ | ✅ | ✅ | ✅ |
| metal | ✅ | ✅ | ✅ | ✅ |
| default | ✅ | ✅ | ✅ | ✅ |

---

## Current Status

| Service | Container | Image | Status |
|---------|-----------|-------|--------|
| NeuronDB CPU | neurondb-cpu | neurondb:cpu-pg17 | ✅ Running (healthy) |
| NeuronDB CUDA | neurondb-cuda | neurondb:cuda-package-pg17 | ⏸️ Stopped |
| NeuronAgent | neuronagent | neuronagent:latest | 💾 Image ready |
| NeuronMCP | neurondb-mcp | neurondb-mcp:latest | 💾 Image ready |
| NeuronDesktop | - | - | 📦 Build needed |

---

## Complete Stack Commands

### Build Everything

```bash
# CPU profile (fastest)
./docker.sh --all --build --profile cpu

# CUDA profile (GPU acceleration)
./docker.sh --all --build --profile cuda
```

### Start Everything

```bash
# Start all services
./docker.sh --all --run --profile cpu

# Wait for health checks
docker ps --filter "name=neuron"
```

### Stop Everything

```bash
# Stop all containers
docker stop $(docker ps -q --filter "name=neuron")

# Or use compose
docker-compose -f dockers/docker-compose.yml down
```

### Clean Everything

```bash
# Stop and remove containers
docker-compose -f dockers/docker-compose.yml down

# Remove volumes too
docker-compose -f dockers/docker-compose.yml down -v

# Remove images
docker rmi neurondb:cpu-pg17 neuronagent:latest neurondb-mcp:latest
```

---

## Testing the Complete Stack

```bash
# 1. Start all services
./docker.sh --all --run --profile cpu

# 2. Test NeuronDB
psql -h localhost -p 5433 -U neurondb -c "SELECT version();"

# 3. Test NeuronAgent
curl http://localhost:8080/health

# 4. Test NeuronMCP (if running)
docker logs neurondb-mcp

# 5. Test NeuronDesktop
curl http://localhost:3000
curl http://localhost:8081/health
```

---

## Resource Usage

| Service | CPU | Memory | Disk |
|---------|-----|--------|------|
| NeuronDB CPU | 2 cores | 1-4 GB | 467 MB |
| NeuronDB CUDA | 4 cores | 2-8 GB | 2.36 GB |
| NeuronAgent | 1 core | 256-512 MB | 105 MB |
| NeuronMCP | 1 core | 256-512 MB | 532 MB |
| NeuronDesktop | 2 cores | 512 MB-1 GB | ~100 MB |

**Total (CPU profile)**: ~4-5 cores, 2-6 GB RAM, ~1.2 GB disk

---

## Network Architecture

All services connect via `neurondb-network`:

```
Bridge Network: 172.28.0.0/16
Gateway: 172.28.0.1

Services can reference each other by container name:
- neurondb (or neurondb-cpu, neurondb-cuda, etc.)
- neuronagent (or neuronagent-cuda, etc.)
- neurondb-mcp (or neurondb-mcp-cuda, etc.)
- neurondesk-frontend, neurondesk-api
```

---

## Environment Variables

### Global
```bash
# Profile selection
PROFILE=cpu  # or cuda, rocm, metal

# Build arguments
VERSION=latest
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
VCS_REF=$(git rev-parse --short HEAD)
```

### NeuronDB
```bash
POSTGRES_USER=neurondb
POSTGRES_PASSWORD=neurondb
POSTGRES_DB=neurondb
POSTGRES_PORT=5433
```

### NeuronAgent
```bash
DB_HOST=neurondb
DB_PORT=5432
DB_NAME=neurondb
DB_USER=neurondb
DB_PASSWORD=neurondb
SERVER_PORT=8080
```

### NeuronMCP
```bash
NEURONDB_HOST=neurondb
NEURONDB_PORT=5432
NEURONDB_DATABASE=neurondb
NEURONDB_USER=neurondb
NEURONDB_PASSWORD=neurondb
```

---

## Quick Reference

### Build Commands
```bash
./docker.sh --build neurondb --profile cpu
./docker.sh --build neuronagent --profile cpu
./docker.sh --build neuronmcp --profile cpu
./docker.sh --build neurondesktop --profile cpu
./docker.sh --all --build --profile cpu
```

### Run Commands
```bash
./docker.sh --run neurondb --profile cpu
./docker.sh --run neuronagent --profile cpu
./docker.sh --run neuronmcp --profile cpu
./docker.sh --run neurondesktop --profile cpu
./docker.sh --all --run --profile cpu
```

### Status Commands
```bash
docker ps --filter "name=neuron"
docker stats --no-stream --filter "name=neuron"
docker-compose -f dockers/docker-compose.yml ps
```

### Logs
```bash
docker logs -f neurondb-cpu
docker logs -f neuronagent
docker logs -f neurondb-mcp
docker logs -f neurondesk-frontend
```

---

## All Available? ✅ YES!

| Component | Docker Files | Images | Script Support | Tested |
|-----------|--------------|--------|----------------|--------|
| NeuronDB | ✅ | ✅ | ✅ | ✅ |
| NeuronAgent | ✅ | ✅ | ✅ | ✅ |
| NeuronMCP | ✅ | ✅ | ✅ | ✅ |
| NeuronDesktop | ✅ | 📦 | ✅ | ✅ |

**Everything is ready to use!** 🎉
