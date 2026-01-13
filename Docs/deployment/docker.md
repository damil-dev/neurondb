# Docker Deployment Guide

Complete guide for deploying the NeuronDB ecosystem using Docker.

## Overview

Docker deployment provides the easiest and most consistent way to run all NeuronDB ecosystem components. Each component includes Docker configurations with support for CPU and GPU variants.

## Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- For GPU support: NVIDIA Docker runtime (CUDA) or ROCm drivers (ROCm)

## Quick Start

### Start All Services

```bash
# From repository root
docker compose up -d
```

### Component-Specific Deployment

**NeuronDB (database only):**
```bash
docker compose up -d neurondb
```

**NeuronAgent:**
```bash
docker compose up -d neuronagent
```

**NeuronMCP:**
```bash
docker compose up -d neuronmcp
```

**NeuronDesktop:**
```bash
docker compose up -d neurondesk-api neurondesk-frontend
```

## Unified Docker Orchestration

For unified orchestration of all services, see the [Unified Docker Guide](docker-unified.md) which provides:

- Single command to build and run all services
- Automatic networking between containers
- GPU variant support (CUDA, ROCm, Metal)
- Health checks and service management

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Docker Network: neurondb-network                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌────────────┐│
│  │  NeuronDB    │◄─────┤  NeuronAgent │      │  NeuronMCP  ││
│  │  (PostgreSQL)│      │  (REST API)  │      │  (MCP)      ││
│  │  Port: 5433  │      │  Port: 8080  │      │  (stdio)    ││
│  └──────────────┘      └──────────────┘      └────────────┘│
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              NeuronDesktop                              ││
│  │  Frontend: 3000  │  Backend: 8081                       ││
│  └─────────────────────────────────────────────────────────┘│
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

Each component uses environment variables for configuration. See component-specific Docker guides:

- [NeuronDB Docker](../../dockers/neurondb/readme.md)
- [NeuronAgent Docker](../../dockers/neuronagent/readme.md)
- [NeuronMCP Docker](../../dockers/neuronmcp/readme.md)
- [NeuronDesktop README](../../NeuronDesktop/readme.md)

### Network Configuration

Services communicate via Docker network using service names (not container names):

- **NeuronAgent → NeuronDB**: `neurondb:5432` (service name, internal Docker network)
- **NeuronMCP → NeuronDB**: `neurondb:5432` (service name, internal Docker network)
- **NeuronDesktop → NeuronDB**: `neurondb:5432` (service name, internal Docker network)
- **External Access (from host)**: `localhost:5433` (mapped host port)

**Important:** Inside Docker network, use service name `neurondb` (not container name `neurondb-cpu`). From your host machine, use `localhost:5433`.

## GPU Variants

### CUDA (NVIDIA)

```bash
# Build and run CUDA variant (service is `neurondb-cuda`)
docker compose --profile cuda build neurondb-cuda
docker compose --profile cuda up -d neurondb-cuda
```

### ROCm (AMD)

```bash
# Build and run ROCm variant (service is `neurondb-rocm`)
docker compose --profile rocm build neurondb-rocm
docker compose --profile rocm up -d neurondb-rocm
```

### Metal (Apple Silicon)

```bash
# Build and run Metal variant (service is `neurondb-metal`)
docker compose --profile metal build neurondb-metal
docker compose --profile metal up -d neurondb-metal
```

## Service Management

### Start Services

```bash
docker compose up -d
```

### Stop Services

```bash
docker compose stop
```

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f neurondb
docker compose logs -f agent-server
docker compose logs -f neurondb-mcp
docker compose logs -f neurondesktop
```

### Restart Services

```bash
docker compose restart
```

### Remove Services

```bash
# Stop and remove containers
docker compose down

# Remove containers, networks, and volumes
docker compose down -v
```

## Health Checks

### Verify Services

```bash
# Check container status
docker compose ps

# Test NeuronDB
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" \
  -c "SELECT neurondb.version();"

# Test NeuronAgent
curl http://localhost:8080/health

# Test NeuronDesktop
curl http://localhost:8081/health
```

## Data Persistence

### Volumes

Docker volumes are used for data persistence:

- **NeuronDB**: PostgreSQL data directory
- **NeuronDesktop**: Database and application data

### Backup

```bash
# Backup NeuronDB data
docker exec neurondb-cpu pg_dump -U neurondb neurondb > backup.sql

# Backup NeuronDesktop data
docker exec neurondesktop-api pg_dump -U neurondesk neurondesk > neurondesk_backup.sql
```

## Troubleshooting

### Services Cannot Connect

1. Verify containers are running: `docker compose ps`
2. Check network connectivity: `docker network inspect neurondb-network`
3. Verify environment variables: `docker compose config`
4. Check logs: `docker compose logs`

### Port Conflicts

Change ports in `docker-compose.yml` or `.env` file:

```yaml
ports:
  - "5434:5432"  # Change external port
```

### GPU Not Detected

1. **CUDA**: Verify NVIDIA Docker runtime: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
2. **ROCm**: Check devices: `ls -la /dev/kfd /dev/dri`
3. Check Docker Compose profiles: `docker compose --profile cuda config`

## Best Practices

1. **Use `.env` files**: Store environment variables in `.env` files
2. **Health checks**: Monitor service health regularly
3. **Resource limits**: Set appropriate CPU and memory limits
4. **Backup strategy**: Implement regular backup procedures
5. **Network security**: Use Docker networks to isolate services
6. **Secrets management**: Use Docker secrets or external secret management

## Related Documentation

- [Unified Docker Guide](docker-unified.md) - Complete unified orchestration guide
- [NeuronDB Docker](../../dockers/neurondb/readme.md) - NeuronDB-specific Docker guide
- [NeuronAgent Docker](../../dockers/neuronagent/readme.md) - NeuronAgent Docker guide
- [NeuronMCP Docker](../../dockers/neuronmcp/readme.md) - NeuronMCP Docker guide
- [Ecosystem Integration](../ecosystem/integration.md) - Component integration guide

## Official Documentation

For comprehensive Docker deployment guides:
** [https://www.neurondb.ai/docs/docker](https://www.neurondb.ai/docs/docker)**

