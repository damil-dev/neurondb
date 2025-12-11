# NeuronDB Ecosystem - Unified Build System

## Overview

A modern, unified build system that seamlessly integrates Docker orchestration with native source builds. Choose the build method that fits your needs.

## Quick Reference

### Docker Build (Containerized)
```bash
# Build and run all services
make docker-build
make docker-run

# GPU variants
make docker-build-cuda && make docker-run-cuda
make docker-build-rocm && make docker-run-rocm
make docker-build-metal && make docker-run-metal

# Management
make docker-logs
make docker-status
make docker-health
make docker-stop
make docker-clean
```

### Source Build (Native)
```bash
# Build all components from source
make build

# Individual components
make build-neurondb      # Uses NeuronDB/build.sh
make build-neuronagent   # Uses NeuronAgent/Makefile
make build-neuronmcp     # Uses NeuronMCP/Makefile

# Test and install
make test
make install

# Clean
make clean
```

## Architecture

```
Root Makefile (Unified Interface)
├── Docker Commands (docker-*)
│   ├── docker-build-*    → docker-compose.yml
│   ├── docker-run-*      → docker-compose.yml
│   └── docker-*          → Docker management
│
└── Source Commands (build-*)
    ├── build-neurondb    → NeuronDB/build.sh
    ├── build-neuronagent → NeuronAgent/Makefile
    └── build-neuronmcp   → NeuronMCP/Makefile
```

## Features

✅ **Dual Build System**: Docker and source builds coexist seamlessly
✅ **Modern Docker Compose v2**: No version field, profiles, health checks
✅ **Automatic Integration**: Delegates to existing build scripts
✅ **Clean Separation**: `docker-*` vs `build-*` prefixes
✅ **Comprehensive Help**: `make help` shows all commands
✅ **Production Ready**: Follows all modern best practices

## File Structure

```
/home/pge/pge/neurondb/
├── Makefile              # Unified build system
├── docker-compose.yml    # Docker orchestration
├── .env.example          # Environment template
├── .dockerignore         # Build optimization
├── DOCKER.md             # Docker documentation
├── BUILD_SYSTEM.md       # This file
│
├── NeuronDB/
│   ├── build.sh          # Source build script
│   └── Makefile          # Component Makefile
│
├── NeuronAgent/
│   └── Makefile          # Component Makefile
│
└── NeuronMCP/
    └── Makefile          # Component Makefile
```

## Usage Examples

### Development Workflow

**Using Docker (Recommended for quick start):**
```bash
make docker-build
make docker-run
make docker-logs
```

**Using Source Build (For development/debugging):**
```bash
make build-neurondb
make build-neuronagent
make build-neuronmcp
```

### Production Deployment

**Docker:**
```bash
make docker-build-cuda
make docker-run-cuda
```

**Source:**
```bash
make build
make install
```

## Integration Details

### Docker System
- Uses `docker-compose.yml` at root
- Supports profiles: default, cuda, rocm, metal
- Automatic service discovery via Docker network
- Health checks and structured logging

### Source Build System
- Delegates to existing build infrastructure:
  - `NeuronDB/build.sh` for NeuronDB
  - `NeuronAgent/Makefile` for NeuronAgent
  - `NeuronMCP/Makefile` for NeuronMCP
- Preserves all existing build options
- No changes to component build systems

## Benefits

1. **No Conflicts**: Docker and source builds are completely separate
2. **Backward Compatible**: Existing build scripts unchanged
3. **Unified Interface**: Single `make help` shows everything
4. **Modern Standards**: Follows Docker Compose v2 best practices
5. **Flexible**: Use Docker for deployment, source for development

## Commands Summary

| Category | Command | Description |
|----------|---------|-------------|
| **Docker Build** | `make docker-build` | Build all services (CPU) |
| **Docker Build** | `make docker-build-cuda` | Build CUDA GPU variant |
| **Docker Run** | `make docker-run` | Start all services |
| **Docker Run** | `make docker-run-cuda` | Start CUDA variant |
| **Docker Mgmt** | `make docker-logs` | View logs |
| **Docker Mgmt** | `make docker-status` | Check status |
| **Source Build** | `make build` | Build all from source |
| **Source Build** | `make build-neurondb` | Build NeuronDB |
| **Source Test** | `make test` | Run all tests |
| **Source Install** | `make install` | Install all |
| **Source Clean** | `make clean` | Clean artifacts |

## Next Steps

1. **Docker**: Copy `.env.example` to `.env` and customize
2. **Source**: Ensure PostgreSQL and build dependencies are installed
3. **Both**: Run `make help` to see all available commands

For detailed documentation:
- Docker: See `DOCKER.md`
- Source: See component-specific READMEs
