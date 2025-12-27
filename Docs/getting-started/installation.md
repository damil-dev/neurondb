# Installation Guide

Complete installation guide for the NeuronDB ecosystem.

## Prerequisites

### System Requirements

- **PostgreSQL**: 16, 17, or 18
- **Operating System**: Linux, macOS, or Windows (with WSL2)
- **Docker**: 20.10+ and Docker Compose 2.0+ (for containerized deployment)
- **Go**: 1.23+ (for building from source)
- **Node.js**: 20+ (for NeuronDesktop frontend)

### Optional Requirements

- **GPU Support**: CUDA (NVIDIA), ROCm (AMD), or Metal (Apple Silicon) for GPU acceleration
- **Build Tools**: C compiler (GCC or Clang), Make, PostgreSQL development headers

## Installation Methods

### Method 1: Docker (Recommended)

Docker installation provides the easiest and most consistent setup.

#### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd neurondb2

# Start all services
docker-compose up -d
```

#### Component-Specific Docker Setup

**NeuronDB:**
```bash
cd NeuronDB/docker
docker compose up -d neurondb
```

**NeuronAgent:**
```bash
cd NeuronAgent/docker
docker compose up -d agent-server
```

**NeuronMCP:**
```bash
cd NeuronMCP/docker
docker compose up -d neurondb-mcp
```

**NeuronDesktop:**
```bash
cd NeuronDesktop
docker-compose up -d
```

See the [Docker Deployment Guide](../deployment/docker.md) for detailed instructions.

### Method 2: Source Build

Build and install each component from source.

#### Step 1: Install NeuronDB Extension

```bash
cd NeuronDB
make install PG_CONFIG=/usr/local/pgsql/bin/pg_config

# Create extension in database
psql -d your_database -c "CREATE EXTENSION neurondb;"
```

For detailed instructions, see [NeuronDB Installation Guide](../../NeuronDB/INSTALL.md).

#### Step 2: Build NeuronAgent

```bash
cd NeuronAgent
go build ./cmd/agent-server

# Run with configuration
./agent-server -config configs/config.yaml
```

See [NeuronAgent README](../../NeuronAgent/README.md) for configuration details.

#### Step 3: Build NeuronMCP

```bash
cd NeuronMCP
go build ./cmd/neurondb-mcp

# Run with environment variables
export NEURONDB_HOST=localhost
export NEURONDB_PORT=5432
export NEURONDB_DATABASE=neurondb
./neurondb-mcp
```

See [NeuronMCP README](../../NeuronMCP/README.md) for setup details.

#### Step 4: Build NeuronDesktop

**Backend:**
```bash
cd NeuronDesktop/api
go build ./cmd/server
./server
```

**Frontend:**
```bash
cd NeuronDesktop/frontend
npm install
npm run dev
```

See [NeuronDesktop README](../../NeuronDesktop/README.md) for detailed setup.

### Method 3: Package Installation

Install using platform-specific packages (DEB/RPM).

```bash
# Debian/Ubuntu
sudo dpkg -i neurondb_*.deb

# RHEL/CentOS
sudo rpm -i neurondb_*.rpm
```

See [Packaging Documentation](../../Docs/PACKEGE.md) for package build instructions.

## Database Setup

### Create Database

```bash
createdb neurondb
```

### Install Extensions

```bash
psql -d neurondb -c "CREATE EXTENSION neurondb;"
```

### Run Migrations

**NeuronAgent:**
```bash
cd NeuronAgent
./scripts/run_migrations.sh
```
This runs all migrations including `migrations/001_initial_schema.sql` and subsequent migrations.

**NeuronDesktop:**
```bash
cd NeuronDesktop
createdb neurondesk
./scripts/setup_neurondesktop.sh
```
This runs all migrations including `api/migrations/001_initial_schema.sql` and subsequent migrations.

## Verification

### Verify NeuronDB

```bash
psql -d neurondb -c "SELECT neurondb.version();"
```

### Verify NeuronAgent

```bash
curl http://localhost:8080/health
```

### Verify NeuronMCP

```bash
# Check if binary exists and is executable
which neurondb-mcp
```

### Verify NeuronDesktop

```bash
# Frontend
curl http://localhost:3000

# Backend
curl http://localhost:8081/health
```

## Configuration

### Environment Variables

Each component requires specific environment variables. See component-specific documentation:

- [NeuronDB Configuration](../../NeuronDB/docs/configuration.md)
- [NeuronAgent Configuration](../../NeuronAgent/README.md#configuration)
- [NeuronMCP Configuration](../../NeuronMCP/README.md#configuration)
- [NeuronDesktop Configuration](../../NeuronDesktop/README.md#configuration)

### Configuration Files

- **NeuronAgent**: `NeuronAgent/configs/config.yaml`
- **NeuronMCP**: `NeuronMCP/mcp-config.json`
- **NeuronDesktop**: Environment variables or `.env` file

## Next Steps

1. **[Quick Start Guide](quickstart.md)** - Run your first queries
2. **[Component Documentation](../components/README.md)** - Learn about each component
3. **[Integration Guide](../ecosystem/integration.md)** - Connect components together

## Troubleshooting

### Common Issues

- **Connection Errors**: Verify database is running and connection parameters are correct
- **Extension Not Found**: Ensure NeuronDB extension is installed in the database
- **Port Conflicts**: Check if ports 5432, 8080, 8081, 3000 are available
- **Build Errors**: Verify all prerequisites are installed

For detailed troubleshooting, see:
- [NeuronDB Troubleshooting](../../NeuronDB/docs/troubleshooting.md)
- [Official Documentation](https://www.neurondb.ai/docs/troubleshooting)

## Official Documentation

For comprehensive installation guides and platform-specific instructions:
**🌐 [https://www.neurondb.ai/docs/installation](https://www.neurondb.ai/docs/installation)**

