# NeuronDB Scripts

**Professional automation scripts for the NeuronDB ecosystem**

This directory contains practical scripts for managing, deploying, monitoring, and maintaining NeuronDB components.

---

## 📋 Quick Reference

| Script | Purpose | Common Usage |
|--------|---------|--------------|
| **ecosystem-setup.sh** | Complete ecosystem setup | `./ecosystem-setup.sh --mode docker --all` |
| **install.sh** | Simple one-command installer | `./install.sh` |
| **health-check.sh** | Quick health verification | `./health-check.sh` |
| **integration-test.sh** | Comprehensive testing | `./integration-test.sh --tier 0` |
| **backup-database.sh** | Database backup | `./backup-database.sh --format custom` |
| **restore-database.sh** | Database restore | `./restore-database.sh --backup backup.dump` |
| **monitor-status.sh** | Real-time monitoring | `./monitor-status.sh --watch` |
| **view-logs.sh** | View component logs | `./view-logs.sh neuronagent --follow` |
| **cleanup.sh** | Clean resources | `./cleanup.sh --all --dry-run` |

---

## 🎯 Script Categories

### 🚀 Setup & Installation

#### `ecosystem-setup.sh`
**Professional one-command setup for the entire NeuronDB ecosystem**

```bash
# Docker deployment (recommended for getting started)
./ecosystem-setup.sh --mode docker --all

# Install specific components with packages
./ecosystem-setup.sh --mode deb --components NeuronDB NeuronAgent

# Custom database configuration
./ecosystem-setup.sh --mode rpm --components NeuronDB NeuronMCP \
    --db-host db.example.com --db-password secret
```

**Features:**
- ✅ Multiple installation modes: Docker, DEB, RPM, macOS packages
- ✅ Flexible component selection with automatic dependency resolution
- ✅ Database schema setup and migrations
- ✅ Service management (systemd integration)
- ✅ Comprehensive verification
- ✅ Uninstall support with optional data removal
- ✅ Dry-run mode for safe testing

**Options:**
- `--mode [docker|deb|rpm|mac]` - Installation mode
- `--all` - Install all components
- `--components COMP1 COMP2...` - Install specific components
- `--db-host HOST` - Database host (default: localhost)
- `--db-port PORT` - Database port
- `--db-name NAME` - Database name (default: neurondb)
- `--db-user USER` - Database user
- `--db-password PASS` - Database password
- `--skip-setup` - Skip database setup
- `--skip-services` - Skip service startup
- `--uninstall` - Uninstall components
- `--remove-data` - Remove data during uninstall (⚠️ destructive!)
- `--verbose` - Enable verbose output
- `--dry-run` - Preview changes without applying

**Environment Variables:**
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=neurondb
export DB_USER=postgres
export DB_PASSWORD=your_password
```

---

#### `install.sh`
**Simple one-command installer for quick setup**

```bash
# Full installation
./install.sh

# Skip dependency installation
./install.sh --skip-deps

# Skip database setup
./install.sh --skip-setup
```

**What it does:**
1. Detects operating system (Ubuntu, Debian, RHEL, macOS)
2. Installs system dependencies
3. Detects PostgreSQL version
4. Builds and installs NeuronDB extension
5. Sets up database schemas
6. Verifies installation

**Supported Platforms:**
- Ubuntu / Debian
- RHEL / CentOS / Rocky Linux
- macOS (with Homebrew)

---

### ✅ Testing & Verification

#### `health-check.sh`
**Quick health check for all services (30 seconds)**

```bash
# Run health check
./health-check.sh

# Expected output:
# ✓ NeuronDB SQL query successful
# ✓ NeuronAgent REST API responding
# ✓ NeuronMCP server responding
# All smoke tests passed!
```

**Tests:**
1. NeuronDB extension loaded and queryable
2. NeuronAgent REST API responding (HTTP 200)
3. NeuronMCP server responding to MCP protocol

**Exit Codes:**
- `0` - All tests passed
- `1` - One or more tests failed

---

#### `integration-test.sh`
**Comprehensive integration testing suite (5-10 minutes)**

```bash
# Run all tiers
./integration-test.sh

# Run specific tier
./integration-test.sh --tier 0    # Basic extension
./integration-test.sh --tier 3    # ML algorithms

# Run multiple tiers
./integration-test.sh --tier 0 --tier 1 --tier 2

# Verbose output
./integration-test.sh --verbose
```

**Test Tiers:**

| Tier | Category | Tests |
|------|----------|-------|
| **0** | Basic Extension | Extension loading, version, schema creation |
| **1** | Vector Operations | Vector columns, HNSW/IVF indexes, kNN queries |
| **2** | Hybrid Search | Vector + full-text search combination |
| **3** | ML Algorithms | Classification, regression, clustering (52+ algorithms) |
| **4** | Embeddings | Embedding generation, storage, retrieval |
| **5** | NeuronAgent | API endpoints, agent creation, messaging |
| **6** | NeuronMCP | MCP protocol, tools, resources |

**Output:**
```
=== NeuronDB Integration Verification ===

[TIER 0] Basic Extension Tests
✓ Extension loads successfully
✓ Version information correct
✓ Schema created properly

...

=== Summary ===
Tests Passed: 45/45
Tests Failed: 0/45
Overall: PASSED
```

---

### 💾 Backup & Restore

#### `backup-database.sh`
**Professional database backup with multiple formats**

```bash
# Custom format backup (recommended - compressed, supports parallel restore)
./backup-database.sh --format custom

# SQL format with compression
./backup-database.sh --format sql --compress

# Directory format for large databases (parallel dump)
./backup-database.sh --format directory --output /backups/neurondb

# Custom retention policy
./backup-database.sh --retention 7
```

**Features:**
- ✅ Multiple backup formats (SQL, custom, directory)
- ✅ Automatic compression
- ✅ Retention policy management
- ✅ Database size detection
- ✅ Detailed backup information

**Formats:**
- **SQL** - Plain SQL dump (text format, portable)
- **Custom** - PostgreSQL custom format (compressed, supports parallel restore)
- **Directory** - Directory format (parallel dump/restore, best for large DBs)

**Options:**
- `--format [sql|custom|directory]` - Backup format
- `--output PATH` - Output directory (default: ./backups)
- `--compress` - Compress SQL backups
- `--retention DAYS` - Keep backups for N days (default: 30)

**Environment Variables:**
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=neurondb
export DB_USER=postgres
export DB_PASSWORD=your_password
```

---

#### `restore-database.sh`
**Database restore supporting all backup formats**

```bash
# Restore from custom format
./restore-database.sh --backup neurondb_backup_20250101_120000.dump

# Restore from SQL backup
./restore-database.sh --backup neurondb_backup_20250101_120000.sql

# Restore from directory format with 8 parallel jobs
./restore-database.sh --backup neurondb_backup_20250101_120000_dir --jobs 8

# Drop database before restore
./restore-database.sh --backup backup.dump --drop

# Clean database objects before restore
./restore-database.sh --backup backup.dump --clean
```

**Features:**
- ✅ Auto-detects backup format
- ✅ Parallel restore for directory format
- ✅ Optional database drop/recreation
- ✅ Clean mode for object recreation
- ✅ Restore verification

**Options:**
- `--backup PATH` - Backup file or directory (required)
- `--format [sql|custom|directory]` - Force backup format (auto-detected if not specified)
- `--drop` - Drop existing database before restore
- `--clean` - Clean (drop) database objects before recreating
- `--jobs N` - Number of parallel jobs for directory format (default: 4)

---

### 📊 Monitoring & Operations

#### `monitor-status.sh`
**Real-time monitoring of all NeuronDB components**

```bash
# Single status check
./monitor-status.sh

# Continuous monitoring (updates every 5 seconds)
./monitor-status.sh --watch

# JSON output for integration
./monitor-status.sh --json

# Specify deployment mode
./monitor-status.sh --mode docker --watch
```

**Features:**
- ✅ Auto-detects deployment mode (Docker/Native)
- ✅ Health status for all components
- ✅ Version information
- ✅ Resource usage (CPU, memory)
- ✅ Service endpoints
- ✅ JSON output for automation

**Display:**
```
╔═══════════════════════════════════════════════════════════════╗
║              NeuronDB Ecosystem Status Monitor                ║
╚═══════════════════════════════════════════════════════════════╝

Deployment Mode: docker

┌─────────────────────────────────────────────────────────────┐
│                    Component Status                         │
├─────────────────┬─────────────┬───────────┬────────┬────────┤
│ Component       │ Status      │ Version   │ Memory │ CPU    │
├─────────────────┼─────────────┼───────────┼────────┼────────┤
│ NeuronDB        │ healthy     │ 1.0.0     │ 256MB  │ 5.2%   │
│ NeuronAgent     │ healthy     │ 0.1.0     │ 128MB  │ 2.1%   │
│ NeuronMCP       │ healthy     │ stdio     │ 64MB   │ 0.5%   │
│ NeuronDesktop   │ healthy     │ 0.1.0     │ 512MB  │ 8.3%   │
└─────────────────┴─────────────┴───────────┴────────┴────────┘

Service Endpoints:
  ● NeuronDB:        postgresql://localhost:5433/neurondb
  ● NeuronAgent:     http://localhost:8080
  ● NeuronMCP:       stdio protocol
  ● NeuronDesktop:   http://localhost:8081 (API)
                     http://localhost:3000 (UI)
```

**Options:**
- `--mode [docker|native]` - Deployment mode (auto-detected)
- `--watch` - Continuous monitoring with 5-second updates
- `--json` - JSON output for automation

---

#### `view-logs.sh`
**View and follow logs from all components**

```bash
# View all logs (last 50 lines)
./view-logs.sh

# Follow NeuronAgent logs
./view-logs.sh neuronagent --follow

# View last 100 lines of NeuronDB logs
./view-logs.sh neurondb --lines 100

# Follow all logs in Docker mode
./view-logs.sh all --follow --mode docker
```

**Components:**
- `neurondb` - Database server logs
- `neuronagent` - Agent service logs
- `neuronmcp` - MCP server logs
- `neurondesktop` - Desktop service logs
- `all` - All component logs (default)

**Options:**
- `--follow`, `-f` - Follow log output in real-time (tail -f)
- `--lines N` - Number of lines to show (default: 50)
- `--mode [docker|native]` - Deployment mode (auto-detected)

**Auto-detection:**
- Docker mode: Uses `docker logs` command
- Native mode: Finds log files in standard locations
- Systemd integration: Falls back to `journalctl` if needed

---

### 🐳 Docker Management

#### `docker-run-neurondb.sh`
**Run NeuronDB database container**

```bash
# Start with defaults (CPU)
./docker-run-neurondb.sh

# Start with GPU support
GPU=cuda ./docker-run-neurondb.sh

# Custom configuration
PORT=5434 GPU=rocm ./docker-run-neurondb.sh
```

**Environment Variables:**
- `PORT` - PostgreSQL port (default: 5433)
- `GPU` - GPU backend: cpu, cuda, rocm, metal (default: cpu)
- `POSTGRES_PASSWORD` - Database password (default: neurondb)

---

#### `docker-run-neuronagent.sh`
**Run NeuronAgent container**

```bash
./docker-run-neuronagent.sh
```

---

#### `docker-run-neuronmcp.sh`
**Run NeuronMCP container**

```bash
./docker-run-neuronmcp.sh
```

---

#### `docker-run-tests.sh`
**Run all tests in Docker**

```bash
./docker-run-tests.sh
```

---

#### `docker-test-neurondb.sh`
**Test Docker deployment**

```bash
./docker-test-neurondb.sh
```

---

#### `docker-verify-dependencies.sh`
**Verify all required libraries in Docker containers**

```bash
# Check default container
./docker-verify-dependencies.sh

# Check specific container
./docker-verify-dependencies.sh neurondb-cpu
```

**What it checks:**
- PostgreSQL libraries
- ML libraries (XGBoost, LightGBM, CatBoost)
- GPU libraries (CUDA, ROCm, Metal)
- ONNX Runtime
- OpenMP and Eigen
- Python dependencies
- Library path configuration
- Functional tests

---

### 🧹 Maintenance

#### `cleanup.sh`
**Clean build artifacts, Docker resources, logs, and temporary files**

```bash
# Preview what would be cleaned (dry run)
./cleanup.sh --all --dry-run

# Clean everything (Docker, logs, build artifacts, cache)
./cleanup.sh --all

# Clean only Docker resources
./cleanup.sh --docker

# Clean logs and build artifacts
./cleanup.sh --logs --build

# Clean cache (node_modules, venv, etc.)
./cleanup.sh --cache
```

**Features:**
- ✅ Docker cleanup (containers, images, volumes)
- ✅ Log file removal
- ✅ Build artifact cleanup
- ✅ Cache directory cleanup
- ✅ Dry-run mode for safety
- ✅ Interactive confirmation

**Options:**
- `--all` - Clean everything
- `--docker` - Clean Docker containers, images, volumes
- `--logs` - Clean log files
- `--build` - Clean build artifacts and binaries
- `--cache` - Clean cache directories (node_modules, venv)
- `--dry-run` - Preview without making changes

**⚠️ Warning:** `--docker` will stop and remove all containers and volumes, resulting in data loss!

---

## 🔧 Common Workflows

### Complete Fresh Installation

```bash
# 1. Run unified setup (Docker recommended)
./ecosystem-setup.sh --mode docker --all

# 2. Verify installation
./integration-test.sh --tier 0

# 3. Quick health check
./health-check.sh

# 4. Monitor status
./monitor-status.sh
```

---

### Source Installation (Native)

```bash
# 1. Simple installation
./install.sh

# 2. Verify
./health-check.sh

# 3. Run comprehensive tests
./integration-test.sh
```

---

### Docker Deployment

```bash
# 1. Start all services
docker compose up -d

# Or use make targets
make docker-run

# 2. Verify dependencies
./docker-verify-dependencies.sh

# 3. Run Docker tests
./docker-test-neurondb.sh

# 4. Health check
./health-check.sh

# 5. Monitor
./monitor-status.sh --watch
```

---

### Package Installation

```bash
# Build packages first (if needed)
cd packaging/deb
./build-all-deb.sh
cd ../..

# Install with ecosystem setup
./scripts/ecosystem-setup.sh --mode deb --all

# Verify
./scripts/integration-test.sh --tier 0
```

---

### Backup & Restore Workflow

```bash
# 1. Create backup
./backup-database.sh --format custom --retention 30

# 2. List backups
ls -lh backups/

# 3. Restore if needed
./restore-database.sh --backup backups/neurondb_backup_20250101_120000.dump

# 4. Verify restore
./health-check.sh
```

---

### Development Testing

```bash
# Quick smoke test (30 seconds)
./health-check.sh

# Full integration testing (5-10 minutes)
./integration-test.sh

# Specific tier testing
./integration-test.sh --tier 0  # Basic extension
./integration-test.sh --tier 1  # Vector ops
./integration-test.sh --tier 3  # ML algorithms

# Continuous monitoring during development
./monitor-status.sh --watch
```

---

### Troubleshooting & Debugging

```bash
# 1. Check component status
./monitor-status.sh

# 2. View logs
./view-logs.sh all
./view-logs.sh neuronagent --follow

# 3. Run health check
./health-check.sh

# 4. Run basic integration tests
./integration-test.sh --tier 0 --verbose

# 5. Check Docker dependencies (if using Docker)
./docker-verify-dependencies.sh
```

---

### Cleanup & Maintenance

```bash
# Preview cleanup (safe)
./cleanup.sh --all --dry-run

# Clean logs only
./cleanup.sh --logs

# Clean build artifacts
./cleanup.sh --build

# Clean Docker resources
./cleanup.sh --docker

# Full cleanup
./cleanup.sh --all

# Then rebuild if needed
make build              # Native build
make docker-build       # Docker build
```

---

### Uninstallation

```bash
# Preview uninstall (Docker)
./ecosystem-setup.sh --mode docker --all --uninstall --dry-run

# Uninstall Docker deployment (keep volumes)
./ecosystem-setup.sh --mode docker --all --uninstall

# Uninstall with data removal (⚠️ destructive!)
./ecosystem-setup.sh --mode docker --all --uninstall --remove-data

# Uninstall packages
./ecosystem-setup.sh --mode deb --components NeuronDB NeuronAgent --uninstall
```

---

## 📚 Script Development Guidelines

### Naming Convention
- Use **kebab-case** for script names: `ecosystem-setup.sh`, `backup-database.sh`
- Use descriptive names: `integration-test.sh` not `test.sh`
- Prefix Docker scripts: `docker-run-neurondb.sh`

### Script Structure
```bash
#!/bin/bash
#
# Script Name: my-script.sh
# Description: Brief description of what the script does
# Usage: ./my-script.sh [options]

set -euo pipefail  # Exit on error, undefined variables, pipe failures

# Configuration and Constants
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly NC='\033[0m'

# Functions
show_help() {
    # Help text
}

main() {
    # Main logic
}

main "$@"
```

### Best Practices

1. **Error Handling**
   - Use `set -euo pipefail`
   - Check prerequisites
   - Provide helpful error messages

2. **Documentation**
   - Include header comment with description and usage
   - Provide `--help` option
   - Add examples

3. **Modularity**
   - Break complex logic into functions
   - Reuse common functions
   - Keep scripts focused

4. **User Experience**
   - Color-coded output for readability
   - Progress indicators
   - Verbose mode option
   - Dry-run mode for safety

5. **Environment Variables**
   - Use standard environment variables
   - Provide sensible defaults
   - Document all variables

---

## 🆘 Troubleshooting

### Common Issues

#### Script Permission Denied
```bash
# Make script executable
chmod +x scripts/script-name.sh
```

#### Environment Variables Not Set
```bash
# Check current variables
env | grep DB_

# Set required variables
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=neurondb
```

#### Database Connection Failed
```bash
# Test connection manually
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT 1;"

# Check database is running
docker compose ps neurondb

# Check logs
./view-logs.sh neurondb
```

#### Docker Container Not Found
```bash
# List running containers
docker compose ps

# Start services
docker compose up -d

# Check logs
./view-logs.sh --follow
```

---

## 📊 Script Statistics

| Category | Scripts | Description |
|----------|---------|-------------|
| **Setup & Installation** | 2 | Ecosystem setup, simple installer |
| **Testing & Verification** | 2 | Health checks, integration tests |
| **Backup & Restore** | 2 | Database backup and restore |
| **Monitoring** | 2 | Status monitoring, log viewing |
| **Docker** | 6 | Docker container management |
| **Maintenance** | 1 | Cleanup and maintenance |
| **Total** | 15 | Professional production scripts |

---

## 🔗 Related Resources

### Documentation
- **[Main README](../readme.md)** - Project overview
- **[Quick Start](../QUICKSTART.md)** - Get started quickly
- **[Documentation Index](../documentation.md)** - Complete documentation

### Component Documentation
- **[NeuronDB](../NeuronDB/readme.md)** - Database extension
- **[NeuronAgent](../NeuronAgent/readme.md)** - Agent runtime
- **[NeuronMCP](../NeuronMCP/readme.md)** - MCP server
- **[NeuronDesktop](../NeuronDesktop/readme.md)** - Web interface

### Guides
- **[Installation Guide](../NeuronDB/INSTALL.md)** - Detailed installation
- **[Deployment Guide](../NeuronAgent/docs/DEPLOYMENT.md)** - Production deployment
- **[Docker Guide](../dockers/readme.md)** - Container deployment
- **[Testing Guide](../NeuronAgent/TESTING.md)** - Testing strategies

---

## 🤝 Contributing

### Adding New Scripts

1. Create script in `scripts/` directory
2. Follow naming conventions (kebab-case)
3. Include documentation header
4. Add `--help` option
5. Test thoroughly
6. Update this README
7. Submit pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for full guidelines.

---

## 📝 Examples Moved

Example scripts have been moved to the `examples/` directory:

- **LLM Training Examples**: `examples/llm_training/`
  - `train_postgres_llm.py` - Train custom LLM models
  - `export_to_ollama.sh` - Export models to Ollama
  - `start_custom_llm_system.sh` - Start custom LLM server
  - `stop_custom_llm_system.sh` - Stop custom LLM server

- **Data Loading Examples**: `examples/data_loading/`
  - `load_huggingface_dataset.py` - Load HuggingFace datasets

---

## 🎓 Learning Path

### New Users
1. Start with [Quick Start Guide](../QUICKSTART.md)
2. Run `./health-check.sh` to verify setup
3. Try `./monitor-status.sh --watch` to see the system running
4. Explore `./view-logs.sh` to understand component behavior

### Developers
1. Use `./integration-test.sh` regularly during development
2. Monitor with `./monitor-status.sh --watch` while coding
3. Clean up with `./cleanup.sh --build` between rebuilds
4. Back up test data with `./backup-database.sh`

### DevOps/Operators
1. Deploy with `./ecosystem-setup.sh --mode docker --all`
2. Set up monitoring with `./monitor-status.sh --json` (integrate with your tools)
3. Implement backup strategy with `./backup-database.sh --retention 30`
4. Use `./view-logs.sh` for troubleshooting

---

**Last Updated:** 2025-12-31  
**Scripts Version:** 2.0.0  
**Maintainer:** NeuronDB Team

---

## 📬 Support

For issues or questions:
- **GitHub Issues**: [github.com/neurondb/neurondb/issues](https://github.com/neurondb/neurondb/issues)
- **Documentation**: [www.neurondb.ai/docs](https://www.neurondb.ai/docs)
- **Community**: [discord.gg/neurondb](https://discord.gg/neurondb)
