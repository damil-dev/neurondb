# NeuronDB Scripts

**Utility scripts for installation, setup, testing, and maintenance**

This directory contains production-ready scripts for managing the NeuronDB ecosystem.

---

## 📋 Script Categories

### 🚀 Setup & Installation

| Script | Description | Usage |
|--------|-------------|-------|
| **`neurondb-setup.sh`** | Complete ecosystem setup | `./neurondb-setup.sh` |
| **`install.sh`** | Install NeuronDB components | `./install.sh` |
| **`setup_neurondb_ecosystem.sh`** | Unified database setup | `./setup_neurondb_ecosystem.sh` |

### ✅ Testing & Verification

| Script | Description | Usage |
|--------|-------------|-------|
| **`smoke-test.sh`** | Quick health checks | `./smoke-test.sh` |
| **`verify_neurondb_integration.sh`** | Comprehensive integration tests | `./verify_neurondb_integration.sh` |
| **`test_neurondb_queries.sh`** | SQL query testing | `./test_neurondb_queries.sh` |
| **`verify_neurondb_docker_dependencies.sh`** | Verify Docker dependencies | `./verify_neurondb_docker_dependencies.sh` |

### 🐳 Docker Management

| Script | Description | Usage |
|--------|-------------|-------|
| **`run_neurondb_docker.sh`** | Run NeuronDB container | `./run_neurondb_docker.sh` |
| **`run_neuronagent_docker.sh`** | Run NeuronAgent container | `./run_neuronagent_docker.sh` |
| **`run_neuronmcp_docker.sh`** | Run NeuronMCP container | `./run_neuronmcp_docker.sh` |
| **`test_neurondb_docker.sh`** | Test Docker deployment | `./test_neurondb_docker.sh` |
| **`run_tests_docker.sh`** | Run all tests in Docker | `./run_tests_docker.sh` |
| **`check_container_libraries.sh`** | Verify container libraries | `./check_container_libraries.sh` |

### 🔧 Maintenance & Operations

| Script | Description | Usage |
|--------|-------------|-------|
| **`start_custom_llm_system.sh`** | Start custom LLM setup | `./start_custom_llm_system.sh` |
| **`stop_custom_llm_system.sh`** | Stop custom LLM setup | `./stop_custom_llm_system.sh` |
| **`export_to_ollama.sh`** | Export models to Ollama | `./export_to_ollama.sh [model]` |
| **`load_huggingface_dataset.py`** | Load HuggingFace datasets | `python load_huggingface_dataset.py` |
| **`train_postgres_llm.py`** | Train PostgreSQL-specific LLMs | `python train_postgres_llm.py` |

---

## 🎯 Common Workflows

### Complete Fresh Installation

```bash
# 1. Run unified setup (creates DB, installs extensions, runs migrations)
./scripts/setup_neurondb_ecosystem.sh

# 2. Verify installation with comprehensive tests
./scripts/verify_neurondb_integration.sh

# 3. Quick health check
./scripts/smoke-test.sh
```

### Docker Deployment

```bash
# Start all services
docker compose up -d

# Verify Docker dependencies
./scripts/verify_neurondb_docker_dependencies.sh

# Run Docker tests
./scripts/test_neurondb_docker.sh

# Check specific service
./scripts/run_neurondb_docker.sh  # Just NeuronDB
```

### Development Testing

```bash
# Quick smoke test (30 seconds)
./scripts/smoke-test.sh

# Full integration testing (5 minutes)
./scripts/verify_neurondb_integration.sh

# Specific tier testing
./scripts/verify_neurondb_integration.sh --tier 0  # Basic extension
./scripts/verify_neurondb_integration.sh --tier 1  # Vector ops
./scripts/verify_neurondb_integration.sh --tier 3  # ML algorithms
```

---

## 📖 Script Details

### Setup Scripts

#### `neurondb-setup.sh`

Complete automated setup for the entire NeuronDB ecosystem.

**What it does:**
- Detects platform (macOS, Linux)
- Installs system dependencies
- Builds all components
- Configures database
- Runs migrations
- Verifies installation

**Usage:**
```bash
./scripts/neurondb-setup.sh

# With custom database
export DB_NAME=mydb
export DB_USER=myuser
./scripts/neurondb-setup.sh
```

**Environment Variables:**
- `DB_HOST` - Database host (default: localhost)
- `DB_PORT` - Database port (default: 5432)
- `DB_NAME` - Database name (default: neurondb)
- `DB_USER` - Database user (default: postgres)
- `DB_PASSWORD` - Database password (optional)

---

#### `setup_neurondb_ecosystem.sh`

Unified database setup for all components.

**What it does:**
1. Creates database if it doesn't exist
2. Installs NeuronDB extension
3. Sets up NeuronMCP schema (13 tables, 30+ functions)
4. Runs NeuronAgent migrations (4 files)
5. Verifies all schemas

**Usage:**
```bash
# Use defaults
./scripts/setup_neurondb_ecosystem.sh

# Custom configuration
export DB_HOST=dbhost
export DB_PORT=5432
export DB_NAME=neurondb
export DB_USER=postgres
export DB_PASSWORD=secret
./scripts/setup_neurondb_ecosystem.sh
```

**Output:**
```
=== NeuronDB Ecosystem Setup ===
✓ Database neurondb exists
✓ NeuronDB extension installed
✓ NeuronMCP schema setup complete
✓ NeuronAgent migrations complete
✓ Verification successful
```

---

### Testing Scripts

#### `smoke-test.sh`

Quick health check for all services (30 seconds).

**What it tests:**
1. NeuronDB extension loaded
2. NeuronAgent API responding
3. NeuronMCP server responding

**Usage:**
```bash
./scripts/smoke-test.sh

# Expected output:
# ✓ NeuronDB SQL query successful
# ✓ NeuronAgent REST API responding
# ✓ NeuronMCP server responding
# All smoke tests passed!
```

**Exit Codes:**
- `0` - All tests passed
- `1` - One or more tests failed

---

#### `verify_neurondb_integration.sh`

Comprehensive integration testing (5-10 minutes).

**Test Tiers:**

| Tier | Category | Tests |
|------|----------|-------|
| **0** | Basic Extension | Load, version, schema |
| **1** | Vector Operations | Create, index, kNN query |
| **2** | Hybrid Search | Vector + full-text |
| **3** | ML Algorithms | Classification, regression, clustering |
| **4** | Embeddings | Generation, storage, query |
| **5** | NeuronAgent | API, agents, messaging |
| **6** | NeuronMCP | Protocol, tools |

**Usage:**
```bash
# Run all tiers
./scripts/verify_neurondb_integration.sh

# Run specific tier
./scripts/verify_neurondb_integration.sh --tier 0
./scripts/verify_neurondb_integration.sh --tier 3

# Run multiple tiers
./scripts/verify_neurondb_integration.sh --tier 0 --tier 1

# Verbose output
./scripts/verify_neurondb_integration.sh --verbose
```

**Example Output:**
```
=== NeuronDB Integration Verification ===

[TIER 0] Basic Extension Tests
✓ Extension loads successfully
✓ Version information correct
✓ Schema created properly

[TIER 1] Vector Operations Tests
✓ Vector column creation
✓ HNSW index creation
✓ kNN query execution

... (continued for all tiers)

=== Summary ===
Tests Passed: 45/45
Tests Failed: 0/45
Overall: PASSED
```

---

#### `test_neurondb_queries.sh`

Test specific SQL queries and operations.

**Usage:**
```bash
./scripts/test_neurondb_queries.sh

# Test specific query file
./scripts/test_neurondb_queries.sh tests/vector_ops.sql
```

---

### Docker Scripts

#### `run_neurondb_docker.sh`

Run NeuronDB database container.

**Usage:**
```bash
# Start with defaults
./scripts/run_neurondb_docker.sh

# Start with GPU support
GPU=cuda ./scripts/run_neurondb_docker.sh

# Custom configuration
PORT=5433 GPU=cuda ./scripts/run_neurondb_docker.sh
```

**Environment Variables:**
- `PORT` - PostgreSQL port (default: 5433)
- `GPU` - GPU backend: `cpu`, `cuda`, `rocm`, `metal` (default: cpu)
- `POSTGRES_PASSWORD` - Database password (default: neurondb)

---

#### `verify_neurondb_docker_dependencies.sh`

Verify all required libraries are present in Docker container.

**What it checks:**
- PostgreSQL libraries
- ML libraries (XGBoost, LightGBM, CatBoost)
- GPU libraries (CUDA, ROCm, Metal)
- Python dependencies
- System libraries

**Usage:**
```bash
./scripts/verify_neurondb_docker_dependencies.sh

# Check specific container
./scripts/verify_neurondb_docker_dependencies.sh neurondb-cpu
```

---

### Maintenance Scripts

#### `export_to_ollama.sh`

Export NeuronDB models to Ollama format.

**Usage:**
```bash
# Export specific model
./scripts/export_to_ollama.sh my_model

# Export all models
./scripts/export_to_ollama.sh --all
```

---

#### `load_huggingface_dataset.py`

Load datasets from HuggingFace Hub into NeuronDB.

**Features:**
- Automatic schema detection
- Embedding generation
- Index creation
- Batch loading with progress

**Usage:**
```python
# Load dataset
python scripts/load_huggingface_dataset.py \
    --dataset sentence-transformers/embedding-training-data \
    --split train \
    --limit 10000 \
    --auto-embed \
    --create-indexes
```

**Arguments:**
- `--dataset` - HuggingFace dataset name (required)
- `--split` - Dataset split (default: train)
- `--limit` - Number of rows to load (optional)
- `--auto-embed` - Generate embeddings automatically
- `--create-indexes` - Create HNSW indexes
- `--batch-size` - Batch size for loading (default: 1000)

---

#### `train_postgres_llm.py`

Train PostgreSQL-specific LLM models.

**Usage:**
```python
python scripts/train_postgres_llm.py \
    --model gpt-3.5-turbo \
    --data sql_queries.json \
    --output postgres-llm-v1
```

---

## 🔧 Script Development

### Writing New Scripts

**Guidelines:**

1. **Naming Convention:** Use `snake_case.sh` for shell scripts
2. **Shebang:** Always use `#!/bin/bash` or `#!/usr/bin/env python3`
3. **Documentation:** Include header comment with description and usage
4. **Error Handling:** Use `set -e` and proper error messages
5. **Configurability:** Use environment variables for configuration
6. **Testing:** Test on clean installation before committing

**Template:**

```bash
#!/bin/bash
#
# Script Name: my_script.sh
# Description: Brief description of what the script does
# Usage: ./my_script.sh [options]
# Author: Your Name
# Date: 2025-01-30

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# Configuration
DEFAULT_VALUE="${ENV_VAR:-default}"

# Functions
log_info() {
    echo -e "${BLUE}INFO:${NC} $1"
}

log_success() {
    echo -e "${GREEN}SUCCESS:${NC} $1"
}

log_error() {
    echo -e "${RED}ERROR:${NC} $1" >&2
}

# Main script logic
main() {
    log_info "Starting script..."
    # Your code here
    log_success "Script completed!"
}

# Run main function
main "$@"
```

---

### Testing Scripts

Before committing new scripts:

```bash
# Test on clean install
docker compose down -v
docker compose up -d
./scripts/your_script.sh

# Test error handling
./scripts/your_script.sh --invalid-option

# Test with different configurations
ENV_VAR=value1 ./scripts/your_script.sh
ENV_VAR=value2 ./scripts/your_script.sh
```

---

## 📊 Script Statistics

| Category | Count |
|----------|-------|
| **Setup Scripts** | 3 |
| **Testing Scripts** | 5 |
| **Docker Scripts** | 6 |
| **Maintenance Scripts** | 5 |
| **Total Scripts** | 19 |

---

## 🆘 Troubleshooting

### Common Issues

#### Script Permission Denied

```bash
# Make script executable
chmod +x scripts/script_name.sh
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
```

#### Docker Container Not Found

```bash
# List running containers
docker compose ps

# Start services
docker compose up -d

# Check logs
docker compose logs neurondb
```

---

## 🤝 Contributing

### Adding New Scripts

1. Create script in `scripts/` directory
2. Follow naming and style guidelines
3. Add documentation header
4. Test thoroughly
5. Update this README
6. Submit pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for full guidelines.

---

## 📝 Additional Documentation

- **[Main README](../README.md)** - Project overview
- **[Quick Start](../QUICKSTART.md)** - Get started quickly
- **[Documentation Index](../DOCUMENTATION.md)** - Complete documentation
- **[Deployment Guide](../NeuronAgent/docs/DEPLOYMENT.md)** - Production deployment

---

## 🔗 Related Resources

- **Setup Guide:** [SETUP_SCRIPT_README.md](SETUP_SCRIPT_README.md)
- **Docker Guide:** [../dockers/README.md](../dockers/README.md)
- **Testing Guide:** [../NeuronAgent/TESTING.md](../NeuronAgent/TESTING.md)

---

**Last Updated:** 2025-01-30  
**Scripts Version:** 1.0.0
