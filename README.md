# NeuronDB — PostgreSQL AI ecosystem

<div align="center">
  <img src="neurondb.png" alt="NeuronDB" width="360" />

  <p>
    <a href="https://www.postgresql.org/">
      <img alt="PostgreSQL 16/17/18" src="https://img.shields.io/badge/PostgreSQL-16%2C17%2C18-blue.svg" />
    </a>
    <a href="https://github.com/neurondb/neurondb/actions/workflows/neurondb-build-and-test.yml">
      <img alt="CI: NeuronDB" src="https://github.com/neurondb/neurondb/actions/workflows/neurondb-build-and-test.yml/badge.svg?branch=main" />
    </a>
    <a href="https://github.com/neurondb/neurondb/actions/workflows/integration-tests-full-ecosystem.yml">
      <img alt="CI: Integration" src="https://github.com/neurondb/neurondb/actions/workflows/integration-tests-full-ecosystem.yml/badge.svg?branch=main" />
    </a>
    <a href="https://github.com/neurondb/neurondb/actions/workflows/security-scan.yml">
      <img alt="Security scan" src="https://github.com/neurondb/neurondb/actions/workflows/security-scan.yml/badge.svg?branch=main" />
    </a>
    <a href="https://github.com/neurondb/neurondb/actions/workflows/publish-all-container-images.yml">
      <img alt="Docker Images" src="https://github.com/neurondb/neurondb/actions/workflows/publish-all-container-images.yml/badge.svg?branch=main" />
    </a>
  </p>
  <p>
    <a href="#gpu-profiles-cuda--rocm--metal">
      <img alt="GPU Backends" src="https://img.shields.io/badge/GPU-CUDA%20%7C%20ROCm%20%7C%20Metal-green.svg" />
    </a>
    <a href="LICENSE">
      <img alt="License: Proprietary" src="https://img.shields.io/badge/license-proprietary-red.svg" />
    </a>
    <a href="https://www.neurondb.ai/docs">
      <img alt="Docs" src="https://img.shields.io/badge/docs-neurondb.ai-brightgreen.svg" />
    </a>
  </p>

  <p><strong>Vector search, embeddings, and ML primitives in PostgreSQL</strong>, with optional services for <strong>agents</strong>, <strong>MCP</strong>, and a <strong>desktop UI</strong>.</p>
</div>

> [!TIP]
> New here? Start with [`Docs/getting-started/simple-start.md`](Docs/getting-started/simple-start.md) or jump to [`QUICKSTART.md`](QUICKSTART.md).

## Table of contents

- [What you can build](#what-you-can-build)
- [Architecture](#architecture)
- [Installation](#installation)
  - [Quick start (Docker)](#quick-start-docker)
  - [Native install](#native-install)
  - [Minimal mode (extension only)](#minimal-mode-extension-only)
- [Service URLs & ports](#service-urls--ports)
- [Documentation](#documentation)
- [Repo layout](#repo-layout)
- [Benchmarks](#benchmarks)
- [GPU profiles (CUDA / ROCm / Metal)](#gpu-profiles-cuda--rocm--metal)
- [Contributing / security / license](#contributing--security--license)
- [Project statistics](#project-statistics)

## What you can build

- **Semantic & hybrid search**: vector similarity + SQL filters + full-text search
- **RAG pipelines**: store, retrieve, and serve context with Postgres-native primitives
- **Agent backends**: durable memory and tool execution backed by PostgreSQL
- **MCP integrations**: MCP clients connecting to NeuronDB via tools/resources

## Architecture

```mermaid
flowchart LR
  subgraph DB["NeuronDB PostgreSQL"]
    EXT["NeuronDB extension"]
  end
  AG["NeuronAgent"] -->|SQL| DB
  MCP["NeuronMCP"] -->|tools/resources| DB
  UI["NeuronDesktop UI"] --> API["NeuronDesktop API"]
  API -->|SQL| DB
```

> [!NOTE]
> The root `docker-compose.yml` starts the ecosystem services together. You can also run each component independently (see component READMEs).

## Installation

### Quick start (Docker)

```bash
docker compose up -d
./scripts/health-check.sh
```

<details>
<summary><strong>Prerequisites checklist</strong></summary>

- [ ] Docker 20.10+ installed
- [ ] Docker Compose 2.0+ installed
- [ ] 4 GB+ RAM available
- [ ] Ports 5433, 8080, 8081, 3000 available

</details>

> [!IMPORTANT]
> Prefer a step-by-step guide? See [`QUICKSTART.md`](QUICKSTART.md).

### Native install

Install the NeuronDB extension directly into your existing PostgreSQL installation.

<details>
<summary><strong>Build and install steps</strong></summary>

**Prerequisites:**
- PostgreSQL 16, 17, or 18 development headers
- C compiler (gcc or clang)
- Make

**Build:**
```bash
cd NeuronDB
make
sudo make install
```

**Enable extension:**
```sql
CREATE EXTENSION neurondb;
```

**Configure (if needed):**

Some features require preloading. Add to `postgresql.conf`:
```ini
shared_preload_libraries = 'neurondb'
```

Then restart PostgreSQL:
```bash
sudo systemctl restart postgresql
```

**Configuration parameters (GUCs):**
```ini
# Vector index settings
neurondb.hnsw_ef_search = 40          # HNSW search quality
neurondb.enable_seqscan = on          # Allow sequential scans

# Memory settings
neurondb.maintenance_work_mem = 256MB # Index build memory
```

**Upgrade path:**
```sql
-- Check current version
SELECT extversion FROM pg_extension WHERE extname = 'neurondb';

-- Upgrade to latest
ALTER EXTENSION neurondb UPDATE;
```

</details>

For detailed installation instructions, see [`NeuronDB/INSTALL.md`](NeuronDB/INSTALL.md).

### Minimal mode (extension only)

Use NeuronDB as a PostgreSQL extension only, without the Agent, MCP, or Desktop services.

**Benefits:**
- ✅ No extra services or ports
- ✅ Minimal resource footprint
- ✅ Full vector search, ML algorithms, and embeddings
- ✅ Works with any PostgreSQL client

**Installation:**

Follow the [Native install](#native-install) steps above. That's it! You now have vector search and ML capabilities in PostgreSQL.

**Usage:**
```sql
-- Create a table with vectors
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding VECTOR(1536)
);

-- Create HNSW index
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);

-- Vector similarity search
SELECT id, content
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;
```

No additional services, ports, or configuration required!

## Service URLs & ports

| Service | How to reach it |
|---|---|
| NeuronDB (PostgreSQL) | `postgresql://neurondb:neurondb@localhost:5433/neurondb` |
| NeuronAgent | `http://localhost:8080/health` |
| NeuronDesktop UI | `http://localhost:3000` |
| NeuronDesktop API | `http://localhost:8081/health` |

## Documentation

- **Start here**: [`DOCUMENTATION.md`](DOCUMENTATION.md) (documentation index)
- **Beginner walkthrough**: [`Docs/getting-started/simple-start.md`](Docs/getting-started/simple-start.md)
- **Technical quick start**: [`QUICKSTART.md`](QUICKSTART.md)
- **Official docs**: [`neurondb.ai/docs`](https://www.neurondb.ai/docs)

### Module-wise Documentation

<details>
<summary><strong>NeuronDB documentation</strong></summary>

- **Getting Started**: [`installation.md`](NeuronDB/docs/getting-started/installation.md) • [`quickstart.md`](NeuronDB/docs/getting-started/quickstart.md)
- **Vector Search**: [`indexing.md`](NeuronDB/docs/vector-search/indexing.md) • [`distance-metrics.md`](NeuronDB/docs/vector-search/distance-metrics.md) • [`quantization.md`](NeuronDB/docs/vector-search/quantization.md)
- **Hybrid Search**: [`overview.md`](NeuronDB/docs/hybrid-search/overview.md) • [`multi-vector.md`](NeuronDB/docs/hybrid-search/multi-vector.md) • [`faceted-search.md`](NeuronDB/docs/hybrid-search/faceted-search.md)
- **RAG Pipeline**: [`overview.md`](NeuronDB/docs/rag/overview.md) • [`document-processing.md`](NeuronDB/docs/rag/document-processing.md) • [`llm-integration.md`](NeuronDB/docs/rag/llm-integration.md)
- **ML Algorithms**: [`clustering.md`](NeuronDB/docs/ml-algorithms/clustering.md) • [`classification.md`](NeuronDB/docs/ml-algorithms/classification.md) • [`regression.md`](NeuronDB/docs/ml-algorithms/regression.md)
- **ML Embeddings**: [`embedding-generation.md`](NeuronDB/docs/ml-embeddings/embedding-generation.md) • [`model-management.md`](NeuronDB/docs/ml-embeddings/model-management.md)
- **GPU Support**: [`cuda-support.md`](NeuronDB/docs/gpu/cuda-support.md) • [`rocm-support.md`](NeuronDB/docs/gpu/rocm-support.md) • [`metal-support.md`](NeuronDB/docs/gpu/metal-support.md)
- **Operations**: [`troubleshooting.md`](NeuronDB/docs/troubleshooting.md) • [`configuration.md`](NeuronDB/docs/configuration.md) • [`playbook.md`](NeuronDB/docs/operations/playbook.md)

</details>

<details>
<summary><strong>NeuronAgent documentation</strong></summary>

- **Architecture**: [`ARCHITECTURE.md`](NeuronAgent/docs/ARCHITECTURE.md)
- **API Reference**: [`API.md`](NeuronAgent/docs/API.md)
- **CLI Guide**: [`CLI_GUIDE.md`](NeuronAgent/docs/CLI_GUIDE.md)
- **Connectors**: [`CONNECTORS.md`](NeuronAgent/docs/CONNECTORS.md)
- **Deployment**: [`DEPLOYMENT.md`](NeuronAgent/docs/DEPLOYMENT.md)
- **Troubleshooting**: [`TROUBLESHOOTING.md`](NeuronAgent/docs/TROUBLESHOOTING.md)

</details>

<details>
<summary><strong>NeuronMCP documentation</strong></summary>

- **Setup Guide**: [`NEURONDB_MCP_SETUP.md`](NeuronMCP/docs/NEURONDB_MCP_SETUP.md)
- **Tool & Resource Catalog**: [`tool-resource-catalog.md`](NeuronMCP/docs/tool-resource-catalog.md)
- **Examples**: [`README.md`](NeuronMCP/docs/examples/README.md) • [`example-transcript.md`](NeuronMCP/docs/examples/example-transcript.md)

</details>

<details>
<summary><strong>NeuronDesktop documentation</strong></summary>

- **API Reference**: [`API.md`](NeuronDesktop/docs/API.md)
- **Deployment**: [`DEPLOYMENT.md`](NeuronDesktop/docs/DEPLOYMENT.md)
- **Integration**: [`INTEGRATION.md`](NeuronDesktop/docs/INTEGRATION.md)
- **NeuronAgent Usage**: [`NEURONAGENT_USAGE.md`](NeuronDesktop/docs/NEURONAGENT_USAGE.md)
- **NeuronMCP Setup**: [`NEURONMCP_SETUP.md`](NeuronDesktop/docs/NEURONMCP_SETUP.md)

</details>

## Repo layout

| Component | Path | What it is |
|---|---|---|
| NeuronDB | `NeuronDB/` | PostgreSQL extension with vector search, ML algorithms, GPU acceleration (CUDA/ROCm/Metal), embeddings, RAG pipeline, hybrid search, and background workers |
| NeuronAgent | `NeuronAgent/` | Agent runtime + REST/WebSocket API (Go) with multi-agent collaboration, workflow engine, HITL, tools, memory, budget management, and evaluation framework |
| NeuronMCP | `NeuronMCP/` | MCP server for MCP-compatible clients (Go) with tools and resources |
| NeuronDesktop | `NeuronDesktop/` | Web UI + API for the ecosystem providing a unified interface |

### Component READMEs

- [`NeuronDB/README.md`](NeuronDB/README.md)
- [`NeuronAgent/README.md`](NeuronAgent/README.md)
- [`NeuronMCP/README.md`](NeuronMCP/README.md)
- [`NeuronDesktop/README.md`](NeuronDesktop/README.md)

### Examples

- [Examples index](examples/README.md)
- [Semantic search docs example](examples/semantic-search-docs/)
- [RAG chatbot (PDFs) example](examples/rag-chatbot-pdfs/)
- [Agent tools example](examples/agent-tools/)
- [MCP integration example](examples/mcp-integration/)

## Benchmarks

NeuronDB includes a benchmark suite to evaluate vector search, hybrid search, and RAG performance.

### Quick start

Run all benchmarks:

```bash
cd NeuronDB/benchmark
./run_bm.sh
```

This validates connectivity and runs the vector/hybrid/RAG benchmark groups.

### Benchmark suite

| Benchmark | Purpose | Datasets | Metrics |
|---|---|---|---|
| **Vector** | Vector similarity search performance | SIFT-128, GIST-960, GloVe-100 | QPS, Recall, Latency (avg, p50, p95, p99) |
| **Hybrid** | Combined vector + full-text search | BEIR (nfcorpus, msmarco, etc.) | NDCG, MAP, Recall, Precision |
| **RAG** | End-to-end RAG pipeline quality | MTEB, BEIR, RAGAS | Faithfulness, Relevancy, Context Precision |

<details>
<summary><strong>Run individual benchmarks</strong></summary>

```bash
# Vector benchmark
cd NeuronDB/benchmark/vector
./run_bm.py --prepare --load --run --datasets sift-128-euclidean --max-queries 100

# Hybrid benchmark
cd NeuronDB/benchmark/hybrid
./run_bm.py --prepare --load --run --datasets nfcorpus --model all-MiniLM-L6-v2

# RAG benchmark
cd NeuronDB/benchmark/rag
./run_bm.py --prepare --verify --run --benchmarks mteb
```

</details>

## GPU profiles (CUDA / ROCm / Metal)

The root `docker-compose.yml` supports profiles:

- **CPU (default)**: `docker compose up -d`
- **CUDA**: `docker compose --profile cuda up -d`
- **ROCm**: `docker compose --profile rocm up -d`

Ports differ per profile (see [`env.example`](env.example)):

- **CPU**: `POSTGRES_PORT=5433`
- **CUDA**: `POSTGRES_CUDA_PORT=5434`
- **ROCm**: `POSTGRES_ROCM_PORT=5435`

<details>
<summary><strong>Common Docker commands</strong></summary>

```bash
# Stop everything (keep data)
docker compose down

# Stop everything (delete data volumes)
docker compose down -v

# See status
docker compose ps

# Tail logs
docker compose logs -f neurondb neuronagent neuronmcp neurondesk-api neurondesk-frontend
```

</details>

## Contributing / security / license

- **Contributing**: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- **Security**: [`SECURITY.md`](SECURITY.md)
- **License**: [`LICENSE`](LICENSE) (proprietary)

## Project statistics

<details>
<summary><strong>Stats snapshot (may change)</strong></summary>

- **473 SQL functions** in NeuronDB extension
- **52+ ML algorithms** supported
- **100+ MCP tools** available
- **4 integrated components** working together
- **3 PostgreSQL versions** supported (16, 17, 18)
- **4 GPU platforms** supported (CPU, CUDA, ROCm, Metal)

</details>

<details>
<summary><strong>Platform & version coverage</strong></summary>

| Category | Supported Versions |
|---|---|
| **PostgreSQL** | 16, 17, 18 |
| **Go** | 1.21, 1.22, 1.23, 1.24 |
| **Node.js** | 18 LTS, 20 LTS, 22 LTS |
| **Operating Systems** | Ubuntu 20.04, Ubuntu 22.04, macOS 13 (Ventura), macOS 14 (Sonoma) |
| **Architectures** | linux/amd64, linux/arm64 |

</details>
