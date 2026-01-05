# NeuronDB — PostgreSQL AI Ecosystem

<div align="center">
  <img src="neurondb.png" alt="NeuronDB" width="360" />

  <p>
    <a href="https://www.postgresql.org/">
      <img alt="PostgreSQL 16/17/18" src="https://img.shields.io/badge/PostgreSQL-16%2C17%2C18-blue.svg" />
    </a>
    <a href="LICENSE">
      <img alt="License: Proprietary" src="https://img.shields.io/badge/license-proprietary-red.svg" />
    </a>
    <a href="https://www.neurondb.ai/docs">
      <img alt="Docs" src="https://img.shields.io/badge/docs-neurondb.ai-brightgreen.svg" />
    </a>
  </p>

  <p><strong>Vector search, embeddings, and ML primitives inside PostgreSQL</strong> — plus optional services for <strong>agents</strong>, <strong>MCP</strong>, and a <strong>desktop UI</strong>.</p>
</div>

> [!TIP]
> New here? Start with [`Docs/getting-started/simple-start.md`](Docs/getting-started/simple-start.md) or jump straight to [`QUICKSTART.md`](QUICKSTART.md).

## Table of contents

- [What you can build](#what-you-can-build)
- [Architecture](#architecture)
- [Quick start (Docker)](#quick-start-docker)
- [Service URLs & ports](#service-urls--ports)
- [Choose your path](#choose-your-path)
- [Repo layout](#repo-layout)
- [Docs & examples](#docs--examples)
- [Benchmarks](#benchmarks)
- [GPU profiles (CUDA / ROCm / Metal)](#gpu-profiles-cuda--rocm--metal)
- [Contributing / security / license](#contributing--security--license)
- [Project statistics](#project-statistics)

## What you can build

- **Semantic / hybrid search**: vectors + SQL
- **RAG**: store, retrieve, and serve context
- **Agents**: API runtime backed by Postgres memory
- **MCP integrations**: MCP clients talking to NeuronDB via tools/resources

## Architecture

```mermaid
flowchart LR
  subgraph DB[NeuronDB (PostgreSQL)]
    EXT[NeuronDB extension]
  end
  AG[NeuronAgent] -->|SQL / Postgres| DB
  MCP[NeuronMCP] -->|tools/resources| DB
  UI[NeuronDesktop UI] --> API[NeuronDesktop API]
  API -->|SQL / Postgres| DB
```

> [!NOTE]
> The root `docker-compose.yml` brings up the ecosystem services together. You can also run each component independently (see component READMEs).

## Quick start (Docker)

```bash
docker compose up -d
./scripts/health-check.sh
```

<details>
<summary><strong>Prerequisites checklist</strong></summary>

- [ ] Docker 20.10+ installed
- [ ] Docker Compose 2.0+ installed
- [ ] 4GB+ RAM available
- [ ] Ports 5433, 8080, 8081, 3000 available

</details>

> [!IMPORTANT]
> Prefer a step-by-step guide? See [`QUICKSTART.md`](QUICKSTART.md).

## Service URLs & ports

| Service | How to reach it |
|---|---|
| NeuronDB (PostgreSQL) | `postgresql://neurondb:neurondb@localhost:5433/neurondb` |
| NeuronAgent | `http://localhost:8080/health` |
| NeuronDesktop UI | `http://localhost:3000` |
| NeuronDesktop API | `http://localhost:8081/health` |

## Choose your path

| You are... | Start here |
|---|---|
| New to the project | [`DOCUMENTATION.md`](DOCUMENTATION.md) — complete documentation index |
| Want the easiest walkthrough | [`Docs/getting-started/simple-start.md`](Docs/getting-started/simple-start.md) — beginner-friendly guide |
| Comfortable with Docker/CLI | [`QUICKSTART.md`](QUICKSTART.md) — technical quick start |
| Looking for a specific component | See the component READMEs below |

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

## Docs & examples

**Docs**

- **Complete documentation index**: [`DOCUMENTATION.md`](DOCUMENTATION.md)
- **Quick start guide**: [`QUICKSTART.md`](QUICKSTART.md)
- **Official docs**: [`neurondb.ai/docs`](https://www.neurondb.ai/docs)

**Examples**

- [Examples README](examples/README.md) — complete examples collection
- [Semantic Search](examples/semantic-search-docs/) — document search
- [RAG Chatbot](examples/rag-chatbot-pdfs/) — PDF-based RAG
- [Agent Tools](examples/agent-tools/) — agent integrations
- [MCP Integration](examples/mcp-integration/) — MCP client setup

## Benchmarks

NeuronDB includes a comprehensive benchmark suite to evaluate vector search, hybrid search, and RAG performance.

### Quick start

Run all benchmarks:

```bash
cd NeuronDB/benchmark
./run_bm.sh
```

This will:
- ✅ Check PostgreSQL connection
- ✅ Run **Vector** benchmark (SIFT-128 dataset, HNSW index)
- ✅ Run **Hybrid** benchmark (BEIR nfcorpus dataset)
- ✅ Run **RAG** benchmark (database verification)

### Benchmark suite

| Benchmark | Purpose | Datasets | Metrics |
|---|---|---|---|
| **Vector** | Vector similarity search performance | SIFT-128, GIST-960, GloVe-100 | QPS, Recall, Latency (avg, p50, p95, p99) |
| **Hybrid** | Combined vector + full-text search | BEIR (nfcorpus, msmarco, etc.) | NDCG, MAP, Recall, Precision |
| **RAG** | End-to-end RAG pipeline quality | MTEB, BEIR, RAGAS | Faithfulness, Relevancy, Context Precision |

### Sample results

**Vector Benchmark** (SIFT-128-Euclidean, 1M vectors, HNSW index):
- **Recall@10**: 100.00%
- **QPS**: ~1.9 queries/second
- **Avg Latency**: ~520ms
- **P95 Latency**: ~545ms

**Test Configuration**:
- Dataset: SIFT-128-Euclidean (1,000,000 training vectors, 10,000 test queries)
- Index: HNSW (m=16, ef_construction=200, ef_search=100)
- K-value: 10
- Vector dimension: 128

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
<summary><strong>Stats snapshot</strong></summary>

- **473 SQL functions** in NeuronDB extension
- **52+ ML algorithms** supported
- **100+ MCP tools** available
- **4 integrated components** working together
- **3 PostgreSQL versions** supported (16, 17, 18)
- **4 GPU platforms** supported (CPU, CUDA, ROCm, Metal)

</details>
