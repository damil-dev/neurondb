# NeuronDB — PostgreSQL AI Ecosystem

<p align="center">
  <img src="neurondb.png" alt="NeuronDB" width="360" />
</p>

<p align="center">
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

> Vector search, embeddings, and ML primitives inside PostgreSQL — plus optional services for **agents**, **MCP**, and a **desktop UI**.

## What you can build

**NeuronDB enables AI-powered applications directly in PostgreSQL**

- **Semantic / hybrid search** in Postgres (vectors + SQL)
- **RAG** (store, retrieve, and serve context)
- **Agents** with an API runtime backed by Postgres memory
- **MCP integrations** (MCP clients talking to NeuronDB via tools/resources)

## Quick start (Docker, recommended)

**Get running in under 5 minutes**

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

Prefer a step-by-step guide? See [`QUICKSTART.md`](QUICKSTART.md)

<details>
<summary><strong>Service URLs & ports</strong></summary>

| Service | How to reach it |
|---|---|
| NeuronDB (PostgreSQL) | `postgresql://neurondb:neurondb@localhost:5433/neurondb` |
| NeuronAgent | `http://localhost:8080/health` |
| NeuronDesktop UI | `http://localhost:3000` |
| NeuronDesktop API | `http://localhost:8081/health` |

</details>

## Choose your path

**Not sure where to start?** Pick the option that best describes you:

| You are... | Start here |
|---|---|
| New to the project | [`DOCUMENTATION.md`](DOCUMENTATION.md) - Complete documentation index |
| Want the easiest walkthrough | [`Docs/getting-started/simple-start.md`](Docs/getting-started/simple-start.md) - Beginner-friendly guide |
| Comfortable with Docker/CLI | [`QUICKSTART.md`](QUICKSTART.md) - Technical quick start |
| Looking for a specific component | See the component READMEs below |

## Repo layout

| Component | Path | What it is |
|---|---|---|
| NeuronDB | `NeuronDB/` | PostgreSQL extension with vector search, 52+ ML algorithms, GPU acceleration (CUDA/ROCm/Metal), embeddings, RAG pipeline, hybrid search, and background workers |
| NeuronAgent | `NeuronAgent/` | Agent runtime + REST/WebSocket API (Go) with multi-agent collaboration, workflow engine, HITL, 20+ tools, hierarchical memory, budget management, and evaluation framework |
| NeuronMCP | `NeuronMCP/` | MCP server for MCP-compatible clients (Go) with 100+ tools and resources |
| NeuronDesktop | `NeuronDesktop/` | Web UI + API for the ecosystem providing unified interface |

## Documentation

- **Complete documentation index**: `DOCUMENTATION.md` (start here for full docs)
- **Quick start guide**: `QUICKSTART.md` (get running in minutes)
- **Simple start**: `Docs/getting-started/simple-start.md` (beginner-friendly)
- **Official docs**: [https://www.neurondb.ai/docs](https://www.neurondb.ai/docs)

## Component READMEs

**Explore individual components:**

- [`NeuronDB/README.md`](NeuronDB/README.md) - PostgreSQL extension
- [`NeuronAgent/README.md`](NeuronAgent/README.md) - Agent runtime
- [`NeuronMCP/README.md`](NeuronMCP/README.md) - MCP server
- [`NeuronDesktop/README.md`](NeuronDesktop/README.md) - Web UI

## Examples

**Ready to build?** Check out our runnable examples:

- [Examples README](examples/README.md) - Complete examples collection
- [Semantic Search](examples/semantic-search-docs/) - Document search
- [RAG Chatbot](examples/rag-chatbot-pdfs/) - PDF-based RAG
- [Agent Tools](examples/agent-tools/) - Agent integrations
- [MCP Integration](examples/mcp-integration/) - MCP client setup

<details>
<summary><strong>GPU profiles (CUDA / ROCm / Metal)</strong></summary>

The root `docker-compose.yml` supports profiles:

- CPU (default):
  - `docker compose up -d`
- CUDA:
  - `docker compose --profile cuda up -d`
- ROCm:
  - `docker compose --profile rocm up -d`

Ports differ per profile (see `env.example`):

- CPU: `POSTGRES_PORT=5433`
- CUDA: `POSTGRES_CUDA_PORT=5434`
- ROCm: `POSTGRES_ROCM_PORT=5435`

</details>

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

## Contributing / Security / License

**Want to contribute?**

- **Contributing**: [`CONTRIBUTING.md`](CONTRIBUTING.md) - How to contribute
- **Security**: [`SECURITY.md`](SECURITY.md) - Security policy
- **License**: [`LICENSE`](LICENSE) - Proprietary license

---

<details>
<summary><strong>Project Statistics</strong></summary>

- **473 SQL functions** in NeuronDB extension
- **52+ ML algorithms** supported
- **100+ MCP tools** available
- **4 integrated components** working together
- **3 PostgreSQL versions** supported (16, 17, 18)
- **4 GPU platforms** supported (CPU, CUDA, ROCm, Metal)

</details>
