# NeuronDB — PostgreSQL AI Ecosystem

<p align="center">
  <img src="neurondb.png" alt="NeuronDB" width="360" />
</p>

<p align="center">
  <a href="https://github.com/neurondb/NeurondB/actions/workflows/build-matrix.yml">
    <img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/neurondb/NeurondB/build-matrix.yml?branch=main&label=build" />
  </a>
  <a href="https://github.com/neurondb/NeurondB/actions/workflows/integration-tests.yml">
    <img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/neurondb/NeurondB/integration-tests.yml?branch=main&label=tests" />
  </a>
  <a href="https://codecov.io/gh/neurondb/NeurondB">
    <img alt="Coverage" src="https://img.shields.io/codecov/c/github/neurondb/NeurondB?label=coverage" />
  </a>
  <a href="https://github.com/neurondb/NeurondB/releases">
    <img alt="Version" src="https://img.shields.io/github/v/release/neurondb/NeurondB?label=version&sort=semver" />
  </a>
  <a href="https://www.postgresql.org/">
    <img alt="PostgreSQL 16/17/18" src="https://img.shields.io/badge/PostgreSQL-16%2C17%2C18-blue.svg" />
  </a>
  <a href="https://www.neurondb.ai/docs">
    <img alt="Docs" src="https://img.shields.io/badge/docs-neurondb.ai-brightgreen.svg" />
  </a>
  <a href="LICENSE">
    <img alt="License: Proprietary" src="https://img.shields.io/badge/license-proprietary-red.svg" />
  </a>
</p>

> Vector search, embeddings, and ML primitives inside PostgreSQL — plus optional services for **agents**, **MCP**, and a **desktop UI**.

## What you can build

**NeuronDB enables AI-powered applications directly in PostgreSQL**

- **Semantic / hybrid search** in Postgres (vectors + SQL)
- **RAG** (store, retrieve, and serve context)
- **Agents** with an API runtime backed by Postgres memory
- **MCP integrations** (MCP clients talking to NeuronDB via tools/resources)

## Platform Support

### PostgreSQL Versions

**NeuronDB officially supports:**
- ✅ **PostgreSQL 16** - Full support
- ✅ **PostgreSQL 17** - Full support (recommended)
- ✅ **PostgreSQL 18** - Full support

All versions are tested in CI/CD and supported for production use. The extension is built against each PostgreSQL version's extension API.

### Operating Systems

| OS | Status | Notes |
|---|---|---|
| **Linux** | ✅ Fully Supported | Ubuntu 20.04+, Debian 11+, Rocky Linux 8+, RHEL 8+ |
| **macOS** | ✅ Fully Supported | macOS 13.0+ (Ventura), Apple Silicon (M1/M2/M3) recommended |
| **Windows** | ✅ Supported via Docker | Windows 10/11 with WSL2 or Docker Desktop |

### GPU Acceleration Platforms

| Platform | Status | Requirements |
|---|---|---|
| **CPU** | ✅ Always Available | No additional requirements |
| **CUDA (NVIDIA)** | ✅ Fully Supported | NVIDIA Driver 525.60+, CUDA 11.8+ or 12.0+ |
| **ROCm (AMD)** | ✅ Fully Supported | ROCm 5.7+ or 6.0+, compatible AMD GPUs |
| **Metal (Apple)** | ✅ Fully Supported | macOS 13.0+, Apple Silicon (M1/M2/M3) |

**Note:** GPU acceleration is optional. NeuronDB works fully on CPU-only systems.

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

## System Requirements & Dependencies

### Core Requirements (Docker Installation)

**Minimum system requirements:**
- **Docker**: 20.10+ (with Docker Compose 2.0+)
- **RAM**: 4GB minimum, 8GB+ recommended
- **Disk Space**: 10GB+ for images and data
- **CPU**: 2+ cores recommended
- **Ports**: 5433, 8080, 8081, 3000 available

**Operating Systems Supported:**
| OS | Version | Notes |
|---|---|---|
| **Linux** | Ubuntu 20.04+, Debian 11+, Rocky Linux 8+ | Full support, all GPU backends |
| **macOS** | 13.0+ (Ventura) | Apple Silicon (M1/M2/M3) recommended for Metal GPU |
| **Windows** | 10/11 | Via WSL2 or Docker Desktop |

### GPU Platform Support

NeuronDB supports multiple GPU acceleration backends:

| Platform | GPU Vendor | Driver Requirements | CUDA/ROCm Version | Docker Profile |
|---|---|---|---|---|
| **CPU** | N/A | None | N/A | `default` or `cpu` |
| **CUDA** | NVIDIA | NVIDIA Driver 525.60+ | CUDA 11.8+ or 12.0+ | `cuda` |
| **ROCm** | AMD | ROCm 5.7+ or 6.0+ | ROCm 5.7+ / 6.0+ | `rocm` |
| **Metal** | Apple Silicon | macOS 13.0+ | Built-in (Metal Performance Shaders) | `metal` |

**GPU Driver Setup:**

<details>
<summary><strong>NVIDIA CUDA Setup</strong></summary>

**Requirements:**
- NVIDIA GPU with Compute Capability 6.0+ (Pascal or newer)
- NVIDIA Driver 525.60+ (check: `nvidia-smi`)
- CUDA Toolkit 11.8+ or 12.0+ (optional, for development)
- Docker with NVIDIA Container Toolkit

**Verify installation:**
```bash
# Check driver
nvidia-smi

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

**Usage:**
```bash
docker compose --profile cuda up -d
```

</details>

<details>
<summary><strong>AMD ROCm Setup</strong></summary>

**Requirements:**
- AMD GPU with ROCm support (Radeon RX 6000+, Instinct MI series)
- ROCm 5.7+ or 6.0+ installed
- Docker with ROCm device access

**Verify installation:**
```bash
# Check ROCm
rocm-smi

# Verify devices
ls -la /dev/kfd /dev/dri
```

**Usage:**
```bash
docker compose --profile rocm up -d
```

</details>

<details>
<summary><strong>Apple Metal Setup</strong></summary>

**Requirements:**
- macOS 13.0+ (Ventura or newer)
- Apple Silicon (M1/M2/M3) - Intel Macs not supported for Metal
- Docker Desktop for Mac

**Usage:**
```bash
docker compose --profile metal up -d
```

**Note:** Metal support is built into macOS, no additional drivers needed.

</details>

### Source Build Dependencies

If building from source instead of using Docker:

**PostgreSQL:**
- PostgreSQL 16, 17, or 18
- PostgreSQL development headers (`postgresql-server-dev-*`)

**Build Tools:**
| Tool | Minimum Version | Purpose |
|---|---|---|
| **GCC** | 7.0+ | C/C++ compiler (Linux) |
| **Clang** | 10.0+ | C/C++ compiler (macOS/Linux) |
| **Make** | 3.81+ | Build system |
| **pkg-config** | 0.29+ | Library detection |

**Development Libraries:**
- `libcurl4-openssl-dev` (or `libcurl-devel`) - HTTP operations
- `libssl-dev` (or `openssl-devel`) - Encryption
- `zlib1g-dev` (or `zlib-devel`) - Compression

**Optional ML Libraries (for enhanced ML features):**
- **XGBoost** - Gradient boosting (optional)
- **LightGBM** - Gradient boosting (optional)
- **CatBoost** - Gradient boosting (optional)

**Go Components (NeuronAgent, NeuronMCP, NeuronDesktop):**
- Go 1.21+ (1.23 recommended)
- Node.js 18+ and npm (for NeuronDesktop frontend)

**Python (for examples and tools):**
- Python 3.8+ (3.10+ recommended)
- pip package manager

### Runtime Dependencies

**PostgreSQL Extension (NeuronDB):**
- PostgreSQL 16, 17, or 18 running
- Extension loaded: `CREATE EXTENSION neurondb;`

**NeuronAgent:**
- Go runtime (included in Docker image)
- PostgreSQL connection
- Optional: External LLM API keys (OpenAI, Anthropic, etc.)

**NeuronMCP:**
- Go runtime (included in Docker image)
- PostgreSQL connection
- MCP-compatible client

**NeuronDesktop:**
- Go runtime (backend, included in Docker image)
- Node.js runtime (frontend, included in Docker image)
- PostgreSQL connection

### Optional Dependencies

**For Embeddings:**
- HuggingFace API key (or local model)
- ONNX Runtime (optional, for local inference)

**For Development:**
- Git
- Code editor/IDE
- Testing frameworks (pytest, etc.)

**For Production:**
- Reverse proxy (nginx, Traefik, etc.)
- SSL certificates
- Monitoring tools (Prometheus, Grafana)

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
