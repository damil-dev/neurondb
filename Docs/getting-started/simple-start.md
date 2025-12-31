# Simple Start

**Goal:** Get a working local environment with minimal friction.

## Choose your path

**Pick the method that works best for you:**

| Method | Best For | Time | Difficulty |
|--------|----------|------|------------|
| **Docker** (recommended) | Fastest start | 5 min | Easy |
| **Native build** | Custom setup | 30+ min | Advanced |

## Docker quickstart (typical)

**Prerequisites checklist:**

- [ ] Docker 20.10+ installed
- [ ] Docker Compose 2.0+ installed
- [ ] 4GB+ RAM available

**Steps:**

1. **From repo root**, run:
   ```bash
   docker compose up -d
   ```

2. **Verify** Postgres is reachable:
   ```bash
   docker compose exec neurondb psql -U neurondb -d neurondb -c "SELECT neurondb.version();"
   ```

**Expected output:** `1.0.0` (or current version)

## Native quickstart (outline)

<details>
<summary><strong>Native Installation Steps</strong></summary>

**For advanced users only** - Requires PostgreSQL development headers

1. **Build the extension** in `NeuronDB/` (see [`NeuronDB/INSTALL.md`](../../NeuronDB/INSTALL.md))
2. **Install** it into your Postgres `shared_preload_libraries` / extension directory
3. **Create extension:**
   ```sql
   CREATE EXTENSION neurondb;
   ```
4. **Test** with a basic query or load sample data from [`examples/`](../../examples/)

</details>

## Next steps

**Continue your journey:**

- [ ] Read [`architecture.md`](architecture.md) to understand the moving parts
- [ ] Try examples from [`examples/`](../../examples/)
- [ ] Explore the [complete documentation](../../DOCUMENTATION.md)
- [ ] If something fails, check [`troubleshooting.md`](troubleshooting.md)

---

<details>
<summary><strong>Quick Tips</strong></summary>

- **Docker is recommended** for the easiest setup
- **Read the architecture guide** to understand how components work together
- **Check troubleshooting** if you encounter issues
- **Start simple** - get it running first, then explore advanced features

</details>


