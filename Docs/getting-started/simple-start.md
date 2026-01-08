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

1. **From repo root**, start all services:
   ```bash
   # Start all services (CPU profile, default)
   docker compose up -d
   
   # Or start just NeuronDB extension
   docker compose up -d neurondb
   
   # Or start with GPU support (CUDA)
   docker compose --profile cuda up -d
   ```

2. **Wait for services to be healthy** (30-60 seconds):
   ```bash
   docker compose ps
   ```
   
   All services should show "healthy" status.

3. **Verify** Postgres is reachable and extension is installed:
   ```bash
   # Create extension (if not already created)
   docker compose exec neurondb psql -U neurondb -d neurondb -c "CREATE EXTENSION IF NOT EXISTS neurondb;"
   
   # Check version
   docker compose exec neurondb psql -U neurondb -d neurondb -c "SELECT neurondb.version();"
   ```

**Expected output:** `2.0` (or current version)

4. **Test with a simple vector query:**
   ```bash
   docker compose exec neurondb psql -U neurondb -d neurondb <<EOF
   CREATE TABLE test_vectors (
     id SERIAL PRIMARY KEY,
     name TEXT,
     embedding vector(3)
   );
   INSERT INTO test_vectors (name, embedding) VALUES
     ('apple', '[1.0, 0.0, 0.0]'::vector),
     ('banana', '[0.0, 1.0, 0.0]'::vector);
   SELECT name, embedding FROM test_vectors;
   DROP TABLE test_vectors;
   EOF
   ```

**Expected output:**
```
 name  |   embedding    
-------+----------------
 apple | [1,0,0]
 banana | [0,1,0]
(2 rows)
```

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

4. **Verify installation:**
   ```sql
   SELECT neurondb.version();
   ```
   
   **Expected output:** `2.0`

5. **Test** with a basic query or load sample data from [`examples/`](../../examples/):
   ```sql
   -- Quick test
   CREATE TABLE test (id SERIAL, vec vector(3));
   INSERT INTO test (vec) VALUES ('[1,2,3]'::vector);
   SELECT * FROM test;
   DROP TABLE test;
   ```

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


