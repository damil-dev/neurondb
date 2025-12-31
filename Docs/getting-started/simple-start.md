# Simple Start

Goal: get a working local environment with minimal friction.

## Choose your path

- **Docker (recommended for fastest start)**: use root `docker-compose.yml` or `dockers/` compose files.
- **Native build**: build/install the extension and run Postgres locally.

## Docker quickstart (typical)

1. Confirm you have Docker + Docker Compose.
2. From repo root, look for:
   - `docker-compose.yml`
   - or `dockers/docker-compose.yml`
3. Bring the stack up:
   - `docker compose up -d`
4. Verify Postgres is reachable and extension is installed.

## Native quickstart (outline)

1. Build the extension in `NeuronDB/` (see `NeuronDB/INSTALL.md`).
2. Install it into your Postgres `shared_preload_libraries` / extension directory.
3. `CREATE EXTENSION neurondb;`
4. Run a basic query / load sample data from `examples/`.

## Next steps

- Read `Docs/getting-started/architecture.md` to understand the moving parts.
- If something fails, go to `Docs/getting-started/troubleshooting.md`.


