# NeuronDB

NeuronDB is a PostgreSQL extension implemented in `NeuronDB/`.

## What it is

- A Postgres extension that defines types (for example `vector`), operators, and index access methods.
- SQL definitions for the extension live in `NeuronDB/neurondb--1.0.sql`.

## Where to look in the code

- Extension SQL surface: `NeuronDB/neurondb--1.0.sql`
- C/CUDA sources: `NeuronDB/src/`
- Headers: `NeuronDB/include/`
- NeuronDB docs: `NeuronDB/docs/`

## Docker

- Compose services: `neurondb` (cpu), `neurondb-cuda`, `neurondb-rocm`, `neurondb-metal`
- See: `dockers/neurondb/readme.md` and repo-root `docker-compose.yml`.

## Minimal verification

After Postgres is running:

- `CREATE EXTENSION IF NOT EXISTS neurondb;`
- `SELECT neurondb.version();`

(These are defined in `NeuronDB/neurondb--1.0.sql`.)
