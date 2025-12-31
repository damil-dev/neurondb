# Quick Start

This guide is a short, code-accurate path to verifying the system works.

## Prerequisites

- Docker and Docker Compose plugin (`docker compose`)
- `psql` installed locally (or use `docker compose exec` to run `psql` inside the container)

## 1) Start NeuronDB (CPU profile)

From the repository root:

```bash
docker compose up -d neurondb
```

## 2) Create the extension and verify it loads

```bash
PGPASSWORD=neurondb psql -h localhost -p 5433 -U neurondb -d neurondb   -c "CREATE EXTENSION IF NOT EXISTS neurondb; SELECT neurondb.version();"
```

## 3) Minimal vector table, index, and query

NeuronDB defines `vector`, operator classes (for example `vector_l2_ops`), and the `<->` operator in `NeuronDB/neurondb--1.0.sql`.

```sql
CREATE TABLE documents (
  id bigserial PRIMARY KEY,
  embedding vector(3)
);

INSERT INTO documents (embedding) VALUES
  ('[1,0,0]'),
  ('[0,1,0]'),
  ('[0,0,1]');

CREATE INDEX documents_embedding_hnsw_idx
  ON documents
  USING hnsw (embedding vector_l2_ops);

SELECT id, embedding <-> '[1,0,0]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 3;
```

## 4) Start NeuronAgent and check health (optional)

```bash
docker compose up -d neuronagent
curl -sS http://localhost:8080/health
```

## 5) Start NeuronDesktop (optional)

```bash
docker compose up -d neurondesk-api neurondesk-frontend
```

- UI: `http://localhost:3000/`
- API: `http://localhost:8081/health`

## 6) NeuronMCP (optional)

NeuronMCP runs as a container in this repo’s docker compose. Start it with:

```bash
docker compose up -d neuronmcp
```

For client configuration and tool details, see `NeuronMCP/readme.md` and `NeuronMCP/TOOLS_REFERENCE.md`.

## Next steps

- Docker and profiles: `dockers/readme.md`
- NeuronDB extension docs: `NeuronDB/docs/`
- NeuronAgent OpenAPI: `NeuronAgent/openapi/openapi.yaml`
