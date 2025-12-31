# NeuronAgent

NeuronAgent is a Go service in `NeuronAgent/`.

## What it is

- HTTP server providing APIs for agent workflows.
- Exposes a health endpoint at `/health` (see `NeuronAgent/cmd/agent-server/main.go`).

## Where to look in the code

- Server entrypoint: `NeuronAgent/cmd/agent-server/main.go`
- Internal packages: `NeuronAgent/internal/`
- OpenAPI spec: `NeuronAgent/openapi/openapi.yaml`

## Docker

- Compose service: `neuronagent` (plus GPU-profile variants).
- From repo root: `docker compose up -d neuronagent`

## Minimal verification

- `curl -sS http://localhost:8080/health`
