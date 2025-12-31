# Components

This repository contains four primary components that share the same PostgreSQL database.

## Summary

| Component | Code location | Docker Compose service(s) | Default port(s) | Primary docs |
|---|---|---|---|---|
| NeuronDB (Postgres extension) | `NeuronDB/` | `neurondb`, `neurondb-cuda`, `neurondb-rocm`, `neurondb-metal` | 5433 (cpu), 5434 (cuda), 5435 (rocm), 5436 (metal) | `Docs/components/neurondb.md` |
| NeuronAgent (service) | `NeuronAgent/` | `neuronagent` (and GPU variants) | 8080 | `Docs/components/neuronagent.md` |
| NeuronMCP (MCP server) | `NeuronMCP/` | `neuronmcp` (and GPU variants) | stdio (container runs with tty/stdin) | `Docs/components/neuronmcp.md` |
| NeuronDesktop (API + UI) | `NeuronDesktop/` | `neurondesk-api`, `neurondesk-frontend` | 8081 (api), 3000 (ui) | `Docs/components/neurondesktop.md` |

## Notes

- Canonical orchestration: repo-root `docker-compose.yml`.
- Docker files by component: `dockers/`.
