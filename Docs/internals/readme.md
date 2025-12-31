# Internals

This section is for deeper dives: internals, performance tuning, deployment patterns, and how to extend the ecosystem.

## Where the code lives

- Extension internals: `NeuronDB/src/`
- Extension headers/APIs: `NeuronDB/include/`
- Agent service: `NeuronAgent/internal/`
- MCP server: `NeuronMCP/internal/`
- Desktop: `NeuronDesktop/`

## Suggested reading (code-anchored)

### Production Deployment
- NeuronAgent: `NeuronAgent/docs/DEPLOYMENT.md`
- NeuronDesktop: `NeuronDesktop/docs/`
- Repo security overview: `SECURITY.md`
- Docker orchestration and profiles: `dockers/readme.md` and `docker-compose.yml`

### Performance & Scaling
- NeuronDB GPU docs: `NeuronDB/docs/gpu/`
- NeuronDB performance docs: `NeuronDB/docs/performance/`

### Architecture & Design
- NeuronAgent architecture: `NeuronAgent/docs/`
- Ecosystem integration: `Docs/ecosystem/integration.md`

### API References
- NeuronDB SQL surface:
  - extension SQL definitions: `NeuronDB/neurondb--1.0.sql`
  - generated API reference doc: `NeuronDB/docs/sql-api.md`
- NeuronAgent OpenAPI: `NeuronAgent/openapi/openapi.yaml`
- NeuronMCP tools reference: `NeuronMCP/TOOLS_REFERENCE.md`

### Development
- Contributing: `CONTRIBUTING.md`
- NeuronAgent testing: `NeuronAgent/TESTING.md`
- NeuronDB stability notes: `NeuronDB/docs/function-stability.md`

### Building from Source
- NeuronDB build: `NeuronDB/INSTALL.md`
- Component build overview: `Docs/getting-started/installation.md#method-2-source-build`

### Custom Integrations
- Integration guide: `Docs/ecosystem/integration.md`

