# NeuronMCP

NeuronMCP is an MCP (Model Context Protocol) server in `NeuronMCP/`.

## What it is

- A JSON-RPC tool server intended for MCP clients.
- The server binary entrypoint is `NeuronMCP/cmd/neurondb-mcp/main.go`.

## Where to look in the code

- Server entrypoint: `NeuronMCP/cmd/neurondb-mcp/main.go`
- Server internals: `NeuronMCP/internal/`
- Tool reference: `NeuronMCP/TOOLS_REFERENCE.md`

## Docker

- Compose service: `neuronmcp` (plus GPU-profile variants).
- From repo root: `docker compose up -d neuronmcp`

## Notes

The compose configuration runs the container with `stdin_open` and `tty` enabled (see `docker-compose.yml`).
