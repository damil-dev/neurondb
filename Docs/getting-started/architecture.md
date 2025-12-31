# Simple Architecture

This is a newcomer-friendly explanation of how the major components fit together.

## Components

- **Postgres + NeuronDB (`NeuronDB/`)**: where data lives; provides vector types, operators, indexes, and SQL APIs.
- **NeuronAgent (`NeuronAgent/`)**: service layer for agent workflows and higher-level operations.
- **NeuronMCP (`NeuronMCP/`)**: MCP server exposing tools to LLM clients.
- **NeuronDesktop (`NeuronDesktop/`)**: local UI + API for managing setup and running queries.

## Typical usage patterns

### Pattern A: “Just the DB”
- Install extension
- Store embeddings
- Create an index
- Run vector search queries

### Pattern B: DB + Agent
- Use the Agent service to orchestrate ingestion and retrieval workflows
- Keep DB as the source of truth

### Pattern C: DB + MCP
- Connect an LLM client to MCP server
- Use database tools (schemas, queries, indexing) via tool calls

### Pattern D: Desktop (local development)
- Use the Desktop UI to set up and interact with the stack locally

## Where to look next

- Setup: `Docs/getting-started/simple-start.md`
- Troubleshooting: `Docs/getting-started/troubleshooting.md`


