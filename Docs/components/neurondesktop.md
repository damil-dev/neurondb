# NeuronDesktop Component

Unified web interface for MCP servers, NeuronDB, and NeuronAgent.

## Overview

NeuronDesktop is a comprehensive, production-ready web application that provides a unified interface for managing and interacting with:
- **MCP Servers** - Model Context Protocol servers with tool inspection and testing
- **NeuronDB** - Vector database with semantic search and collection management
- **NeuronAgent** - AI agent runtime with session management

## Key Capabilities

| Feature | Description |
|---------|-------------|
| **Unified Interface** | Single dashboard for all NeuronDB ecosystem components |
| **Real-time Communication** | WebSocket support for live updates |
| **Secure Authentication** | API key-based authentication with rate limiting |
| **Professional UI** | Modern, responsive design with smooth animations |
| **Comprehensive Logging** | Request/response logging with detailed analytics |
| **Metrics & Monitoring** | Built-in metrics collection and health checks |
| **MCP Integration** | Full MCP server integration and testing |
| **Agent Management** | Create and manage AI agents through the UI |

## Documentation

### Local Documentation

- **[Component README](../../NeuronDesktop/README.md)** - Overview and features
- **[API Reference](../../NeuronDesktop/docs/API.md)** - Complete API documentation
- **[Deployment Guide](../../NeuronDesktop/docs/DEPLOYMENT.md)** - Production deployment
- **[Integration Guide](../../NeuronDesktop/docs/INTEGRATION.md)** - Component integration
- **[NeuronAgent Usage](../../NeuronDesktop/docs/NEURONAGENT_USAGE.md)** - Using NeuronAgent in NeuronDesktop
- **[NeuronMCP Setup](../../NeuronDesktop/docs/NEURONMCP_SETUP.md)** - MCP server setup

### Official Documentation

- **[NeuronDesktop Guide](https://www.neurondb.ai/docs/neurondesktop)** - Web interface documentation
- **[API Reference](https://www.neurondb.ai/docs/neurondesktop/api)** - Complete API documentation
- **[Deployment Guide](https://www.neurondb.ai/docs/neurondesktop/deployment)** - Production deployment

## Installation

### Docker (Recommended)

```bash
cd NeuronDesktop
docker-compose up -d
```

### Manual Setup

**Backend:**
```bash
cd NeuronDesktop/api
go build ./cmd/server
./server
```

**Frontend:**
```bash
cd NeuronDesktop/frontend
npm install
npm run dev
```

## Quick Start

### Access the Interface

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8081

### Configure Profiles

1. Navigate to Settings
2. Create a new profile
3. Configure connections to:
   - NeuronDB (PostgreSQL connection)
   - NeuronAgent (HTTP API endpoint and API key)
   - NeuronMCP (MCP server configuration)

### Use the Interface

- **MCP Console**: Test and use MCP tools
- **NeuronDB Search**: Perform vector searches
- **Agent Management**: Create and manage agents
- **Metrics Dashboard**: View system metrics and health

## Features

### Core Features
- **Unified Interface**: Single dashboard for all components
- **Real-time Communication**: WebSocket support for live updates
- **Secure Authentication**: API key-based authentication
- **Professional UI**: Modern, responsive design
- **Comprehensive Logging**: Request/response logging
- **Metrics & Monitoring**: Built-in metrics collection

### Technical Features
- **Modular Architecture**: Clean separation of concerns
- **Production Ready**: Error handling, graceful shutdown, connection pooling
- **Docker Support**: Complete Docker Compose setup
- **Type Safety**: Full TypeScript frontend, strongly-typed Go backend
- **Validation**: Comprehensive input validation and SQL injection protection
- **Rate Limiting**: Configurable rate limits per API key

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (Frontend)                    │
│              Next.js + React + TypeScript                │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP/WebSocket
┌──────────────────────▼──────────────────────────────────┐
│              NeuronDesktop API (Backend)                │
│                      Go + Gorilla Mux                   │
├─────────────────────────────────────────────────────────┤
│  MCP Proxy  │  NeuronDB Client  │  Agent Client         │
└──────┬──────────────┬───────────────┬──────────────────┘
       │              │               │
┌──────▼──────┐  ┌───▼────┐  ┌───────▼────────┐
│  MCP Server │  │NeuronDB│  │  NeuronAgent   │
│  (stdio)   │  │(Postgres)│  │  (HTTP API)   │
└─────────────┘  └─────────┘  └───────────────┘
```

## Configuration

### Environment Variables

**Backend:**
- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` - Database connection
- `SERVER_PORT` - Server port (default: 8081)
- `LOG_LEVEL` - Log level (debug, info, warn, error)
- `CORS_ALLOWED_ORIGINS` - CORS origins (comma-separated)

**Frontend:**
- `NEXT_PUBLIC_API_URL` - Backend API URL

## API Endpoints

### Key Endpoints

- `GET /health` - Health check
- `GET /api/v1/profiles` - List profiles
- `GET /api/v1/profiles/{id}/mcp/tools` - List MCP tools
- `POST /api/v1/profiles/{id}/neurondb/search` - Vector search
- `GET /api/v1/metrics` - Application metrics

See [API Documentation](../../NeuronDesktop/docs/API.md) for complete API reference.

## System Requirements

- **Backend**: Go 1.23+
- **Frontend**: Node.js 20+
- **Database**: PostgreSQL 16+ (for NeuronDesktop's own database; separate from NeuronDB database)
- **Network**: Ports 3000 (frontend) and 8081 (backend) available

## Location

**Component Directory**: [`NeuronDesktop/`](../../NeuronDesktop/)

## Support

- **Documentation**: [NeuronDesktop/docs/](../../NeuronDesktop/docs/)
- **Official Docs**: [https://www.neurondb.ai/docs/neurondesktop](https://www.neurondb.ai/docs/neurondesktop)
- **Issues**: [GitHub Issues](https://github.com/neurondb/NeurondB/issues)

