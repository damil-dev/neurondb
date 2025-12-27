# Getting Started with NeuronDB Ecosystem

Welcome to the NeuronDB ecosystem! This guide will help you get started with installing and running all components.

## Overview

The NeuronDB ecosystem consists of four integrated components:

1. **NeuronDB** - PostgreSQL extension for vector search and ML
2. **NeuronAgent** - AI agent runtime system
3. **NeuronMCP** - Model Context Protocol server
4. **NeuronDesktop** - Unified web interface

## Quick Links

- **[Installation Guide](installation.md)** - Step-by-step installation instructions
- **[Quick Start Guide](quickstart.md)** - Get up and running in minutes

## Prerequisites

Before you begin, ensure you have:

- **PostgreSQL 16, 17, or 18** installed
- **Docker and Docker Compose** (for containerized deployment)
- **Go 1.23+** (for building from source)
- **Node.js 20+** (for NeuronDesktop frontend, if building from source)

## Installation Options

### Option 1: Docker (Recommended)

The easiest way to get started is using Docker Compose:

```bash
# Start all services
docker-compose up -d
```

See the [Docker Deployment Guide](../deployment/docker.md) for detailed instructions.

### Option 2: Source Build

Build and install each component from source:

1. Install NeuronDB extension
2. Build and run NeuronAgent
3. Build and run NeuronMCP
4. Build and run NeuronDesktop

See the [Installation Guide](installation.md) for detailed steps.

## Next Steps

1. **[Install the Ecosystem](installation.md)** - Complete installation guide
2. **[Quick Start Tutorial](quickstart.md)** - Run your first queries
3. **[Component Documentation](../components/README.md)** - Learn about each component
4. **[Integration Guide](../ecosystem/integration.md)** - Connect components together

## Official Documentation

For comprehensive guides and tutorials, visit:
**🌐 [https://www.neurondb.ai/docs](https://www.neurondb.ai/docs)**

## Component-Specific Guides

- [NeuronDB Installation](../components/neurondb.md#installation)
- [NeuronAgent Setup](../components/neuronagent.md#setup)
- [NeuronMCP Configuration](../components/neuronmcp.md#configuration)
- [NeuronDesktop Deployment](../components/neurondesktop.md#deployment)

