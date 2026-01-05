# Complete Quick Start Guide

Get the NeuronDB ecosystem up and running in minutes.

## Prerequisites

- Docker and Docker Compose
- 8GB+ RAM
- 20GB+ disk space
- PostgreSQL 14+ (if not using Docker)

## Installation

### Option 1: Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/neurondb/neurondb2.git
cd neurondb2

# Copy environment file
cp env.example .env

# Edit .env with your settings (optional)
# nano .env

# Start all services
docker compose up -d

# Wait for services to start
sleep 30

# Run smoke tests
./scripts/smoke-test.sh
```

### Option 2: Manual Installation

See `Docs/deployment/` for detailed installation instructions.

## First Steps

### 1. Access NeuronDesktop

Open your browser:
```
http://localhost:3000
```

### 2. Create Account

- Click "Register" (if local auth enabled)
- Or click "Login with OIDC" (if OIDC configured)

### 3. Complete Onboarding

The onboarding wizard will guide you through:
1. Database connection
2. MCP configuration
3. Agent setup
4. Demo dataset (optional)

### 4. Create Your First Agent

```python
from neuronagent import NeuronAgentClient

client = NeuronAgentClient(
    base_url="http://localhost:8080",
    api_key="your-api-key"
)

agent = client.agents.create_agent(
    name="my-first-agent",
    system_prompt="You are a helpful assistant",
    model_name="gpt-4"
)
```

### 5. Start a Session

```python
session = client.sessions.create_session(
    agent_id=agent.id,
    external_user_id="user-123"
)

response = client.sessions.send_message(
    session_id=session.id,
    role="user",
    content="Hello, agent!"
)

print(response.content)
```

## Next Steps

### Load Data

```python
# Ingest dataset
from neuronagent import NeuronAgentClient

client = NeuronAgentClient(...)

# Via API
import requests
requests.post(
    "http://localhost:8081/api/v1/profiles/{profile_id}/neurondb/ingest",
    json={
        "source_type": "file",
        "source_path": "/path/to/data.csv",
        "auto_embed": True,
        "create_index": True
    }
)
```

### Use RAG

```python
# Search documents
results = client.neurondb.search(
    collection="documents",
    query="What is NeuronDB?",
    limit=5
)

# Use in agent context
agent_response = client.sessions.send_message(
    session_id=session.id,
    role="user",
    content=f"Based on these documents: {results}, answer my question"
)
```

### Monitor System

```bash
# Health check
./scripts/health-check.sh

# View logs
docker logs -f neurondb-cpu
docker logs -f neuronagent
docker logs -f neurondesk-api

# Observability dashboard
open http://localhost:3001  # Grafana
```

## Common Tasks

### Backup

```bash
./scripts/backup.sh
```

### Restore

```bash
./scripts/restore.sh backups/neurondb_backup_20250101_120000.tar.gz
```

### Update

```bash
git pull
docker compose pull
docker compose up -d
```

### Stop

```bash
docker compose down
```

### Restart

```bash
docker compose restart
```

## Troubleshooting

See `Docs/TROUBLESHOOTING.md` for detailed troubleshooting guide.

### Quick Fixes

**Services not starting**:
```bash
docker compose logs
docker compose restart
```

**Database connection failed**:
```bash
docker exec neurondb-cpu pg_isready
```

**Port conflicts**:
```bash
# Check ports
netstat -tuln | grep -E "5432|8080|8081"

# Change ports in docker-compose.yml
```

## Resources

- **Documentation**: `Docs/`
- **Examples**: `examples/`
- **API Reference**: `Docs/`
- **SDKs**: `sdks/`

## Support

- **GitHub Issues**: Report bugs
- **Email**: support@neurondb.ai
- **Documentation**: Full docs in `Docs/`

## What's Next?

1. **Explore Examples**: Check `examples/` directory
2. **Read Documentation**: Full docs in `Docs/`
3. **Join Community**: GitHub Discussions
4. **Contribute**: See `CONTRIBUTING.md`

Happy building! 🚀

