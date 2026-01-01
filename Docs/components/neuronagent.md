# NeuronAgent

NeuronAgent is a production-ready AI agent runtime system providing REST API and WebSocket endpoints for building autonomous agent applications.

## What it is

- **Agent Runtime**: Complete state machine for autonomous task execution with persistent memory
- **REST API**: Full CRUD API for agents, sessions, messages, and advanced features
- **WebSocket Support**: Real-time streaming agent responses
- **Tool System**: Extensible tool registry with 20+ built-in tools
- **Multi-Agent Collaboration**: Agent-to-agent communication and task delegation
- **Workflow Engine**: DAG-based workflow execution with human-in-the-loop support
- **Memory Management**: HNSW-based vector search for long-term memory with hierarchical organization
- **Integration**: Direct integration with NeuronDB for embeddings, LLM, and vector operations

## Key Features & Modules

### Core Agent Runtime
- **Agent State Machine**: Complete execution engine with state management
- **Session Management**: Multi-session support with caching and cleanup
- **Context Management**: Intelligent context loading from messages and memory
- **Prompt Engineering**: Advanced prompt construction with templating
- **LLM Integration**: Integration with NeuronDB LLM functions (OpenAI, HuggingFace)

### Memory & Knowledge
- **Long-term Memory**: HNSW-based vector search for context retrieval
- **Hierarchical Memory**: Multi-level memory organization for better recall
- **Memory Promotion**: Background worker for promoting important memories
- **Event Streaming**: Real-time event capture and summarization

### Tool System (20+ Tools)
- **Core Tools**: SQL (read-only), HTTP (with allowlist), Code (sandboxed), Shell (whitelisted)
- **Browser Tool**: Web automation with Playwright for DOM interaction and navigation
- **Filesystem Tool**: Virtual filesystem integration for file operations
- **Memory Tool**: Direct memory manipulation and retrieval
- **Collaboration Tool**: Multi-agent communication and task delegation
- **NeuronDB Tools**: RAG, Hybrid Search, Reranking, Vector, ML, Analytics, Visualization
- **Multimodal Tool**: Image and multimedia processing
- **Tool Registry**: Extensible system for registering custom tools

### Multi-Agent Collaboration
- **Agent Delegation**: Delegate tasks to specialized agents
- **Inter-Agent Communication**: Message passing between agents
- **Workspace Management**: Shared workspaces for collaborative agents
- **Sub-Agents**: Hierarchical agent structures for complex tasks
- **Collaboration API**: REST endpoints for managing agent collaborations

### Workflow Engine
- **DAG Workflows**: Directed acyclic graph workflow execution
- **Workflow Steps**: Agent, tool, HTTP, approval, and conditional steps
- **Human-in-the-Loop (HITL)**: Approval gates and feedback loops
- **Idempotency**: Idempotent step execution with key-based caching
- **Retries**: Configurable retry logic for workflow steps
- **Workflow API**: Complete CRUD API for workflows and executions

### Planning & Task Management
- **LLM-Based Planning**: Advanced planning with task decomposition
- **Task Plans**: Multi-step plan creation and execution
- **Async Tasks**: Background task execution with job queue
- **Task Notifications**: Alerts and notifications for task events
- **Plans API**: Endpoints for creating, managing, and executing plans

### Quality & Evaluation
- **Reflections**: Agent self-reflection and quality assessment
- **Quality Scoring**: Automated quality scoring for agent responses
- **Evaluation Framework**: Built-in evaluation system for agent performance
- **Verification Agent**: Dedicated agent for verifying outputs
- **Execution Snapshots**: Capture and replay agent execution states

### Budget & Cost Management
- **Cost Tracking**: Real-time cost tracking for LLM usage
- **Budget Management**: Per-agent and per-session budget controls
- **Budget Alerts**: Configurable alerts for budget thresholds
- **Budget API**: Complete API for managing budgets and tracking costs

### Human-in-the-Loop (HITL)
- **Approval Workflows**: Human approval gates in workflows
- **Feedback System**: Collect and integrate human feedback
- **Alert Preferences**: Configurable alert preferences for users
- **HumanLoop API**: Endpoints for approvals and feedback

### Versioning & History
- **Version Management**: Version control for agents and configurations
- **Execution Replay**: Replay previous agent executions
- **Execution Snapshots**: Capture and restore agent states
- **Versions API**: API for managing versions and viewing history

### Observability & Monitoring
- **Prometheus Metrics**: Comprehensive metrics export
- **Structured Logging**: JSON-formatted logs with context
- **Tracing**: Distributed tracing support
- **Debugging Tools**: Advanced debugging capabilities
- **Event Streaming**: Real-time event capture and analysis

### Security & Safety
- **API Key Authentication**: Bcrypt-hashed API keys with rate limiting
- **RBAC**: Role-based access control with fine-grained permissions
- **Data Permissions**: Per-principal data access controls
- **Tool Permissions**: Granular tool access permissions
- **Audit Logging**: Comprehensive audit trail for all operations
- **Safety Moderation**: Content moderation and safety checks

### Integrations & Connectors
- **S3 Connector**: AWS S3 integration for storage
- **GitHub Connector**: GitHub API integration
- **GitLab Connector**: GitLab API integration
- **Slack Connector**: Slack webhook integration
- **Webhooks**: Outbound webhook support for events
- **Secrets Management**: AWS Secrets Manager and HashiCorp Vault integration

### Storage & Persistence
- **Database Storage**: PostgreSQL-based persistence
- **S3 Storage**: Object storage for large files
- **Multimodal Storage**: Specialized storage for images and media
- **Session Caching**: Redis-compatible session caching

### Background Workers
- **Job Queue**: PostgreSQL-based job queue with SKIP LOCKED
- **Worker Pool**: Configurable worker pool with graceful shutdown
- **Async Task Worker**: Background execution of async tasks
- **Memory Promoter**: Promotes important memories to long-term storage
- **Verifier Worker**: Background verification of agent outputs

### Advanced Features
- **Batch Operations**: Batch processing for multiple requests
- **Virtual Filesystem**: Isolated filesystem for agents
- **Token Counting**: Accurate token counting for cost tracking
- **Relationship Management**: Manage relationships between entities
- **Advanced Handlers**: Specialized handlers for complex operations

## Documentation

- **Main README**: `NeuronAgent/README.md`
- **API Reference**: `NeuronAgent/docs/API.md`
- **Architecture**: `NeuronAgent/docs/ARCHITECTURE.md`
- **Deployment**: `NeuronAgent/docs/DEPLOYMENT.md`
- **OpenAPI Spec**: `NeuronAgent/openapi/openapi.yaml`
- **Official Docs**: [https://www.neurondb.ai/docs/neuronagent](https://www.neurondb.ai/docs/neuronagent)

## Docker

- Compose service: `neuronagent` (plus GPU-profile variants)
- From repo root: `docker compose up -d neuronagent`
- See: `NeuronAgent/docker/readme.md`

## Quick Start

### Minimal Verification

```bash
curl -sS http://localhost:8080/health
```

### Create Agent

```bash
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_agent",
    "profile": "general-assistant",
    "tools": ["sql", "http", "browser"]
  }'
```

For complete setup instructions, see `NeuronAgent/README.md`.
