# NeuronDesktop API Complete Reference

**Complete REST API and WebSocket reference for NeuronDesktop.**

> **Version:** 1.0  
> **Base URL:** `http://localhost:8081/api/v1`  
> **Last Updated:** 2025-01-01

## Table of Contents

- [Authentication](#authentication)
- [Profiles](#profiles)
- [NeuronDB Operations](#neurondb-operations)
- [Agent Integration](#agent-integration)
- [MCP Integration](#mcp-integration)
- [Model Management](#model-management)
- [Database Management](#database-management)
- [System Metrics](#system-metrics)
- [WebSocket API](#websocket-api)

---

## Authentication

### JWT Authentication

NeuronDesktop supports JWT (JSON Web Token) authentication:

```
Authorization: Bearer <jwt_token>
```

### OIDC Authentication

NeuronDesktop also supports OIDC (OpenID Connect) authentication for enterprise SSO.

### API Key Authentication

For programmatic access:

```
Authorization: Bearer <api_key>
```

---

## Profiles

### List Profiles

**Endpoint:** `GET /api/v1/profiles`

**Response:** `200 OK`

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "My Profile",
    "user_id": "user123",
    "mcp_config": {
      "command": "neurondb-mcp",
      "args": [],
      "env": {}
    },
    "neurondb_dsn": "host=localhost port=5432 user=neurondb dbname=neurondb",
    "agent_endpoint": "http://localhost:8080",
    "agent_api_key": "key",
    "default_collection": "documents",
    "created_at": "2025-01-01T00:00:00Z",
    "updated_at": "2025-01-01T00:00:00Z"
  }
]
```

---

### Create Profile

**Endpoint:** `POST /api/v1/profiles`

**Request Body:**
```json
{
  "name": "My Profile",
  "mcp_config": {
    "command": "neurondb-mcp",
    "args": [],
    "env": {
      "NEURONDB_HOST": "localhost"
    }
  },
  "neurondb_dsn": "host=localhost port=5432 user=neurondb dbname=neurondb",
  "agent_endpoint": "http://localhost:8080",
  "agent_api_key": "key",
  "default_collection": "documents"
}
```

**Response:** `201 Created`

---

### Get Profile

**Endpoint:** `GET /api/v1/profiles/{id}`

**Path Parameters:**
- `id` (required): Profile UUID

**Response:** `200 OK`

---

### Update Profile

**Endpoint:** `PUT /api/v1/profiles/{id}`

**Path Parameters:**
- `id` (required): Profile UUID

**Request Body:**
```json
{
  "name": "Updated Profile",
  "neurondb_dsn": "host=localhost port=5432 user=neurondb dbname=neurondb",
  "agent_endpoint": "http://localhost:8080",
  "agent_api_key": "new_key"
}
```

**Response:** `200 OK`

---

### Delete Profile

**Endpoint:** `DELETE /api/v1/profiles/{id}`

**Path Parameters:**
- `id` (required): Profile UUID

**Response:** `204 No Content`

---

## NeuronDB Operations

### List Collections

**Endpoint:** `GET /api/v1/profiles/{profile_id}/neurondb/collections`

**Path Parameters:**
- `profile_id` (required): Profile UUID

**Response:** `200 OK`

```json
[
  {
    "name": "documents",
    "vector_dim": 384,
    "num_vectors": 1000
  }
]
```

---

### Search

**Endpoint:** `POST /api/v1/profiles/{profile_id}/neurondb/search`

**Path Parameters:**
- `profile_id` (required): Profile UUID

**Request Body:**
```json
{
  "collection": "documents",
  "query_vector": [0.1, 0.2, 0.3],
  "limit": 10,
  "distance_type": "cosine"
}
```

**Response:** `200 OK`

```json
{
  "results": [
    {
      "id": 1,
      "vector": [0.1, 0.2, 0.3],
      "distance": 0.123,
      "metadata": {}
    }
  ],
  "count": 10
}
```

---

### Execute SQL

**Endpoint:** `POST /api/v1/profiles/{profile_id}/neurondb/sql`

**Path Parameters:**
- `profile_id` (required): Profile UUID

**Request Body:**
```json
{
  "query": "SELECT * FROM documents LIMIT 10"
}
```

**Note:** Only SELECT queries are allowed for safety.

**Response:** `200 OK`

```json
{
  "columns": ["id", "content", "embedding"],
  "rows": [
    [1, "Document 1", "[0.1, 0.2, 0.3]"],
    [2, "Document 2", "[0.4, 0.5, 0.6]"]
  ]
}
```

---

## Agent Integration

### List Agents

**Endpoint:** `GET /api/v1/profiles/{profile_id}/agents`

**Path Parameters:**
- `profile_id` (required): Profile UUID

**Response:** `200 OK`

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "my-agent",
    "description": "A helpful agent",
    "model_name": "gpt-4",
    "enabled_tools": ["sql", "http"]
  }
]
```

---

### Create Agent

**Endpoint:** `POST /api/v1/profiles/{profile_id}/agents`

**Path Parameters:**
- `profile_id` (required): Profile UUID

**Request Body:**
```json
{
  "name": "my-agent",
  "description": "A helpful agent",
  "system_prompt": "You are a helpful assistant.",
  "model_name": "gpt-4",
  "enabled_tools": ["sql", "http"],
  "config": {
    "temperature": 0.7,
    "max_tokens": 1000
  }
}
```

**Response:** `201 Created`

---

### Send Message to Agent

**Endpoint:** `POST /api/v1/profiles/{profile_id}/agents/{agent_id}/messages`

**Path Parameters:**
- `profile_id` (required): Profile UUID
- `agent_id` (required): Agent UUID

**Request Body:**
```json
{
  "content": "Hello!",
  "session_id": "660e8400-e29b-41d4-a716-446655440000",
  "stream": false
}
```

**Response:** `200 OK`

```json
{
  "response": "Hello! How can I help you?",
  "tokens_used": 8,
  "tool_calls": [],
  "session_id": "660e8400-e29b-41d4-a716-446655440000"
}
```

---

## MCP Integration

### List MCP Connections

**Endpoint:** `GET /api/v1/mcp/connections`

**Response:** `200 OK`

```json
[
  {
    "profile_id": "550e8400-e29b-41d4-a716-446655440000",
    "alive": true,
    "connected_at": "2025-01-01T00:00:00Z"
  }
]
```

---

### List MCP Tools

**Endpoint:** `GET /api/v1/profiles/{profile_id}/mcp/tools`

**Path Parameters:**
- `profile_id` (required): Profile UUID

**Response:** `200 OK`

```json
{
  "tools": [
    {
      "name": "vector_search",
      "description": "Perform vector search",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query_vector": {
            "type": "array",
            "items": {"type": "number"}
          },
          "table": {
            "type": "string"
          },
          "limit": {
            "type": "integer"
          }
        },
        "required": ["query_vector", "table"]
      }
    }
  ]
}
```

---

### Call MCP Tool

**Endpoint:** `POST /api/v1/profiles/{profile_id}/mcp/tools/call`

**Path Parameters:**
- `profile_id` (required): Profile UUID

**Request Body:**
```json
{
  "name": "vector_search",
  "arguments": {
    "query_vector": [0.1, 0.2, 0.3],
    "table": "documents",
    "vector_column": "embedding",
    "limit": 10
  }
}
```

**Response:** `200 OK`

```json
{
  "content": [
    {
      "type": "text",
      "text": "Search results..."
    }
  ],
  "isError": false,
  "metadata": {}
}
```

---

### MCP WebSocket

**Endpoint:** `ws://localhost:8081/api/v1/profiles/{profile_id}/mcp/ws`

**Message Format:**
```json
{
  "type": "tool_call",
  "name": "vector_search",
  "arguments": {
    "query_vector": [0.1, 0.2, 0.3],
    "table": "documents"
  }
}
```

**Response:**
```json
{
  "type": "tool_result",
  "content": [
    {
      "type": "text",
      "text": "Results..."
    }
  ],
  "isError": false
}
```

---

## Model Management

### List Models

**Endpoint:** `GET /api/v1/profiles/{profile_id}/models`

**Path Parameters:**
- `profile_id` (required): Profile UUID

**Response:** `200 OK`

```json
[
  {
    "name": "all-MiniLM-L6-v2",
    "type": "embedding",
    "dimensions": 384,
    "config": {}
  }
]
```

---

### Configure Model

**Endpoint:** `POST /api/v1/profiles/{profile_id}/models/{model_name}/config`

**Path Parameters:**
- `profile_id` (required): Profile UUID
- `model_name` (required): Model name

**Request Body:**
```json
{
  "batch_size": 32,
  "max_length": 512
}
```

**Response:** `200 OK`

---

## Database Management

### List Databases

**Endpoint:** `GET /api/v1/profiles/{profile_id}/databases`

**Path Parameters:**
- `profile_id` (required): Profile UUID

**Response:** `200 OK`

```json
[
  {
    "name": "neurondb",
    "size_mb": 1024,
    "num_tables": 10
  }
]
```

---

### Get Database Info

**Endpoint:** `GET /api/v1/profiles/{profile_id}/databases/{database_name}`

**Path Parameters:**
- `profile_id` (required): Profile UUID
- `database_name` (required): Database name

**Response:** `200 OK`

---

## System Metrics

### Get System Metrics

**Endpoint:** `GET /api/v1/metrics/system`

**Response:** `200 OK`

```json
{
  "cpu_usage": 45.5,
  "memory_usage": 60.2,
  "disk_usage": 75.0,
  "network_io": {
    "bytes_sent": 1024000,
    "bytes_recv": 2048000
  }
}
```

---

## WebSocket API

### Agent WebSocket

**Endpoint:** `ws://localhost:8081/api/v1/profiles/{profile_id}/agents/{agent_id}/ws`

**Connect:**
```javascript
const ws = new WebSocket('ws://localhost:8081/api/v1/profiles/{profile_id}/agents/{agent_id}/ws?token=jwt_token');
```

**Send Message:**
```json
{
  "type": "message",
  "content": "Hello!",
  "session_id": "660e8400-e29b-41d4-a716-446655440000"
}
```

**Receive Response:**
```json
{
  "type": "chunk",
  "content": "Hello",
  "done": false
}
```

```json
{
  "type": "complete",
  "response": "Hello! How can I help you?",
  "tokens_used": 8,
  "done": true
}
```

---

## Error Handling

### Error Response Format

```json
{
  "error": "Error Type",
  "message": "Error message",
  "code": "ERROR_CODE",
  "details": {
    "field": "field_name",
    "message": "Validation message"
  }
}
```

### HTTP Status Codes

- `200 OK`: Success
- `201 Created`: Resource created
- `204 No Content`: Success, no content
- `400 Bad Request`: Invalid request
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

---

## Rate Limiting

Rate limits are applied per API key:

- **Limit:** 100 requests per minute
- **Headers:**
  - `X-RateLimit-Limit`: Maximum requests allowed
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Time when limit resets

---

## Related Documentation

- [NeuronDesktop Deployment](../neurondesktop/docs/DEPLOYMENT.md)
- [NeuronDesktop Integration](../neurondesktop/docs/INTEGRATION.md)
- [NeuronAgent Usage](../neurondesktop/docs/NEURONAGENT_USAGE.md)
- [NeuronMCP Setup](../neurondesktop/docs/NEURONMCP_SETUP.md)

---

**Last Updated:** 2025-01-01  
**Documentation Version:** 1.0.0


