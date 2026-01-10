# NeuronAgent API Complete Reference

**Complete REST API and WebSocket reference for NeuronAgent.**

> **Version:** 1.0  
> **Base URL:** `http://localhost:8080/api/v1`  
> **Last Updated:** 2025-01-01

## Table of Contents

- [Authentication](#authentication)
- [Agents](#agents)
- [Sessions](#sessions)
- [Messages](#messages)
- [Tools](#tools)
- [Memory](#memory)
- [Webhooks](#webhooks)
- [WebSocket API](#websocket-api)
- [Error Handling](#error-handling)

---

## Authentication

All API requests require authentication using an API key in the Authorization header:

```
Authorization: Bearer <api_key>
```

**Example:**
```bash
curl -H "Authorization: Bearer your-api-key" \
     http://localhost:8080/api/v1/agents
```

---

## Agents

### List Agents

**Endpoint:** `GET /api/v1/agents`

**Query Parameters:**
- `search` (optional): Search query to filter agents

**Response:** `200 OK`

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "my-agent",
    "description": "A helpful agent",
    "system_prompt": "You are a helpful assistant.",
    "model_name": "gpt-4",
    "memory_table": "agent_memory",
    "enabled_tools": ["sql", "http"],
    "config": {
      "temperature": 0.7,
      "max_tokens": 1000
    },
    "created_at": "2025-01-01T00:00:00Z",
    "updated_at": "2025-01-01T00:00:00Z"
  }
]
```

**Example:**
```bash
curl -H "Authorization: Bearer your-api-key" \
     "http://localhost:8080/api/v1/agents?search=helpful"
```

---

### Create Agent

**Endpoint:** `POST /api/v1/agents`

**Request Body:**
```json
{
  "name": "my-agent",
  "description": "A helpful agent",
  "system_prompt": "You are a helpful assistant.",
  "model_name": "gpt-4",
  "memory_table": "agent_memory",
  "enabled_tools": ["sql", "http"],
  "config": {
    "temperature": 0.7,
    "max_tokens": 1000
  }
}
```

**Response:** `201 Created`

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "my-agent",
  "description": "A helpful agent",
  "system_prompt": "You are a helpful assistant.",
  "model_name": "gpt-4",
  "memory_table": "agent_memory",
  "enabled_tools": ["sql", "http"],
  "config": {
    "temperature": 0.7,
    "max_tokens": 1000
  },
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:00:00Z"
}
```

---

### Get Agent

**Endpoint:** `GET /api/v1/agents/{id}`

**Path Parameters:**
- `id` (required): Agent UUID

**Response:** `200 OK`

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "my-agent",
  "description": "A helpful agent",
  "system_prompt": "You are a helpful assistant.",
  "model_name": "gpt-4",
  "memory_table": "agent_memory",
  "enabled_tools": ["sql", "http"],
  "config": {
    "temperature": 0.7,
    "max_tokens": 1000
  },
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:00:00Z"
}
```

---

### Update Agent

**Endpoint:** `PUT /api/v1/agents/{id}`

**Path Parameters:**
- `id` (required): Agent UUID

**Request Body:**
```json
{
  "name": "updated-agent",
  "description": "Updated description",
  "system_prompt": "Updated prompt",
  "model_name": "gpt-4",
  "memory_table": "agent_memory",
  "enabled_tools": ["sql", "http", "code"],
  "config": {
    "temperature": 0.8,
    "max_tokens": 2000
  }
}
```

**Response:** `200 OK`

---

### Delete Agent

**Endpoint:** `DELETE /api/v1/agents/{id}`

**Path Parameters:**
- `id` (required): Agent UUID

**Response:** `204 No Content`

---

## Sessions

### Create Session

**Endpoint:** `POST /api/v1/sessions`

**Request Body:**
```json
{
  "agent_id": "550e8400-e29b-41d4-a716-446655440000",
  "external_user_id": "user123",
  "metadata": {
    "source": "web"
  }
}
```

**Response:** `201 Created`

```json
{
  "id": "660e8400-e29b-41d4-a716-446655440000",
  "agent_id": "550e8400-e29b-41d4-a716-446655440000",
  "external_user_id": "user123",
  "metadata": {
    "source": "web"
  },
  "created_at": "2025-01-01T00:00:00Z",
  "last_activity_at": "2025-01-01T00:00:00Z"
}
```

---

### Get Session

**Endpoint:** `GET /api/v1/sessions/{id}`

**Path Parameters:**
- `id` (required): Session UUID

**Response:** `200 OK`

```json
{
  "id": "660e8400-e29b-41d4-a716-446655440000",
  "agent_id": "550e8400-e29b-41d4-a716-446655440000",
  "external_user_id": "user123",
  "metadata": {
    "source": "web"
  },
  "created_at": "2025-01-01T00:00:00Z",
  "last_activity_at": "2025-01-01T00:00:00Z"
}
```

---

### Update Session

**Endpoint:** `PUT /api/v1/sessions/{id}`

**Path Parameters:**
- `id` (required): Session UUID

**Request Body:**
```json
{
  "external_user_id": "user456",
  "metadata": {
    "source": "mobile"
  }
}
```

**Response:** `200 OK`

---

### Delete Session

**Endpoint:** `DELETE /api/v1/sessions/{id}`

**Path Parameters:**
- `id` (required): Session UUID

**Response:** `204 No Content`

---

## Messages

### Get Messages

**Endpoint:** `GET /api/v1/sessions/{session_id}/messages`

**Path Parameters:**
- `session_id` (required): Session UUID

**Query Parameters:**
- `limit` (optional): Maximum number of messages (default: 100, max: 1000)
- `offset` (optional): Number of messages to skip (default: 0)

**Response:** `200 OK`

```json
[
  {
    "id": 1,
    "session_id": "660e8400-e29b-41d4-a716-446655440000",
    "role": "user",
    "content": "Hello!",
    "tool_name": null,
    "tool_call_id": null,
    "token_count": 2,
    "metadata": {},
    "created_at": "2025-01-01T00:00:00Z"
  },
  {
    "id": 2,
    "session_id": "660e8400-e29b-41d4-a716-446655440000",
    "role": "assistant",
    "content": "Hello! How can I help you?",
    "tool_name": null,
    "tool_call_id": null,
    "token_count": 8,
    "metadata": {},
    "created_at": "2025-01-01T00:00:01Z"
  }
]
```

---

### Send Message

**Endpoint:** `POST /api/v1/sessions/{session_id}/messages`

**Path Parameters:**
- `session_id` (required): Session UUID

**Request Body:**
```json
{
  "role": "user",
  "content": "What is machine learning?",
  "stream": false,
  "metadata": {}
}
```

**Response:** `200 OK`

```json
{
  "session_id": "660e8400-e29b-41d4-a716-446655440000",
  "agent_id": "550e8400-e29b-41d4-a716-446655440000",
  "response": "Machine learning is a subset of artificial intelligence...",
  "tokens_used": 150,
  "tool_calls": [],
  "tool_results": []
}
```

**Streaming Response:**

Set `"stream": true` to enable streaming. Response will be sent as Server-Sent Events (SSE).

---

## Tools

### List Tools

**Endpoint:** `GET /api/v1/tools`

**Response:** `200 OK`

```json
[
  {
    "name": "sql",
    "description": "Execute SQL queries",
    "schema": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "SQL query to execute"
        }
      },
      "required": ["query"]
    },
    "created_at": "2025-01-01T00:00:00Z",
    "updated_at": "2025-01-01T00:00:00Z"
  }
]
```

---

### Create Tool

**Endpoint:** `POST /api/v1/tools`

**Request Body:**
```json
{
  "name": "custom_tool",
  "description": "Custom tool description",
  "schema": {
    "type": "object",
    "properties": {
      "input": {
        "type": "string"
      }
    },
    "required": ["input"]
  },
  "implementation": "function code or config"
}
```

**Response:** `201 Created`

---

### Get Tool

**Endpoint:** `GET /api/v1/tools/{name}`

**Path Parameters:**
- `name` (required): Tool name

**Response:** `200 OK`

---

### Update Tool

**Endpoint:** `PUT /api/v1/tools/{name}`

**Path Parameters:**
- `name` (required): Tool name

**Request Body:**
```json
{
  "description": "Updated description",
  "schema": {
    "type": "object",
    "properties": {
      "input": {
        "type": "string"
      }
    }
  },
  "implementation": "updated implementation"
}
```

**Response:** `200 OK`

---

### Delete Tool

**Endpoint:** `DELETE /api/v1/tools/{name}`

**Path Parameters:**
- `name` (required): Tool name

**Response:** `204 No Content`

---

## Memory

### Get Memory Chunks

**Endpoint:** `GET /api/v1/sessions/{session_id}/memory`

**Path Parameters:**
- `session_id` (required): Session UUID

**Query Parameters:**
- `limit` (optional): Maximum number of chunks (default: 10)
- `query` (optional): Search query for semantic search

**Response:** `200 OK`

```json
[
  {
    "id": "770e8400-e29b-41d4-a716-446655440000",
    "session_id": "660e8400-e29b-41d4-a716-446655440000",
    "content": "Memory chunk content",
    "embedding": [0.1, 0.2, 0.3],
    "metadata": {},
    "created_at": "2025-01-01T00:00:00Z"
  }
]
```

---

### Add Memory Chunk

**Endpoint:** `POST /api/v1/sessions/{session_id}/memory`

**Path Parameters:**
- `session_id` (required): Session UUID

**Request Body:**
```json
{
  "content": "Important information to remember",
  "metadata": {
    "source": "user_input"
  }
}
```

**Response:** `201 Created`

---

## Webhooks

### List Webhooks

**Endpoint:** `GET /api/v1/webhooks`

**Response:** `200 OK`

```json
[
  {
    "id": "880e8400-e29b-41d4-a716-446655440000",
    "url": "https://example.com/webhook",
    "events": ["message.created", "session.created"],
    "secret": "webhook_secret",
    "created_at": "2025-01-01T00:00:00Z"
  }
]
```

---

### Create Webhook

**Endpoint:** `POST /api/v1/webhooks`

**Request Body:**
```json
{
  "url": "https://example.com/webhook",
  "events": ["message.created", "session.created"],
  "secret": "webhook_secret"
}
```

**Response:** `201 Created`

---

## WebSocket API

### Connect

**Endpoint:** `ws://localhost:8080/api/v1/ws`

**Authentication:** Include API key in query parameter or header

**Example:**
```javascript
const ws = new WebSocket('ws://localhost:8080/api/v1/ws?api_key=your-api-key');
```

### Send Message

```json
{
  "type": "message",
  "session_id": "660e8400-e29b-41d4-a716-446655440000",
  "content": "Hello!",
  "stream": true
}
```

### Receive Response

**Streaming:**
```json
{
  "type": "chunk",
  "content": "Hello",
  "done": false
}
```

**Complete:**
```json
{
  "type": "complete",
  "response": "Hello! How can I help you?",
  "tokens_used": 8,
  "tool_calls": [],
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
  "code": 400,
  "request_id": "request-uuid",
  "details": {}
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

## Related Documentation

- [NeuronAgent Architecture](../internals/neuronagent-architecture.md)
- [NeuronAgent Tools](../reference/neuronagent-tools.md)
- [OpenAPI Specification](../../NeuronAgent/openapi/openapi.yaml)

---

**Last Updated:** 2025-01-01  
**Documentation Version:** 1.0.0




