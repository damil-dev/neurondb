/*=============================================================================
 *
 * setup.sql
 *    Initial Setup Script for NeuronAgent
 *
 * This is the initial setup SQL script for NeuronAgent. It sets up everything
 * needed for NeuronAgent to work with NeuronDB, including:
 * - NeuronDB extension
 * - neurondb_agent schema
 * - All tables (core and advanced features)
 * - All indexes (performance and vector indexes)
 * - All functions and triggers
 * - Schema migrations tracking table
 *
 * This script is idempotent and can be run multiple times safely. It combines
 * all migration files into a single, comprehensive, and well-documented setup.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronAgent/setup.sql
 *
 *=============================================================================
 *
 * TABLE OF CONTENTS
 * ==================
 *
 * 1. Header and Documentation
 * 2. Extension Setup
 * 3. Schema Creation
 * 4. Core Tables
 *    4.1. agents
 *    4.2. sessions
 *    4.3. messages
 *    4.4. memory_chunks
 *    4.5. tools
 *    4.6. jobs
 *    4.7. api_keys
 * 5. Advanced Feature Tables
 *    5.1. agent_relationships
 *    5.2. tool_usage_logs
 *    5.3. cost_logs
 *    5.4. quality_scores
 *    5.5. agent_versions
 *    5.6. plans
 *    5.7. reflections
 * 6. Schema Migrations Table
 * 7. Functions
 *    7.1. update_updated_at()
 *    7.2. update_session_activity()
 * 8. Triggers
 *    8.1. agents_updated_at
 *    8.2. tools_updated_at
 *    8.3. jobs_updated_at
 *    8.4. messages_session_activity
 * 9. Indexes
 *    9.1. Core Table Indexes
 *    9.2. Advanced Feature Indexes
 *    9.3. HNSW Vector Index
 * 10. Footer and Verification
 *
 * PREREQUISITES
 * =============
 *
 * - PostgreSQL 16 or later
 * - NeuronDB extension installed and available
 * - Database user with CREATE privileges
 *
 * USAGE
 * =====
 *
 * To run this initial setup script on a NeuronDB database:
 *
 *   psql -d neurondb -f setup.sql
 *
 * Or from within psql:
 *
 *   \i setup.sql
 *
 * This script is idempotent and can be run multiple times safely.
 * It will create all necessary database objects if they don't already exist.
 *
 * ARCHITECTURE OVERVIEW
 * =====================
 *
 * NeuronAgent uses a PostgreSQL database with the NeuronDB extension for
 * vector operations. The schema is organized into:
 *
 * - Core tables: Essential tables for agent operations (agents, sessions,
 *   messages, memory_chunks, tools, jobs, api_keys)
 *
 * - Advanced features: Extended functionality for collaboration, analytics,
 *   versioning, and quality tracking
 *
 * - Vector search: HNSW indexes for efficient similarity search on embeddings
 *
 * - Automation: Triggers and functions for automatic timestamp updates and
 *   activity tracking
 *
 *============================================================================*/

-- ============================================================================
-- SECTION 1: EXTENSION SETUP
-- ============================================================================
-- 
-- The NeuronDB extension provides:
-- - neurondb_vector type for storing high-dimensional vectors
-- - HNSW index support for fast approximate nearest neighbor search
-- - Vector similarity operators (cosine, L2, inner product)
--
-- This extension must be installed before creating the schema.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS neurondb;

COMMENT ON EXTENSION neurondb IS 
'NeuronDB extension providing vector types and HNSW indexes for efficient vector similarity search. Required for memory_chunks embeddings and vector operations.';

-- ============================================================================
-- SECTION 2: SCHEMA CREATION
-- ============================================================================
--
-- The neurondb_agent schema contains all tables, functions, and related objects
-- for the NeuronAgent system. Using a dedicated schema provides:
-- - Namespace isolation
-- - Easier permission management
-- - Clear organization
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS neurondb_agent;

COMMENT ON SCHEMA neurondb_agent IS 
'NeuronAgent database schema containing all tables, functions, triggers, and indexes for agent runtime operations, session management, tool registry, job queue, authentication, and advanced features.';

-- ============================================================================
-- SECTION 3: CORE TABLES
-- ============================================================================
--
-- Core tables provide the essential functionality for NeuronAgent operations.
-- These tables are required for basic agent functionality.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 3.1. agents - Agent Profiles and Configurations
-- ----------------------------------------------------------------------------
--
-- Stores agent definitions including system prompts, model configurations,
-- enabled tools, and runtime settings. Each agent represents a distinct
-- AI assistant configuration that can be used across multiple sessions.
--
-- Usage Example:
--   INSERT INTO neurondb_agent.agents (name, description, system_prompt, 
--     model_name, enabled_tools, config)
--   VALUES (
--     'general-assistant',
--     'General purpose assistant',
--     'You are a helpful assistant.',
--     'gpt-4',
--     ARRAY['sql', 'http'],
--     '{"temperature": 0.7, "max_tokens": 1000}'::jsonb
--   );
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS neurondb_agent.agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    system_prompt TEXT NOT NULL,
    model_name TEXT NOT NULL,
    memory_table TEXT,
    enabled_tools TEXT[] DEFAULT '{}',
    config JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    version INTEGER DEFAULT 1,
    parent_agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    CONSTRAINT valid_model_name CHECK (model_name ~ '^[a-zA-Z0-9_-]+$'),
    CONSTRAINT valid_memory_table CHECK (memory_table IS NULL OR memory_table ~ '^[a-z][a-z0-9_]*$')
);

COMMENT ON TABLE neurondb_agent.agents IS 
'Stores agent profiles and configurations. Each agent defines a complete AI assistant configuration including system prompt, model selection, enabled tools, and runtime parameters.';

COMMENT ON COLUMN neurondb_agent.agents.id IS 
'Unique identifier for the agent. Generated automatically using gen_random_uuid().';

COMMENT ON COLUMN neurondb_agent.agents.name IS 
'Unique human-readable name for the agent. Used for identification and must be unique across all agents. Example: "general-assistant", "code-assistant".';

COMMENT ON COLUMN neurondb_agent.agents.description IS 
'Optional description of the agent''s purpose and capabilities. Helps users understand what the agent is designed for.';

COMMENT ON COLUMN neurondb_agent.agents.system_prompt IS 
'System prompt that defines the agent''s behavior, personality, and instructions. This is the primary prompt sent to the LLM to configure the agent''s responses. Required field.';

COMMENT ON COLUMN neurondb_agent.agents.model_name IS 
'NeuronDB model identifier specifying which LLM model to use. Must match a valid model name pattern (alphanumeric, underscore, hyphen). Examples: "gpt-4", "gpt-3.5-turbo", "claude-3".';

COMMENT ON COLUMN neurondb_agent.agents.memory_table IS 
'Optional per-agent memory table name for custom memory storage. If specified, must be a valid PostgreSQL identifier (lowercase, starts with letter). Allows agents to have dedicated memory tables.';

COMMENT ON COLUMN neurondb_agent.agents.enabled_tools IS 
'Array of tool names that this agent can use. Tools must be registered in the tools table. Empty array means no tools are enabled. Example: ARRAY[''sql'', ''http'', ''code''].';

COMMENT ON COLUMN neurondb_agent.agents.config IS 
'JSONB configuration object containing LLM runtime parameters. Common fields: temperature (0.0-2.0), max_tokens (integer), top_p (0.0-1.0), frequency_penalty, presence_penalty. Example: {"temperature": 0.7, "max_tokens": 1000, "top_p": 0.9}.';

COMMENT ON COLUMN neurondb_agent.agents.created_at IS 
'Timestamp when the agent was first created. Automatically set to current time on INSERT.';

COMMENT ON COLUMN neurondb_agent.agents.updated_at IS 
'Timestamp when the agent was last modified. Automatically updated by trigger on UPDATE operations.';

COMMENT ON COLUMN neurondb_agent.agents.version IS 
'Version number of the agent. Used for versioning and tracking changes. Defaults to 1. Incremented when creating new versions in agent_versions table.';

COMMENT ON COLUMN neurondb_agent.agents.parent_agent_id IS 
'Optional reference to a parent agent. Used for agent hierarchies and inheritance. When an agent is based on another agent, this links to the original.';

-- ----------------------------------------------------------------------------
-- 3.2. sessions - User Conversation Sessions
-- ----------------------------------------------------------------------------
--
-- Represents a conversation session between a user and an agent. Sessions
-- track conversation state, user identity, and activity timestamps.
-- Multiple messages belong to a single session.
--
-- Usage Example:
--   INSERT INTO neurondb_agent.sessions (agent_id, external_user_id, metadata)
--   VALUES (
--     '123e4567-e89b-12d3-a456-426614174000'::uuid,
--     'user-123',
--     '{"source": "web", "ip": "192.168.1.1"}'::jsonb
--   );
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS neurondb_agent.sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    external_user_id TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_activity_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_external_user_id CHECK (external_user_id IS NULL OR length(external_user_id) > 0)
);

COMMENT ON TABLE neurondb_agent.sessions IS 
'Stores user conversation sessions. Each session represents a conversation between a user and an agent, containing multiple messages. Sessions track activity and can be associated with external user identifiers.';

COMMENT ON COLUMN neurondb_agent.sessions.id IS 
'Unique identifier for the session. Generated automatically using gen_random_uuid().';

COMMENT ON COLUMN neurondb_agent.sessions.agent_id IS 
'Foreign key reference to the agent handling this session. Required. When an agent is deleted, all associated sessions are cascade deleted.';

COMMENT ON COLUMN neurondb_agent.sessions.external_user_id IS 
'Optional external user identifier for integration with external systems. Can be used to link sessions to users in other systems. Must be non-empty if provided.';

COMMENT ON COLUMN neurondb_agent.sessions.metadata IS 
'JSONB object for storing arbitrary session metadata. Common uses: source tracking, IP addresses, user preferences, custom attributes. Example: {"source": "web", "ip": "192.168.1.1", "user_agent": "Mozilla/5.0"}.';

COMMENT ON COLUMN neurondb_agent.sessions.created_at IS 
'Timestamp when the session was created. Automatically set to current time on INSERT.';

COMMENT ON COLUMN neurondb_agent.sessions.last_activity_at IS 
'Timestamp of the last message or activity in this session. Automatically updated by trigger when messages are inserted. Used for session cleanup and activity tracking.';

-- ----------------------------------------------------------------------------
-- 3.3. messages - Conversation History
-- ----------------------------------------------------------------------------
--
-- Stores individual messages in a conversation. Messages can be from users,
-- assistants, system, or tool responses. Supports tool calls and results.
--
-- Usage Example:
--   INSERT INTO neurondb_agent.messages (session_id, role, content, token_count)
--   VALUES (
--     '123e4567-e89b-12d3-a456-426614174000'::uuid,
--     'user',
--     'Hello, how can you help me?',
--     10
--   );
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS neurondb_agent.messages (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    tool_name TEXT,
    tool_call_id TEXT,
    token_count INT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_tool_message CHECK (
        (role = 'tool' AND tool_name IS NOT NULL) OR
        (role != 'tool' AND tool_name IS NULL)
    )
);

COMMENT ON TABLE neurondb_agent.messages IS 
'Stores conversation messages between users and agents. Messages can be from users, assistants, system, or tools. Supports tool call tracking and token counting.';

COMMENT ON COLUMN neurondb_agent.messages.id IS 
'Unique sequential identifier for the message. Auto-incremented BIGSERIAL. Used for ordering messages within a session.';

COMMENT ON COLUMN neurondb_agent.messages.session_id IS 
'Foreign key reference to the session this message belongs to. Required. When a session is deleted, all associated messages are cascade deleted.';

COMMENT ON COLUMN neurondb_agent.messages.role IS 
'Message role indicating the sender type. Valid values: "user" (user input), "assistant" (agent response), "system" (system message), "tool" (tool execution result). Required.';

COMMENT ON COLUMN neurondb_agent.messages.content IS 
'The actual message text content. Required. For tool messages, this typically contains the tool execution result.';

COMMENT ON COLUMN neurondb_agent.messages.tool_name IS 
'Name of the tool that generated this message. Only set when role = "tool". Must match a tool name in the tools table. NULL for non-tool messages.';

COMMENT ON COLUMN neurondb_agent.messages.tool_call_id IS 
'Identifier linking tool results to tool calls. Used to associate tool execution results with the original tool call request. Allows tracking tool call chains.';

COMMENT ON COLUMN neurondb_agent.messages.token_count IS 
'Number of tokens in this message. Used for cost tracking and rate limiting. Can be NULL if token counting is not available or not performed.';

COMMENT ON COLUMN neurondb_agent.messages.metadata IS 
'JSONB object for storing message metadata. Common uses: model used, temperature, finish reason, tool call details, custom attributes. Example: {"model": "gpt-4", "finish_reason": "stop"}.';

COMMENT ON COLUMN neurondb_agent.messages.created_at IS 
'Timestamp when the message was created. Automatically set to current time on INSERT. Used for chronological ordering of messages.';

-- ----------------------------------------------------------------------------
-- 3.4. memory_chunks - Vector-Embedded Long-Term Memory
-- ----------------------------------------------------------------------------
--
-- Stores vector embeddings of important conversation content for long-term
-- memory retrieval. Uses NeuronDB vector type and HNSW indexes for fast
-- similarity search.
--
-- Usage Example:
--   INSERT INTO neurondb_agent.memory_chunks (agent_id, session_id, content, 
--     embedding, importance_score)
--   VALUES (
--     '123e4567-e89b-12d3-a456-426614174000'::uuid,
--     '123e4567-e89b-12d3-a456-426614174001'::uuid,
--     'User prefers dark mode interface',
--     '[0.1, 0.2, ...]'::neurondb_vector(768),
--     0.8
--   );
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS neurondb_agent.memory_chunks (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    message_id BIGINT REFERENCES neurondb_agent.messages(id) ON DELETE SET NULL,
    content TEXT NOT NULL,
    embedding neurondb_vector(768) NOT NULL,
    importance_score REAL DEFAULT 0.5 CHECK (importance_score >= 0 AND importance_score <= 1),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_embedding CHECK (embedding IS NOT NULL)
);

COMMENT ON TABLE neurondb_agent.memory_chunks IS 
'Stores vector embeddings of conversation content for long-term memory. Enables semantic search to retrieve relevant past conversations. Uses NeuronDB vector type with HNSW indexes for efficient similarity search.';

COMMENT ON COLUMN neurondb_agent.memory_chunks.id IS 
'Unique sequential identifier for the memory chunk. Auto-incremented BIGSERIAL.';

COMMENT ON COLUMN neurondb_agent.memory_chunks.agent_id IS 
'Foreign key reference to the agent this memory belongs to. Required. When an agent is deleted, all associated memory chunks are cascade deleted.';

COMMENT ON COLUMN neurondb_agent.memory_chunks.session_id IS 
'Optional foreign key reference to the session this memory chunk originated from. Set to NULL on session deletion (preserves memory even if session is deleted).';

COMMENT ON COLUMN neurondb_agent.memory_chunks.message_id IS 
'Optional foreign key reference to the specific message this memory chunk was extracted from. Set to NULL on message deletion.';

COMMENT ON COLUMN neurondb_agent.memory_chunks.content IS 
'The text content that was embedded. This is the original text that the embedding represents. Required.';

COMMENT ON COLUMN neurondb_agent.memory_chunks.embedding IS 
'Vector embedding of the content. NeuronDB vector type with 768 dimensions (configurable). Required. Generated using embedding models. Used for similarity search using cosine, L2, or inner product distance.';

COMMENT ON COLUMN neurondb_agent.memory_chunks.importance_score IS 
'Score indicating the importance of this memory chunk (0.0 to 1.0). Higher scores indicate more important memories. Used for filtering and prioritization in memory retrieval. Default: 0.5.';

COMMENT ON COLUMN neurondb_agent.memory_chunks.metadata IS 
'JSONB object for storing memory metadata. Common uses: embedding model used, extraction method, tags, custom attributes. Example: {"model": "text-embedding-ada-002", "method": "auto"}.';

COMMENT ON COLUMN neurondb_agent.memory_chunks.created_at IS 
'Timestamp when the memory chunk was created. Automatically set to current time on INSERT.';

-- ----------------------------------------------------------------------------
-- 3.5. tools - Tool Registry
-- ----------------------------------------------------------------------------
--
-- Registry of available tools that agents can use. Each tool defines its
-- handler type, argument schema, and configuration.
--
-- Usage Example:
--   INSERT INTO neurondb_agent.tools (name, description, arg_schema, 
--     handler_type, handler_config)
--   VALUES (
--     'sql_query',
--     'Execute SQL queries',
--     '{"type": "object", "properties": {"query": {"type": "string"}}}'::jsonb,
--     'sql',
--     '{"database": "main"}'::jsonb
--   );
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS neurondb_agent.tools (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    arg_schema JSONB NOT NULL,
    handler_type TEXT NOT NULL CHECK (handler_type IN ('sql', 'http', 'code', 'shell', 'queue', 'ml', 'vector', 'rag', 'analytics', 'hybrid_search', 'reranking')),
    handler_config JSONB DEFAULT '{}',
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_arg_schema CHECK (jsonb_typeof(arg_schema) = 'object')
);

COMMENT ON TABLE neurondb_agent.tools IS 
'Registry of available tools that agents can use. Defines tool metadata, argument schemas, handler types, and configuration. Tools must be registered here before agents can use them.';

COMMENT ON COLUMN neurondb_agent.tools.name IS 
'Unique name identifier for the tool. Used as the primary key and referenced in agent enabled_tools arrays. Must be unique. Examples: "sql", "http", "code", "vector_search".';

COMMENT ON COLUMN neurondb_agent.tools.description IS 
'Human-readable description of what the tool does. Used for agent tool selection and documentation. Required.';

COMMENT ON COLUMN neurondb_agent.tools.arg_schema IS 
'JSON Schema defining the tool''s argument structure. Must be a valid JSON Schema object. Used for validation and agent tool calling. Example: {"type": "object", "properties": {"query": {"type": "string", "description": "SQL query"}}}.';

COMMENT ON COLUMN neurondb_agent.tools.handler_type IS 
'Type of handler that executes this tool. Valid values: "sql" (SQL queries), "http" (HTTP requests), "code" (code execution), "shell" (shell commands), "queue" (job queue), "ml" (machine learning), "vector" (vector operations), "rag" (RAG operations), "analytics" (analytics), "hybrid_search" (hybrid search), "reranking" (reranking). Required.';

COMMENT ON COLUMN neurondb_agent.tools.handler_config IS 
'JSONB configuration object for the tool handler. Contains handler-specific settings. Example for SQL tool: {"database": "main", "timeout": 30}. Example for HTTP tool: {"base_url": "https://api.example.com"}.';

COMMENT ON COLUMN neurondb_agent.tools.enabled IS 
'Whether the tool is currently enabled. Disabled tools cannot be used by agents even if listed in enabled_tools. Default: true. Allows temporarily disabling tools without deleting them.';

COMMENT ON COLUMN neurondb_agent.tools.created_at IS 
'Timestamp when the tool was first registered. Automatically set to current time on INSERT.';

COMMENT ON COLUMN neurondb_agent.tools.updated_at IS 
'Timestamp when the tool was last modified. Automatically updated by trigger on UPDATE operations.';

-- ----------------------------------------------------------------------------
-- 3.6. jobs - Background Job Queue
-- ----------------------------------------------------------------------------
--
-- Background job queue for asynchronous task execution. Supports priority,
-- retries, and status tracking.
--
-- Usage Example:
--   INSERT INTO neurondb_agent.jobs (agent_id, type, status, priority, payload)
--   VALUES (
--     '123e4567-e89b-12d3-a456-426614174000'::uuid,
--     'http_call',
--     'queued',
--     5,
--     '{"url": "https://api.example.com/data", "method": "GET"}'::jsonb
--   );
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS neurondb_agent.jobs (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    type TEXT NOT NULL CHECK (type IN ('http_call', 'sql_task', 'shell_task', 'custom')),
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed', 'cancelled')),
    priority INT DEFAULT 0,
    payload JSONB NOT NULL,
    result JSONB,
    error_message TEXT,
    retry_count INT DEFAULT 0,
    max_retries INT DEFAULT 3,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

COMMENT ON TABLE neurondb_agent.jobs IS 
'Background job queue for asynchronous task execution. Supports multiple job types, priority scheduling, retry logic, and status tracking. Used for long-running or asynchronous operations.';

COMMENT ON COLUMN neurondb_agent.jobs.id IS 
'Unique sequential identifier for the job. Auto-incremented BIGSERIAL.';

COMMENT ON COLUMN neurondb_agent.jobs.agent_id IS 
'Optional foreign key reference to the agent that created or owns this job. Set to NULL on agent deletion (preserves job history).';

COMMENT ON COLUMN neurondb_agent.jobs.session_id IS 
'Optional foreign key reference to the session associated with this job. Set to NULL on session deletion.';

COMMENT ON COLUMN neurondb_agent.jobs.type IS 
'Type of job. Valid values: "http_call" (HTTP requests), "sql_task" (SQL execution), "shell_task" (shell commands), "custom" (custom job types). Required.';

COMMENT ON COLUMN neurondb_agent.jobs.status IS 
'Current status of the job. Valid values: "queued" (waiting to run), "running" (currently executing), "done" (completed successfully), "failed" (failed after retries), "cancelled" (manually cancelled). Default: "queued".';

COMMENT ON COLUMN neurondb_agent.jobs.priority IS 
'Job priority for scheduling. Higher numbers indicate higher priority. Jobs with higher priority are processed first. Default: 0. Can be negative for low-priority jobs.';

COMMENT ON COLUMN neurondb_agent.jobs.payload IS 
'JSONB object containing job-specific data and parameters. Structure depends on job type. Example for HTTP: {"url": "https://api.example.com", "method": "POST", "body": {...}}. Required.';

COMMENT ON COLUMN neurondb_agent.jobs.result IS 
'JSONB object containing the job execution result. Populated when status is "done". Structure depends on job type and success. NULL until job completes.';

COMMENT ON COLUMN neurondb_agent.jobs.error_message IS 
'Error message if the job failed. Populated when status is "failed". Contains error details for debugging. NULL for successful jobs.';

COMMENT ON COLUMN neurondb_agent.jobs.retry_count IS 
'Number of times this job has been retried. Incremented on each retry attempt. Default: 0.';

COMMENT ON COLUMN neurondb_agent.jobs.max_retries IS 
'Maximum number of retry attempts before marking job as failed. Default: 3. Set to 0 to disable retries.';

COMMENT ON COLUMN neurondb_agent.jobs.created_at IS 
'Timestamp when the job was created. Automatically set to current time on INSERT.';

COMMENT ON COLUMN neurondb_agent.jobs.updated_at IS 
'Timestamp when the job was last modified. Automatically updated by trigger on UPDATE operations.';

COMMENT ON COLUMN neurondb_agent.jobs.started_at IS 
'Timestamp when the job started executing. Set when status changes to "running". NULL until job starts.';

COMMENT ON COLUMN neurondb_agent.jobs.completed_at IS 
'Timestamp when the job completed (successfully or failed). Set when status changes to "done" or "failed". NULL until job completes.';

-- ----------------------------------------------------------------------------
-- 3.7. api_keys - Authentication
-- ----------------------------------------------------------------------------
--
-- API keys for authenticating requests to the NeuronAgent API. Supports
-- rate limiting, role-based access, and expiration.
--
-- Usage Example:
--   INSERT INTO neurondb_agent.api_keys (key_hash, key_prefix, 
--     rate_limit_per_minute, roles)
--   VALUES (
--     '$2a$10$...',  -- Bcrypt hash
--     'sk_live_',
--     100,
--     ARRAY['user', 'admin']
--   );
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS neurondb_agent.api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash TEXT NOT NULL UNIQUE,
    key_prefix TEXT NOT NULL,
    organization_id TEXT,
    user_id TEXT,
    rate_limit_per_minute INT DEFAULT 60,
    roles TEXT[] DEFAULT '{user}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    CONSTRAINT valid_roles CHECK (array_length(roles, 1) > 0)
);

COMMENT ON TABLE neurondb_agent.api_keys IS 
'Stores API keys for authenticating requests to the NeuronAgent API. Supports rate limiting, role-based access control, expiration, and usage tracking. Keys are stored as Bcrypt hashes for security.';

COMMENT ON COLUMN neurondb_agent.api_keys.id IS 
'Unique identifier for the API key record. Generated automatically using gen_random_uuid().';

COMMENT ON COLUMN neurondb_agent.api_keys.key_hash IS 
'Bcrypt hash of the API key. The actual key is never stored. Used for authentication by comparing provided key hash with stored hash. Must be unique.';

COMMENT ON COLUMN neurondb_agent.api_keys.key_prefix IS 
'First 8 characters of the original API key (before hashing). Used for identification and display purposes. Helps users identify which key they are using. Required.';

COMMENT ON COLUMN neurondb_agent.api_keys.organization_id IS 
'Optional organization identifier for multi-tenant setups. Can be used to group API keys by organization.';

COMMENT ON COLUMN neurondb_agent.api_keys.user_id IS 
'Optional user identifier linking the API key to a specific user. Used for user-specific access control and tracking.';

COMMENT ON COLUMN neurondb_agent.api_keys.rate_limit_per_minute IS 
'Rate limit for this API key in requests per minute. Used for throttling API usage. Default: 60 requests per minute.';

COMMENT ON COLUMN neurondb_agent.api_keys.roles IS 
'Array of roles assigned to this API key. Defines what permissions the key has. Common roles: "user" (standard access), "admin" (full access), "readonly" (read-only access). Must contain at least one role. Default: {"user"}.';

COMMENT ON COLUMN neurondb_agent.api_keys.metadata IS 
'JSONB object for storing API key metadata. Common uses: key name, description, tags, custom attributes. Example: {"name": "Production Key", "description": "Main production API key"}.';

COMMENT ON COLUMN neurondb_agent.api_keys.created_at IS 
'Timestamp when the API key was created. Automatically set to current time on INSERT.';

COMMENT ON COLUMN neurondb_agent.api_keys.last_used_at IS 
'Timestamp when the API key was last used for authentication. Updated on each successful authentication. NULL if never used. Used for monitoring and cleanup.';

COMMENT ON COLUMN neurondb_agent.api_keys.expires_at IS 
'Optional expiration timestamp for the API key. If set, the key becomes invalid after this time. NULL means the key never expires. Used for temporary keys and security.';

-- ============================================================================
-- SECTION 4: ADVANCED FEATURE TABLES
-- ============================================================================
--
-- Advanced feature tables provide extended functionality for multi-agent
-- collaboration, analytics, cost tracking, quality scoring, versioning,
-- planning, and reflection capabilities.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 4.1. agent_relationships - Multi-Agent Collaboration
-- ----------------------------------------------------------------------------
--
-- Defines relationships between agents for multi-agent collaboration.
-- Supports delegation, collaboration, supervision, and reporting hierarchies.
--
-- Usage Example:
--   INSERT INTO neurondb_agent.agent_relationships (from_agent_id, 
--     to_agent_id, relationship_type)
--   VALUES (
--     '123e4567-e89b-12d3-a456-426614174000'::uuid,
--     '123e4567-e89b-12d3-a456-426614174001'::uuid,
--     'delegates_to'
--   );
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS neurondb_agent.agent_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    to_agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL CHECK (relationship_type IN ('delegates_to', 'collaborates_with', 'supervises', 'reports_to')),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT different_agents CHECK (from_agent_id != to_agent_id)
);

COMMENT ON TABLE neurondb_agent.agent_relationships IS 
'Defines relationships between agents for multi-agent collaboration. Enables agents to delegate tasks, collaborate, supervise, or report to other agents. Supports complex agent hierarchies and workflows.';

COMMENT ON COLUMN neurondb_agent.agent_relationships.id IS 
'Unique identifier for the relationship. Generated automatically using gen_random_uuid().';

COMMENT ON COLUMN neurondb_agent.agent_relationships.from_agent_id IS 
'Foreign key reference to the source agent in the relationship. Required. When this agent is deleted, the relationship is cascade deleted.';

COMMENT ON COLUMN neurondb_agent.agent_relationships.to_agent_id IS 
'Foreign key reference to the target agent in the relationship. Required. Must be different from from_agent_id (agents cannot have relationships with themselves).';

COMMENT ON COLUMN neurondb_agent.agent_relationships.relationship_type IS 
'Type of relationship. Valid values: "delegates_to" (from_agent delegates tasks to to_agent), "collaborates_with" (agents work together), "supervises" (from_agent supervises to_agent), "reports_to" (from_agent reports to to_agent). Required.';

COMMENT ON COLUMN neurondb_agent.agent_relationships.metadata IS 
'JSONB object for storing relationship metadata. Common uses: delegation rules, collaboration parameters, supervision details, custom attributes. Example: {"delegation_threshold": 0.8, "auto_delegate": true}.';

COMMENT ON COLUMN neurondb_agent.agent_relationships.created_at IS 
'Timestamp when the relationship was created. Automatically set to current time on INSERT.';

-- ----------------------------------------------------------------------------
-- 4.2. tool_usage_logs - Tool Usage Analytics
-- ----------------------------------------------------------------------------
--
-- Logs tool usage for analytics, monitoring, and optimization.
-- Tracks execution time, success rates, token usage, and costs.
--
-- Usage Example:
--   INSERT INTO neurondb_agent.tool_usage_logs (agent_id, tool_name, 
--     execution_time_ms, success, tokens_used, cost)
--   VALUES (
--     '123e4567-e89b-12d3-a456-426614174000'::uuid,
--     'sql',
--     150,
--     true,
--     50,
--     0.001
--   );
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS neurondb_agent.tool_usage_logs (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    tool_name TEXT NOT NULL,
    execution_time_ms INTEGER,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    tokens_used INTEGER DEFAULT 0,
    cost REAL DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE neurondb_agent.tool_usage_logs IS 
'Logs tool usage for analytics and monitoring. Tracks execution metrics, success rates, token usage, and costs. Used for performance analysis, cost tracking, and tool optimization.';

COMMENT ON COLUMN neurondb_agent.tool_usage_logs.id IS 
'Unique sequential identifier for the log entry. Auto-incremented BIGSERIAL.';

COMMENT ON COLUMN neurondb_agent.tool_usage_logs.agent_id IS 
'Optional foreign key reference to the agent that used the tool. Set to NULL on agent deletion (preserves historical logs).';

COMMENT ON COLUMN neurondb_agent.tool_usage_logs.session_id IS 
'Optional foreign key reference to the session where the tool was used. Set to NULL on session deletion.';

COMMENT ON COLUMN neurondb_agent.tool_usage_logs.tool_name IS 
'Name of the tool that was used. Must match a tool name in the tools table. Required.';

COMMENT ON COLUMN neurondb_agent.tool_usage_logs.execution_time_ms IS 
'Tool execution time in milliseconds. Used for performance monitoring and optimization. NULL if not measured.';

COMMENT ON COLUMN neurondb_agent.tool_usage_logs.success IS 
'Whether the tool execution was successful. Default: true. Used for failure rate analysis.';

COMMENT ON COLUMN neurondb_agent.tool_usage_logs.error_message IS 
'Error message if the tool execution failed. NULL for successful executions. Used for debugging and error analysis.';

COMMENT ON COLUMN neurondb_agent.tool_usage_logs.tokens_used IS 
'Number of tokens consumed by the tool execution. Used for cost calculation and usage tracking. Default: 0.';

COMMENT ON COLUMN neurondb_agent.tool_usage_logs.cost IS 
'Cost of the tool execution in currency units. Used for cost tracking and budgeting. Default: 0.0.';

COMMENT ON COLUMN neurondb_agent.tool_usage_logs.created_at IS 
'Timestamp when the tool was used. Automatically set to current time on INSERT. Used for time-series analysis.';

-- ----------------------------------------------------------------------------
-- 4.3. cost_logs - Cost Tracking
-- ----------------------------------------------------------------------------
--
-- Tracks costs associated with agent operations including LLM usage,
-- embeddings, tools, storage, and other expenses.
--
-- Usage Example:
--   INSERT INTO neurondb_agent.cost_logs (agent_id, cost_type, tokens_used, 
--     cost, model_name)
--   VALUES (
--     '123e4567-e89b-12d3-a456-426614174000'::uuid,
--     'llm',
--     1000,
--     0.03,
--     'gpt-4'
--   );
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS neurondb_agent.cost_logs (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    cost_type TEXT NOT NULL CHECK (cost_type IN ('llm', 'embedding', 'tool', 'storage', 'other')),
    tokens_used INTEGER DEFAULT 0,
    cost REAL NOT NULL DEFAULT 0.0,
    model_name TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE neurondb_agent.cost_logs IS 
'Tracks costs associated with agent operations. Logs LLM usage, embedding generation, tool execution, storage, and other expenses. Used for cost analysis, budgeting, and billing.';

COMMENT ON COLUMN neurondb_agent.cost_logs.id IS 
'Unique sequential identifier for the cost log entry. Auto-incremented BIGSERIAL.';

COMMENT ON COLUMN neurondb_agent.cost_logs.agent_id IS 
'Optional foreign key reference to the agent that incurred the cost. Set to NULL on agent deletion (preserves cost history).';

COMMENT ON COLUMN neurondb_agent.cost_logs.session_id IS 
'Optional foreign key reference to the session where the cost was incurred. Set to NULL on session deletion.';

COMMENT ON COLUMN neurondb_agent.cost_logs.cost_type IS 
'Type of cost. Valid values: "llm" (LLM API calls), "embedding" (embedding generation), "tool" (tool execution), "storage" (storage costs), "other" (miscellaneous costs). Required.';

COMMENT ON COLUMN neurondb_agent.cost_logs.tokens_used IS 
'Number of tokens used (for LLM and embedding costs). Used for cost calculation and usage tracking. Default: 0.';

COMMENT ON COLUMN neurondb_agent.cost_logs.cost IS 
'Cost amount in currency units. Required. Used for cost aggregation and reporting.';

COMMENT ON COLUMN neurondb_agent.cost_logs.model_name IS 
'Name of the model used (for LLM and embedding costs). Examples: "gpt-4", "text-embedding-ada-002". NULL for non-model costs.';

COMMENT ON COLUMN neurondb_agent.cost_logs.metadata IS 
'JSONB object for storing cost metadata. Common uses: pricing tier, region, custom attributes. Example: {"tier": "premium", "region": "us-east-1"}.';

COMMENT ON COLUMN neurondb_agent.cost_logs.created_at IS 
'Timestamp when the cost was incurred. Automatically set to current time on INSERT. Used for time-series cost analysis.';

-- ----------------------------------------------------------------------------
-- 4.4. quality_scores - Response Quality Tracking
-- ----------------------------------------------------------------------------
--
-- Tracks quality scores for agent responses including accuracy, completeness,
-- clarity, relevance, and overall quality. Supports user feedback.
--
-- Usage Example:
--   INSERT INTO neurondb_agent.quality_scores (agent_id, message_id, 
--     overall_score, accuracy_score, user_feedback)
--   VALUES (
--     '123e4567-e89b-12d3-a456-426614174000'::uuid,
--     12345,
--     0.85,
--     0.90,
--     1
--   );
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS neurondb_agent.quality_scores (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    message_id BIGINT REFERENCES neurondb_agent.messages(id) ON DELETE SET NULL,
    overall_score REAL CHECK (overall_score >= 0 AND overall_score <= 1),
    accuracy_score REAL CHECK (accuracy_score >= 0 AND accuracy_score <= 1),
    completeness_score REAL CHECK (completeness_score >= 0 AND completeness_score <= 1),
    clarity_score REAL CHECK (clarity_score >= 0 AND clarity_score <= 1),
    relevance_score REAL CHECK (relevance_score >= 0 AND relevance_score <= 1),
    confidence REAL CHECK (confidence >= 0 AND confidence <= 1),
    issues JSONB DEFAULT '[]',
    user_feedback INTEGER CHECK (user_feedback >= -1 AND user_feedback <= 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE neurondb_agent.quality_scores IS 
'Tracks quality scores for agent responses. Measures accuracy, completeness, clarity, relevance, and overall quality. Supports automated scoring and user feedback. Used for quality monitoring and improvement.';

COMMENT ON COLUMN neurondb_agent.quality_scores.id IS 
'Unique sequential identifier for the quality score entry. Auto-incremented BIGSERIAL.';

COMMENT ON COLUMN neurondb_agent.quality_scores.agent_id IS 
'Optional foreign key reference to the agent that generated the response. Set to NULL on agent deletion.';

COMMENT ON COLUMN neurondb_agent.quality_scores.session_id IS 
'Optional foreign key reference to the session where the response was generated. Set to NULL on session deletion.';

COMMENT ON COLUMN neurondb_agent.quality_scores.message_id IS 
'Optional foreign key reference to the specific message being scored. Set to NULL on message deletion.';

COMMENT ON COLUMN neurondb_agent.quality_scores.overall_score IS 
'Overall quality score (0.0 to 1.0). Higher scores indicate better quality. NULL if not scored.';

COMMENT ON COLUMN neurondb_agent.quality_scores.accuracy_score IS 
'Accuracy score (0.0 to 1.0). Measures factual correctness and precision. NULL if not scored.';

COMMENT ON COLUMN neurondb_agent.quality_scores.completeness_score IS 
'Completeness score (0.0 to 1.0). Measures how complete and thorough the response is. NULL if not scored.';

COMMENT ON COLUMN neurondb_agent.quality_scores.clarity_score IS 
'Clarity score (0.0 to 1.0). Measures how clear and understandable the response is. NULL if not scored.';

COMMENT ON COLUMN neurondb_agent.quality_scores.relevance_score IS 
'Relevance score (0.0 to 1.0). Measures how relevant the response is to the query. NULL if not scored.';

COMMENT ON COLUMN neurondb_agent.quality_scores.confidence IS 
'Confidence score (0.0 to 1.0). Measures the confidence level of the quality assessment. NULL if not scored.';

COMMENT ON COLUMN neurondb_agent.quality_scores.issues IS 
'JSONB array of identified issues or problems with the response. Example: ["factual_error", "incomplete_answer", "unclear_terminology"]. Default: empty array.';

COMMENT ON COLUMN neurondb_agent.quality_scores.user_feedback IS 
'User feedback rating. Valid values: -1 (negative), 0 (neutral), 1 (positive). NULL if no user feedback provided.';

COMMENT ON COLUMN neurondb_agent.quality_scores.created_at IS 
'Timestamp when the quality score was recorded. Automatically set to current time on INSERT.';

-- ----------------------------------------------------------------------------
-- 4.5. agent_versions - Agent Versioning Support
-- ----------------------------------------------------------------------------
--
-- Stores version history of agents for versioning, rollback, and A/B testing.
-- Each version captures the complete agent configuration at a point in time.
--
-- Usage Example:
--   INSERT INTO neurondb_agent.agent_versions (agent_id, version_number, 
--     system_prompt, model_name, is_active)
--   VALUES (
--     '123e4567-e89b-12d3-a456-426614174000'::uuid,
--     2,
--     'Updated system prompt',
--     'gpt-4',
--     true
--   );
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS neurondb_agent.agent_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    name TEXT,
    description TEXT,
    system_prompt TEXT NOT NULL,
    model_name TEXT NOT NULL,
    enabled_tools TEXT[] DEFAULT '{}',
    config JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(agent_id, version_number)
);

COMMENT ON TABLE neurondb_agent.agent_versions IS 
'Stores version history of agents. Enables versioning, rollback, A/B testing, and configuration management. Each version captures the complete agent configuration at a point in time.';

COMMENT ON COLUMN neurondb_agent.agent_versions.id IS 
'Unique identifier for the version record. Generated automatically using gen_random_uuid().';

COMMENT ON COLUMN neurondb_agent.agent_versions.agent_id IS 
'Foreign key reference to the agent this version belongs to. Required. When an agent is deleted, all associated versions are cascade deleted.';

COMMENT ON COLUMN neurondb_agent.agent_versions.version_number IS 
'Version number for this agent version. Must be unique per agent. Typically incremented sequentially (1, 2, 3, ...). Required.';

COMMENT ON COLUMN neurondb_agent.agent_versions.name IS 
'Optional name for this version. Used for identification and display. Example: "v2.0", "production", "experimental".';

COMMENT ON COLUMN neurondb_agent.agent_versions.description IS 
'Optional description of changes or purpose of this version. Used for documentation and change tracking.';

COMMENT ON COLUMN neurondb_agent.agent_versions.system_prompt IS 
'System prompt for this version. Captures the prompt configuration at the time of version creation. Required.';

COMMENT ON COLUMN neurondb_agent.agent_versions.model_name IS 
'Model name for this version. Captures the model configuration at the time of version creation. Required.';

COMMENT ON COLUMN neurondb_agent.agent_versions.enabled_tools IS 
'Array of enabled tools for this version. Captures the tool configuration at the time of version creation. Default: empty array.';

COMMENT ON COLUMN neurondb_agent.agent_versions.config IS 
'JSONB configuration for this version. Captures the runtime configuration at the time of version creation. Default: empty object.';

COMMENT ON COLUMN neurondb_agent.agent_versions.is_active IS 
'Whether this version is currently active. Only one version per agent should be active at a time. Used for A/B testing and rollback. Default: false.';

COMMENT ON COLUMN neurondb_agent.agent_versions.created_at IS 
'Timestamp when this version was created. Automatically set to current time on INSERT.';

-- ----------------------------------------------------------------------------
-- 4.6. plans - Stored Plans
-- ----------------------------------------------------------------------------
--
-- Stores execution plans for complex multi-step tasks. Plans contain steps
-- and track execution status and results.
--
-- Usage Example:
--   INSERT INTO neurondb_agent.plans (agent_id, task_description, steps, status)
--   VALUES (
--     '123e4567-e89b-12d3-a456-426614174000'::uuid,
--     'Analyze sales data and generate report',
--     '[{"step": 1, "action": "query", "query": "SELECT * FROM sales"}, {"step": 2, "action": "analyze"}]'::jsonb,
--     'created'
--   );
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS neurondb_agent.plans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    task_description TEXT NOT NULL,
    steps JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'created' CHECK (status IN ('created', 'executing', 'completed', 'failed', 'cancelled')),
    result JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

COMMENT ON TABLE neurondb_agent.plans IS 
'Stores execution plans for complex multi-step tasks. Plans contain structured steps and track execution status and results. Used for task planning, execution tracking, and workflow management.';

COMMENT ON COLUMN neurondb_agent.plans.id IS 
'Unique identifier for the plan. Generated automatically using gen_random_uuid().';

COMMENT ON COLUMN neurondb_agent.plans.agent_id IS 
'Optional foreign key reference to the agent that created or owns this plan. Set to NULL on agent deletion.';

COMMENT ON COLUMN neurondb_agent.plans.session_id IS 
'Optional foreign key reference to the session associated with this plan. Set to NULL on session deletion.';

COMMENT ON COLUMN neurondb_agent.plans.task_description IS 
'Human-readable description of the task this plan addresses. Required. Example: "Analyze sales data and generate quarterly report".';

COMMENT ON COLUMN neurondb_agent.plans.steps IS 
'JSONB array of plan steps. Each step typically contains: step number, action, parameters, dependencies. Example: [{"step": 1, "action": "query", "query": "SELECT * FROM sales"}, {"step": 2, "action": "analyze", "depends_on": [1]}]. Required.';

COMMENT ON COLUMN neurondb_agent.plans.status IS 
'Current status of the plan. Valid values: "created" (plan created but not started), "executing" (plan is being executed), "completed" (plan completed successfully), "failed" (plan execution failed), "cancelled" (plan was cancelled). Default: "created".';

COMMENT ON COLUMN neurondb_agent.plans.result IS 
'JSONB object containing the plan execution result. Populated when status is "completed" or "failed". Contains step results, final output, and any errors. NULL until plan completes.';

COMMENT ON COLUMN neurondb_agent.plans.created_at IS 
'Timestamp when the plan was created. Automatically set to current time on INSERT.';

COMMENT ON COLUMN neurondb_agent.plans.updated_at IS 
'Timestamp when the plan was last modified. Automatically updated on status changes.';

COMMENT ON COLUMN neurondb_agent.plans.completed_at IS 
'Timestamp when the plan completed (successfully or failed). Set when status changes to "completed" or "failed". NULL until plan completes.';

-- ----------------------------------------------------------------------------
-- 4.7. reflections - Reflection Logs
-- ----------------------------------------------------------------------------
--
-- Stores reflection logs for agent self-evaluation and improvement.
-- Contains quality scores, issues, suggestions, and retry tracking.
--
-- Usage Example:
--   INSERT INTO neurondb_agent.reflections (agent_id, user_message, 
--     agent_response, quality_score, was_retried)
--   VALUES (
--     '123e4567-e89b-12d3-a456-426614174000'::uuid,
--     'What is the capital of France?',
--     'The capital of France is Paris.',
--     0.95,
--     false
--   );
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS neurondb_agent.reflections (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    message_id BIGINT REFERENCES neurondb_agent.messages(id) ON DELETE SET NULL,
    user_message TEXT NOT NULL,
    agent_response TEXT NOT NULL,
    quality_score REAL CHECK (quality_score >= 0 AND quality_score <= 1),
    accuracy_score REAL CHECK (accuracy_score >= 0 AND accuracy_score <= 1),
    completeness_score REAL CHECK (completeness_score >= 0 AND completeness_score <= 1),
    clarity_score REAL CHECK (clarity_score >= 0 AND clarity_score <= 1),
    relevance_score REAL CHECK (relevance_score >= 0 AND relevance_score <= 1),
    confidence REAL CHECK (confidence >= 0 AND confidence <= 1),
    issues JSONB DEFAULT '[]',
    suggestions JSONB DEFAULT '[]',
    was_retried BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE neurondb_agent.reflections IS 
'Stores reflection logs for agent self-evaluation and improvement. Contains quality assessments, identified issues, improvement suggestions, and retry tracking. Used for continuous learning and quality improvement.';

COMMENT ON COLUMN neurondb_agent.reflections.id IS 
'Unique sequential identifier for the reflection entry. Auto-incremented BIGSERIAL.';

COMMENT ON COLUMN neurondb_agent.reflections.agent_id IS 
'Optional foreign key reference to the agent that generated the response. Set to NULL on agent deletion.';

COMMENT ON COLUMN neurondb_agent.reflections.session_id IS 
'Optional foreign key reference to the session where the reflection occurred. Set to NULL on session deletion.';

COMMENT ON COLUMN neurondb_agent.reflections.message_id IS 
'Optional foreign key reference to the specific message being reflected upon. Set to NULL on message deletion.';

COMMENT ON COLUMN neurondb_agent.reflections.user_message IS 
'The original user message that prompted the agent response. Required. Used for context in reflection analysis.';

COMMENT ON COLUMN neurondb_agent.reflections.agent_response IS 
'The agent response that is being reflected upon. Required. Used for quality assessment and improvement.';

COMMENT ON COLUMN neurondb_agent.reflections.quality_score IS 
'Overall quality score (0.0 to 1.0). Higher scores indicate better quality. NULL if not scored.';

COMMENT ON COLUMN neurondb_agent.reflections.accuracy_score IS 
'Accuracy score (0.0 to 1.0). Measures factual correctness. NULL if not scored.';

COMMENT ON COLUMN neurondb_agent.reflections.completeness_score IS 
'Completeness score (0.0 to 1.0). Measures thoroughness. NULL if not scored.';

COMMENT ON COLUMN neurondb_agent.reflections.clarity_score IS 
'Clarity score (0.0 to 1.0). Measures understandability. NULL if not scored.';

COMMENT ON COLUMN neurondb_agent.reflections.relevance_score IS 
'Relevance score (0.0 to 1.0). Measures relevance to the query. NULL if not scored.';

COMMENT ON COLUMN neurondb_agent.reflections.confidence IS 
'Confidence score (0.0 to 1.0). Measures confidence in the quality assessment. NULL if not scored.';

COMMENT ON COLUMN neurondb_agent.reflections.issues IS 
'JSONB array of identified issues or problems. Example: ["factual_error", "incomplete", "unclear"]. Default: empty array.';

COMMENT ON COLUMN neurondb_agent.reflections.suggestions IS 
'JSONB array of improvement suggestions. Example: ["add_more_context", "clarify_terminology", "provide_examples"]. Default: empty array.';

COMMENT ON COLUMN neurondb_agent.reflections.was_retried IS 
'Whether the response was retried after reflection. Indicates that the agent attempted to improve the response. Default: false.';

COMMENT ON COLUMN neurondb_agent.reflections.created_at IS 
'Timestamp when the reflection was recorded. Automatically set to current time on INSERT.';

-- ============================================================================
-- SECTION 5: SCHEMA MIGRATIONS TABLE
-- ============================================================================
--
-- Tracks applied schema migrations for version control and migration management.
-- Used by the SchemaManager to determine which migrations have been applied.
-- ============================================================================

CREATE TABLE IF NOT EXISTS neurondb_agent.schema_migrations (
    version INT PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE neurondb_agent.schema_migrations IS 
'Tracks applied schema migrations for version control. Used by SchemaManager to determine which migrations have been applied and which are pending. Each migration is recorded with its version number and name.';

COMMENT ON COLUMN neurondb_agent.schema_migrations.version IS 
'Migration version number. Primary key. Used to order migrations and track which have been applied. Example: 1, 2, 3, 4.';

COMMENT ON COLUMN neurondb_agent.schema_migrations.name IS 
'Migration name or description. Used for identification and documentation. Example: "initial_schema", "add_indexes", "add_triggers". Required.';

COMMENT ON COLUMN neurondb_agent.schema_migrations.applied_at IS 
'Timestamp when the migration was applied. Automatically set to current time on INSERT. Used for tracking when migrations were applied.';

-- ============================================================================
-- SECTION 6: FUNCTIONS
-- ============================================================================
--
-- Database functions for automatic updates and activity tracking.
-- These functions are called by triggers to maintain data consistency.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 6.1. update_updated_at() - Automatic Timestamp Updates
-- ----------------------------------------------------------------------------
--
-- Automatically updates the updated_at column when a row is updated.
-- Used by triggers on tables with updated_at columns.
--
-- Usage:
--   Trigger automatically calls this function on UPDATE operations.
--   No manual invocation needed.
-- ----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION neurondb_agent.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION neurondb_agent.update_updated_at() IS 
'Automatically updates the updated_at column to the current timestamp when a row is updated. Used by triggers on agents, tools, jobs, and plans tables. Ensures updated_at always reflects the last modification time.';

-- ----------------------------------------------------------------------------
-- 6.2. update_session_activity() - Session Activity Tracking
-- ----------------------------------------------------------------------------
--
-- Updates the last_activity_at timestamp of a session when a message
-- is inserted. Keeps session activity tracking up-to-date automatically.
--
-- Usage:
--   Trigger automatically calls this function on INSERT into messages table.
--   No manual invocation needed.
-- ----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION neurondb_agent.update_session_activity()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE neurondb_agent.sessions
    SET last_activity_at = NOW()
    WHERE id = NEW.session_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION neurondb_agent.update_session_activity() IS 
'Updates the last_activity_at timestamp of a session when a message is inserted. Called automatically by trigger on messages table. Ensures session activity tracking is always current for cleanup and monitoring purposes.';

-- ============================================================================
-- SECTION 7: TRIGGERS
-- ============================================================================
--
-- Database triggers for automatic data maintenance and consistency.
-- Triggers fire automatically on INSERT or UPDATE operations.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 7.1. agents_updated_at - Auto-update agents.updated_at
-- ----------------------------------------------------------------------------
--
-- Automatically updates the updated_at column when an agent is modified.
-- Fires: BEFORE UPDATE on agents table
-- Action: Sets updated_at to current timestamp
-- ----------------------------------------------------------------------------

DROP TRIGGER IF EXISTS agents_updated_at ON neurondb_agent.agents;
CREATE TRIGGER agents_updated_at 
    BEFORE UPDATE ON neurondb_agent.agents
    FOR EACH ROW 
    EXECUTE FUNCTION neurondb_agent.update_updated_at();

COMMENT ON TRIGGER agents_updated_at ON neurondb_agent.agents IS 
'Automatically updates the updated_at column when an agent row is updated. Fires before UPDATE operations. Ensures updated_at always reflects the last modification time without manual intervention.';

-- ----------------------------------------------------------------------------
-- 7.2. tools_updated_at - Auto-update tools.updated_at
-- ----------------------------------------------------------------------------
--
-- Automatically updates the updated_at column when a tool is modified.
-- Fires: BEFORE UPDATE on tools table
-- Action: Sets updated_at to current timestamp
-- ----------------------------------------------------------------------------

DROP TRIGGER IF EXISTS tools_updated_at ON neurondb_agent.tools;
CREATE TRIGGER tools_updated_at 
    BEFORE UPDATE ON neurondb_agent.tools
    FOR EACH ROW 
    EXECUTE FUNCTION neurondb_agent.update_updated_at();

COMMENT ON TRIGGER tools_updated_at ON neurondb_agent.tools IS 
'Automatically updates the updated_at column when a tool row is updated. Fires before UPDATE operations. Ensures updated_at always reflects the last modification time without manual intervention.';

-- ----------------------------------------------------------------------------
-- 7.3. jobs_updated_at - Auto-update jobs.updated_at
-- ----------------------------------------------------------------------------
--
-- Automatically updates the updated_at column when a job is modified.
-- Fires: BEFORE UPDATE on jobs table
-- Action: Sets updated_at to current timestamp
-- ----------------------------------------------------------------------------

DROP TRIGGER IF EXISTS jobs_updated_at ON neurondb_agent.jobs;
CREATE TRIGGER jobs_updated_at 
    BEFORE UPDATE ON neurondb_agent.jobs
    FOR EACH ROW 
    EXECUTE FUNCTION neurondb_agent.update_updated_at();

COMMENT ON TRIGGER jobs_updated_at ON neurondb_agent.jobs IS 
'Automatically updates the updated_at column when a job row is updated. Fires before UPDATE operations. Ensures updated_at always reflects the last modification time without manual intervention.';

-- ----------------------------------------------------------------------------
-- 7.4. messages_session_activity - Update session.last_activity_at
-- ----------------------------------------------------------------------------
--
-- Automatically updates the session's last_activity_at when a message is inserted.
-- Fires: AFTER INSERT on messages table
-- Action: Updates the associated session's last_activity_at to current timestamp
-- ----------------------------------------------------------------------------

DROP TRIGGER IF EXISTS messages_session_activity ON neurondb_agent.messages;
CREATE TRIGGER messages_session_activity 
    AFTER INSERT ON neurondb_agent.messages
    FOR EACH ROW 
    EXECUTE FUNCTION neurondb_agent.update_session_activity();

COMMENT ON TRIGGER messages_session_activity ON neurondb_agent.messages IS 
'Automatically updates the session''s last_activity_at timestamp when a message is inserted. Fires after INSERT operations. Ensures session activity tracking is always current for cleanup and monitoring purposes.';

-- ============================================================================
-- SECTION 8: INDEXES
-- ============================================================================
--
-- Performance indexes for efficient query execution.
-- Indexes are organized by table and optimized for common query patterns.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 8.1. Core Table Indexes
-- ----------------------------------------------------------------------------

-- Sessions indexes
-- Optimizes queries filtering by agent_id and ordering by activity
CREATE INDEX IF NOT EXISTS idx_sessions_agent_id ON neurondb_agent.sessions(agent_id);
COMMENT ON INDEX neurondb_agent.idx_sessions_agent_id IS 
'Index on sessions.agent_id for efficient lookup of all sessions for a specific agent. Used in queries like: SELECT * FROM sessions WHERE agent_id = ? ORDER BY last_activity_at DESC.';

CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON neurondb_agent.sessions(last_activity_at);
COMMENT ON INDEX neurondb_agent.idx_sessions_last_activity IS 
'Index on sessions.last_activity_at for efficient ordering and filtering by activity time. Used for session cleanup, finding inactive sessions, and activity-based queries.';

-- Messages indexes
-- Optimizes queries filtering by session_id and role, ordering by created_at
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON neurondb_agent.messages(session_id, created_at DESC);
COMMENT ON INDEX neurondb_agent.idx_messages_session_id IS 
'Composite index on messages(session_id, created_at DESC) for efficient retrieval of messages in a session ordered by time. Used in queries like: SELECT * FROM messages WHERE session_id = ? ORDER BY created_at DESC.';

CREATE INDEX IF NOT EXISTS idx_messages_session_role ON neurondb_agent.messages(session_id, role);
COMMENT ON INDEX neurondb_agent.idx_messages_session_role IS 
'Composite index on messages(session_id, role) for efficient filtering of messages by role within a session. Used in queries like: SELECT * FROM messages WHERE session_id = ? AND role = ''user''.';

-- Memory chunks indexes
-- Optimizes queries filtering by agent_id and session_id
CREATE INDEX IF NOT EXISTS idx_memory_chunks_agent_id ON neurondb_agent.memory_chunks(agent_id);
COMMENT ON INDEX neurondb_agent.idx_memory_chunks_agent_id IS 
'Index on memory_chunks.agent_id for efficient lookup of all memory chunks for a specific agent. Used for agent-specific memory retrieval and cleanup.';

CREATE INDEX IF NOT EXISTS idx_memory_chunks_session_id ON neurondb_agent.memory_chunks(session_id);
COMMENT ON INDEX neurondb_agent.idx_memory_chunks_session_id IS 
'Index on memory_chunks.session_id for efficient lookup of memory chunks for a specific session. Used for session-specific memory retrieval.';

-- Jobs indexes
-- Optimizes queries filtering by status and ordering by created_at (partial index)
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON neurondb_agent.jobs(status, created_at) 
    WHERE status IN ('queued', 'running');
COMMENT ON INDEX neurondb_agent.idx_jobs_status_created IS 
'Partial composite index on jobs(status, created_at) for active jobs only. Optimizes job queue queries that fetch queued or running jobs ordered by creation time. Significantly improves job worker performance.';

CREATE INDEX IF NOT EXISTS idx_jobs_agent_session ON neurondb_agent.jobs(agent_id, session_id);
COMMENT ON INDEX neurondb_agent.idx_jobs_agent_session IS 
'Composite index on jobs(agent_id, session_id) for efficient lookup of jobs by agent and session. Used for agent-specific and session-specific job queries.';

-- API keys indexes
-- Optimizes API key lookup by prefix
CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON neurondb_agent.api_keys(key_prefix);
COMMENT ON INDEX neurondb_agent.idx_api_keys_prefix IS 
'Index on api_keys.key_prefix for efficient API key lookup by prefix. Used during authentication to quickly find potential key matches before hash verification.';

-- Agents indexes (from advanced features)
-- Optimizes queries filtering by version and parent_agent_id
CREATE INDEX IF NOT EXISTS idx_agents_version ON neurondb_agent.agents(version);
COMMENT ON INDEX neurondb_agent.idx_agents_version IS 
'Index on agents.version for efficient filtering and grouping by agent version. Used for version-based queries and analytics.';

CREATE INDEX IF NOT EXISTS idx_agents_parent ON neurondb_agent.agents(parent_agent_id);
COMMENT ON INDEX neurondb_agent.idx_agents_parent IS 
'Index on agents.parent_agent_id for efficient lookup of child agents. Used for agent hierarchy queries and finding all agents derived from a parent.';

-- ----------------------------------------------------------------------------
-- 8.2. Advanced Feature Indexes
-- ----------------------------------------------------------------------------

-- Agent relationships indexes
CREATE INDEX IF NOT EXISTS idx_agent_relationships_from ON neurondb_agent.agent_relationships(from_agent_id);
COMMENT ON INDEX neurondb_agent.idx_agent_relationships_from IS 
'Index on agent_relationships.from_agent_id for efficient lookup of relationships where an agent is the source. Used in queries like: SELECT * FROM agent_relationships WHERE from_agent_id = ?.';

CREATE INDEX IF NOT EXISTS idx_agent_relationships_to ON neurondb_agent.agent_relationships(to_agent_id);
COMMENT ON INDEX neurondb_agent.idx_agent_relationships_to IS 
'Index on agent_relationships.to_agent_id for efficient lookup of relationships where an agent is the target. Used in queries like: SELECT * FROM agent_relationships WHERE to_agent_id = ?.';

-- Tool usage logs indexes
CREATE INDEX IF NOT EXISTS idx_tool_usage_agent ON neurondb_agent.tool_usage_logs(agent_id, created_at);
COMMENT ON INDEX neurondb_agent.idx_tool_usage_agent IS 
'Composite index on tool_usage_logs(agent_id, created_at) for efficient time-series analysis of tool usage by agent. Used for agent-specific analytics and performance monitoring.';

CREATE INDEX IF NOT EXISTS idx_tool_usage_tool ON neurondb_agent.tool_usage_logs(tool_name, created_at);
COMMENT ON INDEX neurondb_agent.idx_tool_usage_tool IS 
'Composite index on tool_usage_logs(tool_name, created_at) for efficient time-series analysis of tool usage by tool name. Used for tool-specific analytics and performance monitoring.';

CREATE INDEX IF NOT EXISTS idx_tool_usage_session ON neurondb_agent.tool_usage_logs(session_id, created_at);
COMMENT ON INDEX neurondb_agent.idx_tool_usage_session IS 
'Composite index on tool_usage_logs(session_id, created_at) for efficient time-series analysis of tool usage by session. Used for session-specific analytics.';

-- Cost logs indexes
CREATE INDEX IF NOT EXISTS idx_cost_logs_agent ON neurondb_agent.cost_logs(agent_id, created_at);
COMMENT ON INDEX neurondb_agent.idx_cost_logs_agent IS 
'Composite index on cost_logs(agent_id, created_at) for efficient time-series cost analysis by agent. Used for agent-specific cost tracking and budgeting.';

CREATE INDEX IF NOT EXISTS idx_cost_logs_session ON neurondb_agent.cost_logs(session_id, created_at);
COMMENT ON INDEX neurondb_agent.idx_cost_logs_session IS 
'Composite index on cost_logs(session_id, created_at) for efficient time-series cost analysis by session. Used for session-specific cost tracking.';

CREATE INDEX IF NOT EXISTS idx_cost_logs_type ON neurondb_agent.cost_logs(cost_type, created_at);
COMMENT ON INDEX neurondb_agent.idx_cost_logs_type IS 
'Composite index on cost_logs(cost_type, created_at) for efficient time-series cost analysis by cost type. Used for cost breakdown analysis (LLM, embedding, tool, etc.).';

-- Quality scores indexes
CREATE INDEX IF NOT EXISTS idx_quality_scores_agent ON neurondb_agent.quality_scores(agent_id, created_at);
COMMENT ON INDEX neurondb_agent.idx_quality_scores_agent IS 
'Composite index on quality_scores(agent_id, created_at) for efficient time-series quality analysis by agent. Used for agent-specific quality monitoring and improvement tracking.';

CREATE INDEX IF NOT EXISTS idx_quality_scores_session ON neurondb_agent.quality_scores(session_id, created_at);
COMMENT ON INDEX neurondb_agent.idx_quality_scores_session IS 
'Composite index on quality_scores(session_id, created_at) for efficient time-series quality analysis by session. Used for session-specific quality tracking.';

CREATE INDEX IF NOT EXISTS idx_quality_scores_overall ON neurondb_agent.quality_scores(overall_score);
COMMENT ON INDEX neurondb_agent.idx_quality_scores_overall IS 
'Index on quality_scores.overall_score for efficient filtering and sorting by quality. Used for finding high-quality or low-quality responses, quality distribution analysis.';

-- Agent versions indexes
CREATE INDEX IF NOT EXISTS idx_agent_versions_agent ON neurondb_agent.agent_versions(agent_id, version_number DESC);
COMMENT ON INDEX neurondb_agent.idx_agent_versions_agent IS 
'Composite index on agent_versions(agent_id, version_number DESC) for efficient retrieval of agent versions ordered by version number. Used for version history queries and finding latest versions.';

CREATE INDEX IF NOT EXISTS idx_agent_versions_active ON neurondb_agent.agent_versions(agent_id, is_active) 
    WHERE is_active = true;
COMMENT ON INDEX neurondb_agent.idx_agent_versions_active IS 
'Partial composite index on agent_versions(agent_id, is_active) for active versions only. Optimizes queries that find active versions of agents. Significantly improves A/B testing and version lookup performance.';

-- Plans indexes
CREATE INDEX IF NOT EXISTS idx_plans_agent ON neurondb_agent.plans(agent_id, created_at DESC);
COMMENT ON INDEX neurondb_agent.idx_plans_agent IS 
'Composite index on plans(agent_id, created_at DESC) for efficient retrieval of plans by agent ordered by creation time. Used for agent-specific plan history queries.';

CREATE INDEX IF NOT EXISTS idx_plans_session ON neurondb_agent.plans(session_id, created_at DESC);
COMMENT ON INDEX neurondb_agent.idx_plans_session IS 
'Composite index on plans(session_id, created_at DESC) for efficient retrieval of plans by session ordered by creation time. Used for session-specific plan history queries.';

CREATE INDEX IF NOT EXISTS idx_plans_status ON neurondb_agent.plans(status, created_at);
COMMENT ON INDEX neurondb_agent.idx_plans_status IS 
'Composite index on plans(status, created_at) for efficient filtering and ordering of plans by status. Used for finding plans in specific states (e.g., all executing plans).';

-- Reflections indexes
CREATE INDEX IF NOT EXISTS idx_reflections_agent ON neurondb_agent.reflections(agent_id, created_at DESC);
COMMENT ON INDEX neurondb_agent.idx_reflections_agent IS 
'Composite index on reflections(agent_id, created_at DESC) for efficient retrieval of reflections by agent ordered by creation time. Used for agent-specific reflection history and analysis.';

CREATE INDEX IF NOT EXISTS idx_reflections_session ON neurondb_agent.reflections(session_id, created_at DESC);
COMMENT ON INDEX neurondb_agent.idx_reflections_session IS 
'Composite index on reflections(session_id, created_at DESC) for efficient retrieval of reflections by session ordered by creation time. Used for session-specific reflection history.';

CREATE INDEX IF NOT EXISTS idx_reflections_quality ON neurondb_agent.reflections(quality_score);
COMMENT ON INDEX neurondb_agent.idx_reflections_quality IS 
'Index on reflections.quality_score for efficient filtering and sorting by quality. Used for finding high-quality or low-quality reflections, quality distribution analysis.';

-- ----------------------------------------------------------------------------
-- 8.3. HNSW Vector Index
-- ----------------------------------------------------------------------------
--
-- HNSW (Hierarchical Navigable Small World) index for efficient approximate
-- nearest neighbor search on vector embeddings. This is a specialized index
-- provided by the NeuronDB extension for high-dimensional vector similarity search.
--
-- Parameters:
--   - m = 16: Number of bi-directional links for each element (controls graph connectivity)
--   - ef_construction = 64: Size of the candidate list during index construction
--   - vector_cosine_ops: Operator class for cosine similarity search
--
-- Performance Notes:
--   - Provides O(log n) search time for approximate nearest neighbors
--   - Optimized for high-dimensional vectors (768 dimensions)
--   - Supports cosine similarity, L2 distance, and inner product
--   - Index size grows with number of vectors but provides fast search
-- ----------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_memory_chunks_embedding_hnsw ON neurondb_agent.memory_chunks 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

COMMENT ON INDEX neurondb_agent.idx_memory_chunks_embedding_hnsw IS 
'HNSW (Hierarchical Navigable Small World) index on memory_chunks.embedding for efficient approximate nearest neighbor search. Uses cosine similarity operator class. Optimized for 768-dimensional vectors. Provides O(log n) search time for similarity queries. Parameters: m=16 (graph connectivity), ef_construction=64 (construction quality). Used for semantic memory retrieval queries like: SELECT * FROM memory_chunks ORDER BY embedding <=> query_vector LIMIT 10.';

-- ============================================================================
-- SECTION 9: FOOTER AND VERIFICATION
-- ============================================================================
--
-- Verification queries and usage examples for validating the schema installation
-- and understanding common usage patterns.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Verification Queries
-- ----------------------------------------------------------------------------
--
-- Run these queries after schema installation to verify everything is set up correctly:
--
-- 1. Check schema exists:
--    SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'neurondb_agent';
--
-- 2. Check all tables exist:
--    SELECT table_name FROM information_schema.tables 
--    WHERE table_schema = 'neurondb_agent' ORDER BY table_name;
--
-- 3. Check all functions exist:
--    SELECT routine_name FROM information_schema.routines 
--    WHERE routine_schema = 'neurondb_agent' ORDER BY routine_name;
--
-- 4. Check all triggers exist:
--    SELECT trigger_name, event_object_table FROM information_schema.triggers 
--    WHERE trigger_schema = 'neurondb_agent' ORDER BY event_object_table, trigger_name;
--
-- 5. Check all indexes exist:
--    SELECT indexname FROM pg_indexes 
--    WHERE schemaname = 'neurondb_agent' ORDER BY tablename, indexname;
--
-- 6. Check NeuronDB extension:
--    SELECT * FROM pg_extension WHERE extname = 'neurondb';
--
-- 7. Verify table counts (should all be 0 for fresh installation):
--    SELECT 
--        (SELECT COUNT(*) FROM neurondb_agent.agents) as agents,
--        (SELECT COUNT(*) FROM neurondb_agent.sessions) as sessions,
--        (SELECT COUNT(*) FROM neurondb_agent.messages) as messages,
--        (SELECT COUNT(*) FROM neurondb_agent.memory_chunks) as memory_chunks,
--        (SELECT COUNT(*) FROM neurondb_agent.tools) as tools,
--        (SELECT COUNT(*) FROM neurondb_agent.jobs) as jobs,
--        (SELECT COUNT(*) FROM neurondb_agent.api_keys) as api_keys;
--
-- ----------------------------------------------------------------------------
-- Common Usage Examples
-- ----------------------------------------------------------------------------
--
-- 1. Create an agent:
--    INSERT INTO neurondb_agent.agents (name, description, system_prompt, 
--      model_name, enabled_tools, config)
--    VALUES (
--      'general-assistant',
--      'General purpose assistant',
--      'You are a helpful, harmless, and honest assistant.',
--      'gpt-4',
--      ARRAY['sql', 'http'],
--      '{"temperature": 0.7, "max_tokens": 1000, "top_p": 0.9}'::jsonb
--    );
--
-- 2. Create a session:
--    INSERT INTO neurondb_agent.sessions (agent_id, external_user_id, metadata)
--    VALUES (
--      (SELECT id FROM neurondb_agent.agents WHERE name = 'general-assistant'),
--      'user-123',
--      '{"source": "web", "ip": "192.168.1.1"}'::jsonb
--    );
--
-- 3. Add a message:
--    INSERT INTO neurondb_agent.messages (session_id, role, content, token_count)
--    VALUES (
--      (SELECT id FROM neurondb_agent.sessions ORDER BY created_at DESC LIMIT 1),
--      'user',
--      'Hello, how can you help me?',
--      10
--    );
--
-- 4. Search memory chunks (vector similarity):
--    SELECT id, content, 1 - (embedding <=> query_vector::neurondb_vector(768)) AS similarity
--    FROM neurondb_agent.memory_chunks
--    WHERE agent_id = '123e4567-e89b-12d3-a456-426614174000'::uuid
--    ORDER BY embedding <=> query_vector::neurondb_vector(768)
--    LIMIT 10;
--
-- 5. Get active jobs:
--    SELECT * FROM neurondb_agent.jobs
--    WHERE status IN ('queued', 'running')
--    ORDER BY priority DESC, created_at ASC
--    LIMIT 10;
--
-- 6. Track tool usage:
--    INSERT INTO neurondb_agent.tool_usage_logs (agent_id, tool_name, 
--      execution_time_ms, success, tokens_used, cost)
--    VALUES (
--      '123e4567-e89b-12d3-a456-426614174000'::uuid,
--      'sql',
--      150,
--      true,
--      50,
--      0.001
--    );
--
-- 7. Log costs:
--    INSERT INTO neurondb_agent.cost_logs (agent_id, cost_type, tokens_used, 
--      cost, model_name)
--    VALUES (
--      '123e4567-e89b-12d3-a456-426614174000'::uuid,
--      'llm',
--      1000,
--      0.03,
--      'gpt-4'
--    );
--
-- 8. Query quality scores:
--    SELECT 
--        agent_id,
--        AVG(overall_score) as avg_quality,
--        COUNT(*) as total_responses
--    FROM neurondb_agent.quality_scores
--    WHERE created_at >= NOW() - INTERVAL '7 days'
--    GROUP BY agent_id
--    ORDER BY avg_quality DESC;
--
-- ----------------------------------------------------------------------------
-- Maintenance Notes
-- ----------------------------------------------------------------------------
--
-- 1. Regular Maintenance:
--    - Monitor index bloat: Use pg_stat_user_indexes to check index usage
--    - Vacuum regularly: VACUUM ANALYZE neurondb_agent.*; (or per table)
--    - Monitor table sizes: SELECT pg_size_pretty(pg_total_relation_size('neurondb_agent.tablename'));
--
-- 2. Performance Optimization:
--    - Monitor slow queries: Enable pg_stat_statements extension
--    - Review index usage: Check pg_stat_user_indexes for unused indexes
--    - Consider partitioning large tables (messages, tool_usage_logs) by date
--
-- 3. Backup Considerations:
--    - Include schema_migrations table in backups
--    - Consider separate backup strategies for large tables (messages, memory_chunks)
--    - Vector indexes (HNSW) are large; factor into backup size estimates
--
-- 4. Security:
--    - Review API key permissions regularly
--    - Monitor api_keys.last_used_at for inactive keys
--    - Implement key rotation for expired keys
--    - Use row-level security (RLS) if multi-tenant isolation is needed
--
-- 5. Scaling:
--    - Consider read replicas for read-heavy workloads
--    - Partition large tables (messages, tool_usage_logs, cost_logs) by date
--    - Archive old data to separate tables or storage
--    - Monitor connection pool usage
--
-- ----------------------------------------------------------------------------
-- Schema Version Information
-- ----------------------------------------------------------------------------
--
-- This schema combines the following migration files:
--   - 001_initial_schema.sql (Core tables)
--   - 002_add_indexes.sql (Performance indexes)
--   - 003_add_triggers.sql (Functions and triggers)
--   - 004_advanced_features.sql (Advanced feature tables)
--
-- Schema Version: 1.0
-- Last Updated: 2024-2025
-- Compatible with: PostgreSQL 16+, NeuronDB extension
--
-- ============================================================================
-- END OF SCHEMA
-- ============================================================================

