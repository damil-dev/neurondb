-- ============================================================================
-- NeuronAgent Principals and Permissions Schema Migration
-- ============================================================================
-- This migration creates the principals system and permissions infrastructure.
-- Prerequisites: Migration 001_initial_schema.sql must be run first
-- ============================================================================

-- Principals table: Represents entities that can have permissions (users, orgs, agents, tools, datasets)
CREATE TABLE IF NOT EXISTS neurondb_agent.principals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type TEXT NOT NULL CHECK (type IN ('user', 'org', 'agent', 'tool', 'dataset')),
    name TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_principal_name_per_type UNIQUE (type, name)
);

-- Link API keys to principals
ALTER TABLE neurondb_agent.api_keys 
    ADD COLUMN IF NOT EXISTS principal_id UUID REFERENCES neurondb_agent.principals(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_api_keys_principal_id ON neurondb_agent.api_keys(principal_id);

-- Policies table: Defines permissions for principals
CREATE TABLE IF NOT EXISTS neurondb_agent.policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    principal_id UUID NOT NULL REFERENCES neurondb_agent.principals(id) ON DELETE CASCADE,
    resource_type TEXT NOT NULL,  -- e.g., 'agent', 'tool', 'dataset', 'schema', 'table'
    resource_id TEXT,  -- NULL for wildcard policies
    permissions TEXT[] NOT NULL DEFAULT '{}',  -- e.g., ['read', 'write', 'execute']
    conditions JSONB DEFAULT '{}',  -- ABAC conditions (e.g., {"tags": {"project": "finance", "pii": "true"}})
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_permissions CHECK (array_length(permissions, 1) > 0)
);

CREATE INDEX IF NOT EXISTS idx_policies_principal_id ON neurondb_agent.policies(principal_id);
CREATE INDEX IF NOT EXISTS idx_policies_resource ON neurondb_agent.policies(resource_type, resource_id);

-- Tool permissions per agent
CREATE TABLE IF NOT EXISTS neurondb_agent.tool_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    tool_name TEXT NOT NULL REFERENCES neurondb_agent.tools(name) ON DELETE CASCADE,
    allowed BOOLEAN NOT NULL DEFAULT true,
    conditions JSONB DEFAULT '{}',  -- Additional conditions for tool execution
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_agent_tool_permission UNIQUE (agent_id, tool_name)
);

CREATE INDEX IF NOT EXISTS idx_tool_permissions_agent_id ON neurondb_agent.tool_permissions(agent_id);
CREATE INDEX IF NOT EXISTS idx_tool_permissions_tool_name ON neurondb_agent.tool_permissions(tool_name);

-- Session-scoped tool permissions (overrides agent-level permissions)
CREATE TABLE IF NOT EXISTS neurondb_agent.session_tool_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    tool_name TEXT NOT NULL REFERENCES neurondb_agent.tools(name) ON DELETE CASCADE,
    allowed BOOLEAN NOT NULL DEFAULT true,
    conditions JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_session_tool_permission UNIQUE (session_id, tool_name)
);

CREATE INDEX IF NOT EXISTS idx_session_tool_permissions_session_id ON neurondb_agent.session_tool_permissions(session_id);
CREATE INDEX IF NOT EXISTS idx_session_tool_permissions_tool_name ON neurondb_agent.session_tool_permissions(tool_name);

-- Data permissions: schema, table, row filters, column masks
CREATE TABLE IF NOT EXISTS neurondb_agent.data_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    principal_id UUID NOT NULL REFERENCES neurondb_agent.principals(id) ON DELETE CASCADE,
    schema_name TEXT,
    table_name TEXT,
    row_filter TEXT,  -- SQL WHERE clause for row-level filtering
    column_mask JSONB DEFAULT '{}',  -- Map of column names to masking rules
    permissions TEXT[] NOT NULL DEFAULT '{}',  -- e.g., ['select', 'insert', 'update', 'delete']
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_data_permissions CHECK (array_length(permissions, 1) > 0),
    CONSTRAINT valid_data_permission_target CHECK (
        (schema_name IS NOT NULL) OR (table_name IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_data_permissions_principal_id ON neurondb_agent.data_permissions(principal_id);
CREATE INDEX IF NOT EXISTS idx_data_permissions_schema_table ON neurondb_agent.data_permissions(schema_name, table_name);

-- Audit log table: Comprehensive audit logging for tool calls and SQL statements
CREATE TABLE IF NOT EXISTS neurondb_agent.audit_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    principal_id UUID REFERENCES neurondb_agent.principals(id) ON DELETE SET NULL,
    api_key_id UUID REFERENCES neurondb_agent.api_keys(id) ON DELETE SET NULL,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    action TEXT NOT NULL,  -- e.g., 'tool_call', 'sql_execute', 'agent_execute'
    resource_type TEXT NOT NULL,  -- e.g., 'tool', 'sql', 'agent'
    resource_id TEXT,  -- e.g., tool name, SQL query hash, agent ID
    inputs_hash TEXT,  -- SHA-256 hash of inputs
    outputs_hash TEXT,  -- SHA-256 hash of outputs
    inputs JSONB,  -- Optional: actual inputs (may be truncated for privacy)
    outputs JSONB,  -- Optional: actual outputs (may be truncated for privacy)
    metadata JSONB DEFAULT '{}',  -- Additional context
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON neurondb_agent.audit_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_principal_id ON neurondb_agent.audit_log(principal_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_api_key_id ON neurondb_agent.audit_log(api_key_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_agent_id ON neurondb_agent.audit_log(agent_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_session_id ON neurondb_agent.audit_log(session_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON neurondb_agent.audit_log(action, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_resource ON neurondb_agent.audit_log(resource_type, resource_id, timestamp DESC);

-- Partition audit log by month for performance (optional but recommended for high-volume)
-- This will be created via a separate function or migration if needed

