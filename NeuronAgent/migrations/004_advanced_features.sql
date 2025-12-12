-- Advanced features migration
-- Adds tables for multi-agent collaboration, cost tracking, quality scoring, versioning, plans, and reflections

-- Agent relationships table for multi-agent collaboration
CREATE TABLE IF NOT EXISTS neurondb_agent.agent_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    to_agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL CHECK (relationship_type IN ('delegates_to', 'collaborates_with', 'supervises', 'reports_to')),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT different_agents CHECK (from_agent_id != to_agent_id)
);

CREATE INDEX IF NOT EXISTS idx_agent_relationships_from ON neurondb_agent.agent_relationships(from_agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_relationships_to ON neurondb_agent.agent_relationships(to_agent_id);

-- Tool usage logs for analytics
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

CREATE INDEX IF NOT EXISTS idx_tool_usage_agent ON neurondb_agent.tool_usage_logs(agent_id, created_at);
CREATE INDEX IF NOT EXISTS idx_tool_usage_tool ON neurondb_agent.tool_usage_logs(tool_name, created_at);
CREATE INDEX IF NOT EXISTS idx_tool_usage_session ON neurondb_agent.tool_usage_logs(session_id, created_at);

-- Cost logs for cost tracking
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

CREATE INDEX IF NOT EXISTS idx_cost_logs_agent ON neurondb_agent.cost_logs(agent_id, created_at);
CREATE INDEX IF NOT EXISTS idx_cost_logs_session ON neurondb_agent.cost_logs(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_cost_logs_type ON neurondb_agent.cost_logs(cost_type, created_at);

-- Quality scores for response quality tracking
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
    user_feedback INTEGER CHECK (user_feedback >= -1 AND user_feedback <= 1), -- -1: negative, 0: neutral, 1: positive
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_quality_scores_agent ON neurondb_agent.quality_scores(agent_id, created_at);
CREATE INDEX IF NOT EXISTS idx_quality_scores_session ON neurondb_agent.quality_scores(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_quality_scores_overall ON neurondb_agent.quality_scores(overall_score);

-- Agent versions for versioning support
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

CREATE INDEX IF NOT EXISTS idx_agent_versions_agent ON neurondb_agent.agent_versions(agent_id, version_number DESC);
CREATE INDEX IF NOT EXISTS idx_agent_versions_active ON neurondb_agent.agent_versions(agent_id, is_active) WHERE is_active = true;

-- Plans for stored plans
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

CREATE INDEX IF NOT EXISTS idx_plans_agent ON neurondb_agent.plans(agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_plans_session ON neurondb_agent.plans(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_plans_status ON neurondb_agent.plans(status, created_at);

-- Reflections for reflection logs
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

CREATE INDEX IF NOT EXISTS idx_reflections_agent ON neurondb_agent.reflections(agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reflections_session ON neurondb_agent.reflections(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reflections_quality ON neurondb_agent.reflections(quality_score);

-- Add version column to agents table
ALTER TABLE neurondb_agent.agents ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1;
ALTER TABLE neurondb_agent.agents ADD COLUMN IF NOT EXISTS parent_agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_agents_version ON neurondb_agent.agents(version);
CREATE INDEX IF NOT EXISTS idx_agents_parent ON neurondb_agent.agents(parent_agent_id);


