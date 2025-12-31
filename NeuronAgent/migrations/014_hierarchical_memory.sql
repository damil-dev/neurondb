-- Hierarchical Memory System
-- Implements Short-Term, Mid-Term, and Long-Term Personal Memory tiers
-- Provides automatic memory promotion and expiration

-- Short-Term Memory (STM) - conversation-level, 1 hour TTL
CREATE TABLE IF NOT EXISTS neurondb_agent.memory_stm (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding neurondb_vector(768),
    importance_score FLOAT NOT NULL DEFAULT 0.5,
    access_count INT NOT NULL DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '1 hour')
);

-- Mid-Term Memory (MTM) - topic summaries, 7 days TTL
CREATE TABLE IF NOT EXISTS neurondb_agent.memory_mtm (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    topic TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding neurondb_vector(768),
    importance_score FLOAT NOT NULL DEFAULT 0.6,
    source_stm_ids UUID[],
    pattern_count INT NOT NULL DEFAULT 1,
    last_reinforced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '7 days')
);

-- Long-Term Personal Memory (LPM) - preferences, permanent
CREATE TABLE IF NOT EXISTS neurondb_agent.memory_lpm (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    user_id UUID,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding neurondb_vector(768),
    importance_score FLOAT NOT NULL DEFAULT 0.8,
    source_mtm_ids UUID[],
    confidence FLOAT NOT NULL DEFAULT 0.7,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Memory transitions tracking
CREATE TABLE IF NOT EXISTS neurondb_agent.memory_transitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    from_tier TEXT NOT NULL CHECK (from_tier IN ('stm', 'mtm')),
    to_tier TEXT NOT NULL CHECK (to_tier IN ('mtm', 'lpm')),
    source_id UUID NOT NULL,
    target_id UUID NOT NULL,
    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for STM
CREATE INDEX IF NOT EXISTS idx_memory_stm_agent_id ON neurondb_agent.memory_stm(agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_stm_session_id ON neurondb_agent.memory_stm(session_id);
CREATE INDEX IF NOT EXISTS idx_memory_stm_expires_at ON neurondb_agent.memory_stm(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memory_stm_importance ON neurondb_agent.memory_stm(importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_memory_stm_embedding ON neurondb_agent.memory_stm USING hnsw (embedding neurondb_vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Indexes for MTM
CREATE INDEX IF NOT EXISTS idx_memory_mtm_agent_id ON neurondb_agent.memory_mtm(agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_mtm_topic ON neurondb_agent.memory_mtm(topic);
CREATE INDEX IF NOT EXISTS idx_memory_mtm_expires_at ON neurondb_agent.memory_mtm(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memory_mtm_importance ON neurondb_agent.memory_mtm(importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_memory_mtm_embedding ON neurondb_agent.memory_mtm USING hnsw (embedding neurondb_vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Indexes for LPM
CREATE INDEX IF NOT EXISTS idx_memory_lpm_agent_id ON neurondb_agent.memory_lpm(agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_lpm_user_id ON neurondb_agent.memory_lpm(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memory_lpm_category ON neurondb_agent.memory_lpm(category);
CREATE INDEX IF NOT EXISTS idx_memory_lpm_importance ON neurondb_agent.memory_lpm(importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_memory_lpm_embedding ON neurondb_agent.memory_lpm USING hnsw (embedding neurondb_vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Indexes for transitions
CREATE INDEX IF NOT EXISTS idx_memory_transitions_agent_id ON neurondb_agent.memory_transitions(agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_transitions_source_id ON neurondb_agent.memory_transitions(source_id);
CREATE INDEX IF NOT EXISTS idx_memory_transitions_created_at ON neurondb_agent.memory_transitions(created_at DESC);

COMMENT ON TABLE neurondb_agent.memory_stm IS 'Short-Term Memory: Real-time conversation data with 1-hour TTL. Automatically expires and promotes to MTM based on importance and patterns.';
COMMENT ON TABLE neurondb_agent.memory_mtm IS 'Mid-Term Memory: Topic summaries and recurring patterns with 7-day TTL. Promoted from STM when patterns detected.';
COMMENT ON TABLE neurondb_agent.memory_lpm IS 'Long-Term Personal Memory: User preferences and agent knowledge. Permanent storage for high-importance information.';
COMMENT ON TABLE neurondb_agent.memory_transitions IS 'Tracks memory promotions between tiers for analytics and debugging.';

COMMENT ON COLUMN neurondb_agent.memory_stm.expires_at IS 'Automatic expiration timestamp. STM expires after 1 hour unless accessed or promoted.';
COMMENT ON COLUMN neurondb_agent.memory_mtm.pattern_count IS 'Number of times this pattern has been observed. Higher count increases promotion likelihood.';
COMMENT ON COLUMN neurondb_agent.memory_lpm.confidence IS 'Confidence score for this memory (0-1). Higher confidence indicates more reliable information.';
