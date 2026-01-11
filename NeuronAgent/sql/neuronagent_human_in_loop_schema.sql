-- Human-in-the-loop schema
CREATE TABLE IF NOT EXISTS neurondb_agent.approval_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    request_type TEXT NOT NULL CHECK (request_type IN ('tool_execution', 'agent_action', 'budget_exceeded', 'sensitive_operation')),
    action_description TEXT NOT NULL,
    payload JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'expired')),
    requested_by TEXT,
    approved_by TEXT,
    rejection_reason TEXT,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_approval_requests_status ON neurondb_agent.approval_requests(status, created_at) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_approval_requests_agent ON neurondb_agent.approval_requests(agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_approval_requests_session ON neurondb_agent.approval_requests(session_id, created_at DESC);

-- User feedback table
CREATE TABLE IF NOT EXISTS neurondb_agent.user_feedback (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    message_id BIGINT REFERENCES neurondb_agent.messages(id) ON DELETE SET NULL,
    user_id TEXT,
    feedback_type TEXT NOT NULL CHECK (feedback_type IN ('positive', 'negative', 'neutral', 'correction')),
    rating INT CHECK (rating >= 1 AND rating <= 5),
    comment TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_feedback_agent ON neurondb_agent.user_feedback(agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_feedback_session ON neurondb_agent.user_feedback(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_feedback_type ON neurondb_agent.user_feedback(feedback_type, created_at);










