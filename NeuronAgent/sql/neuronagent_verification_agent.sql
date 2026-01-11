-- Verification Agent Schema
-- Provides quality assurance and output validation for agent executions
-- Supports automated verification rules and verification queue processing

-- Verification queue for pending verifications
CREATE TABLE IF NOT EXISTS neurondb_agent.verification_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    output_id UUID,
    output_content TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    priority TEXT NOT NULL DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

-- Verification results
CREATE TABLE IF NOT EXISTS neurondb_agent.verification_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    queue_id UUID NOT NULL REFERENCES neurondb_agent.verification_queue(id) ON DELETE CASCADE,
    verifier_agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    passed BOOLEAN NOT NULL,
    issues JSONB DEFAULT '[]',
    suggestions JSONB DEFAULT '[]',
    confidence FLOAT NOT NULL DEFAULT 0.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    verified_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Verification rules
CREATE TABLE IF NOT EXISTS neurondb_agent.verification_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    rule_type TEXT NOT NULL CHECK (rule_type IN ('output_format', 'data_accuracy', 'logical_consistency', 'completeness')),
    criteria JSONB NOT NULL DEFAULT '{}',
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for verification_queue
CREATE INDEX IF NOT EXISTS idx_verification_queue_status ON neurondb_agent.verification_queue(status);
CREATE INDEX IF NOT EXISTS idx_verification_queue_priority ON neurondb_agent.verification_queue(priority DESC);
CREATE INDEX IF NOT EXISTS idx_verification_queue_created_at ON neurondb_agent.verification_queue(created_at ASC);
CREATE INDEX IF NOT EXISTS idx_verification_queue_session_id ON neurondb_agent.verification_queue(session_id);

-- Indexes for verification_results
CREATE INDEX IF NOT EXISTS idx_verification_results_queue_id ON neurondb_agent.verification_results(queue_id);
CREATE INDEX IF NOT EXISTS idx_verification_results_verifier_agent_id ON neurondb_agent.verification_results(verifier_agent_id);
CREATE INDEX IF NOT EXISTS idx_verification_results_passed ON neurondb_agent.verification_results(passed);
CREATE INDEX IF NOT EXISTS idx_verification_results_verified_at ON neurondb_agent.verification_results(verified_at DESC);

-- Indexes for verification_rules
CREATE INDEX IF NOT EXISTS idx_verification_rules_agent_id ON neurondb_agent.verification_rules(agent_id);
CREATE INDEX IF NOT EXISTS idx_verification_rules_rule_type ON neurondb_agent.verification_rules(rule_type);
CREATE INDEX IF NOT EXISTS idx_verification_rules_enabled ON neurondb_agent.verification_rules(enabled) WHERE enabled = true;

COMMENT ON TABLE neurondb_agent.verification_queue IS 'Queue of outputs pending verification. Processes outputs through quality assurance checks.';
COMMENT ON TABLE neurondb_agent.verification_results IS 'Results of verification checks including pass/fail status, issues found, and improvement suggestions.';
COMMENT ON TABLE neurondb_agent.verification_rules IS 'Verification rules defining quality criteria for agent outputs. Rules can be enabled or disabled per agent.';

COMMENT ON COLUMN neurondb_agent.verification_queue.priority IS 'Verification priority: low (background), medium (normal), high (immediate).';
COMMENT ON COLUMN neurondb_agent.verification_results.confidence IS 'Confidence score (0-1) indicating reliability of verification result.';
COMMENT ON COLUMN neurondb_agent.verification_rules.criteria IS 'JSONB object defining rule criteria. Format varies by rule_type.';
