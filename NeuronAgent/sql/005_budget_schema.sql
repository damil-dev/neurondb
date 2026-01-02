-- Budget management schema
CREATE TABLE IF NOT EXISTS neurondb_agent.agent_budgets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    budget_amount REAL NOT NULL CHECK (budget_amount >= 0),
    period_type TEXT NOT NULL CHECK (period_type IN ('daily', 'weekly', 'monthly', 'yearly', 'total')),
    start_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    end_date TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(agent_id, period_type) WHERE is_active = true
);

CREATE INDEX IF NOT EXISTS idx_agent_budgets_agent ON neurondb_agent.agent_budgets(agent_id, is_active);
CREATE INDEX IF NOT EXISTS idx_agent_budgets_active ON neurondb_agent.agent_budgets(agent_id) WHERE is_active = true;










