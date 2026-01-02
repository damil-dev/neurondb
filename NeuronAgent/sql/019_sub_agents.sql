/*-------------------------------------------------------------------------
 *
 * 019_sub_agents.sql
 *    Specialized sub-agents for task routing
 *
 * Enables agents to have specializations (planning, research, coding, execution)
 * for automatic task routing and coordination.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronAgent/migrations/019_sub_agents.sql
 *
 *-------------------------------------------------------------------------
 */

/* Agent specializations table */
CREATE TABLE IF NOT EXISTS neurondb_agent.agent_specializations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL UNIQUE REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    specialization_type TEXT NOT NULL CHECK (specialization_type IN ('planning', 'research', 'coding', 'execution', 'analysis', 'general')),
    capabilities TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    config JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

/* Indexes */
CREATE INDEX IF NOT EXISTS idx_agent_specializations_type ON neurondb_agent.agent_specializations(specialization_type);
CREATE INDEX IF NOT EXISTS idx_agent_specializations_agent ON neurondb_agent.agent_specializations(agent_id);

/* Comments */
COMMENT ON TABLE neurondb_agent.agent_specializations IS 'Defines agent specializations for automatic task routing. Agents can specialize in planning, research, coding, execution, analysis, or general tasks.';
COMMENT ON COLUMN neurondb_agent.agent_specializations.specialization_type IS 'Type of specialization: planning, research, coding, execution, analysis, or general';
COMMENT ON COLUMN neurondb_agent.agent_specializations.capabilities IS 'Array of specific capabilities this agent has (e.g., ["python", "sql", "web_scraping"])';
COMMENT ON COLUMN neurondb_agent.agent_specializations.config IS 'Specialization-specific configuration as JSON';
