-- ============================================================================
-- NeuronAgent Execution Snapshots Schema Migration
-- ============================================================================
-- This migration creates the execution snapshots table for replay functionality.
-- Prerequisites: Migration 008_principals_and_permissions.sql must be run first
-- ============================================================================

-- Execution snapshots table: Store complete execution state for replay
CREATE TABLE IF NOT EXISTS neurondb_agent.execution_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    user_message TEXT NOT NULL,
    execution_state JSONB NOT NULL,  -- Complete execution state (inputs, tool calls, outputs, LLM responses)
    deterministic_mode BOOLEAN DEFAULT false,  -- Whether execution was in deterministic mode
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_execution_snapshots_session_id ON neurondb_agent.execution_snapshots(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_execution_snapshots_agent_id ON neurondb_agent.execution_snapshots(agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_execution_snapshots_deterministic ON neurondb_agent.execution_snapshots(deterministic_mode, created_at DESC);

