-- ============================================================================
-- NeuronAgent Workflow Engine Schema Migration
-- ============================================================================
-- This migration creates tables for DAG workflow engine with steps, inputs, outputs,
-- dependencies, retries, and idempotency keys.
-- ============================================================================

-- Workflows table: DAG workflow definitions
CREATE TABLE IF NOT EXISTS neurondb_agent.workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    dag_definition JSONB NOT NULL,  -- DAG structure: nodes, edges, step definitions
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'paused', 'archived')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_workflows_status ON neurondb_agent.workflows(status);
CREATE INDEX IF NOT EXISTS idx_workflows_created_at ON neurondb_agent.workflows(created_at DESC);

-- Workflow steps table: Individual steps in a workflow
CREATE TABLE IF NOT EXISTS neurondb_agent.workflow_steps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES neurondb_agent.workflows(id) ON DELETE CASCADE,
    step_name TEXT NOT NULL,
    step_type TEXT NOT NULL CHECK (step_type IN ('agent', 'tool', 'approval', 'http', 'sql', 'custom')),
    inputs JSONB DEFAULT '{}',
    outputs JSONB DEFAULT '{}',
    dependencies TEXT[] DEFAULT '{}',  -- Array of step names this step depends on
    retry_config JSONB DEFAULT '{}',  -- {max_retries, backoff_multiplier, initial_delay, max_delay}
    idempotency_key TEXT,  -- Optional idempotency key for this step
    compensation_step_id UUID REFERENCES neurondb_agent.workflow_steps(id) ON DELETE SET NULL,  -- Rollback step
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_workflow_step_name UNIQUE (workflow_id, step_name)
);

CREATE INDEX IF NOT EXISTS idx_workflow_steps_workflow_id ON neurondb_agent.workflow_steps(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_steps_idempotency_key ON neurondb_agent.workflow_steps(idempotency_key) WHERE idempotency_key IS NOT NULL;

-- Workflow executions table: Execution instances of workflows
CREATE TABLE IF NOT EXISTS neurondb_agent.workflow_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES neurondb_agent.workflows(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    trigger_type TEXT NOT NULL CHECK (trigger_type IN ('manual', 'schedule', 'webhook', 'db_notify', 'queue')),
    trigger_data JSONB DEFAULT '{}',
    inputs JSONB DEFAULT '{}',
    outputs JSONB DEFAULT '{}',
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow_id ON neurondb_agent.workflow_executions(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_status ON neurondb_agent.workflow_executions(status);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_created_at ON neurondb_agent.workflow_executions(created_at DESC);

-- Workflow step executions table: Execution instances of individual steps
CREATE TABLE IF NOT EXISTS neurondb_agent.workflow_step_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_execution_id UUID NOT NULL REFERENCES neurondb_agent.workflow_executions(id) ON DELETE CASCADE,
    workflow_step_id UUID NOT NULL REFERENCES neurondb_agent.workflow_steps(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped', 'compensated')),
    inputs JSONB DEFAULT '{}',
    outputs JSONB DEFAULT '{}',
    error_message TEXT,
    retry_count INT DEFAULT 0,
    idempotency_key TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_execution_step_idempotency UNIQUE (idempotency_key) WHERE idempotency_key IS NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_workflow_step_executions_execution_id ON neurondb_agent.workflow_step_executions(workflow_execution_id);
CREATE INDEX IF NOT EXISTS idx_workflow_step_executions_step_id ON neurondb_agent.workflow_step_executions(workflow_step_id);
CREATE INDEX IF NOT EXISTS idx_workflow_step_executions_status ON neurondb_agent.workflow_step_executions(status);
CREATE INDEX IF NOT EXISTS idx_workflow_step_executions_idempotency_key ON neurondb_agent.workflow_step_executions(idempotency_key) WHERE idempotency_key IS NOT NULL;

-- Workflow schedules table: Schedule definitions for workflows
CREATE TABLE IF NOT EXISTS neurondb_agent.workflow_schedules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES neurondb_agent.workflows(id) ON DELETE CASCADE,
    cron_expression TEXT NOT NULL,  -- Cron expression for scheduling
    timezone TEXT DEFAULT 'UTC',
    enabled BOOLEAN DEFAULT true,
    next_run_at TIMESTAMPTZ,
    last_run_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_workflow_schedule UNIQUE (workflow_id)
);

CREATE INDEX IF NOT EXISTS idx_workflow_schedules_workflow_id ON neurondb_agent.workflow_schedules(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_schedules_next_run_at ON neurondb_agent.workflow_schedules(next_run_at) WHERE enabled = true AND next_run_at IS NOT NULL;

