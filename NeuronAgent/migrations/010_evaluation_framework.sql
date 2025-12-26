-- ============================================================================
-- NeuronAgent Evaluation Framework Schema Migration
-- ============================================================================
-- This migration creates tables for evaluation framework with golden tasks,
-- expected tool sequences, and SQL side effects comparison.
-- ============================================================================

-- Eval tasks table: Golden tasks for evaluation
CREATE TABLE IF NOT EXISTS neurondb_agent.eval_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_type TEXT NOT NULL,  -- e.g., 'tool_sequence', 'sql_side_effect', 'retrieval', 'end_to_end'
    input TEXT NOT NULL,  -- Input for the task
    expected_output TEXT,  -- Expected output
    expected_tool_sequence JSONB,  -- Expected sequence of tool calls
    golden_sql_side_effects JSONB,  -- Expected SQL side effects (table states)
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_tasks_task_type ON neurondb_agent.eval_tasks(task_type);
CREATE INDEX IF NOT EXISTS idx_eval_tasks_created_at ON neurondb_agent.eval_tasks(created_at DESC);

-- Eval runs table: Evaluation run metadata
CREATE TABLE IF NOT EXISTS neurondb_agent.eval_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_version TEXT NOT NULL,  -- Version identifier for the dataset
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    score REAL,  -- Overall score (0-1)
    total_tasks INT DEFAULT 0,
    passed_tasks INT DEFAULT 0,
    failed_tasks INT DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_dataset_version ON neurondb_agent.eval_runs(dataset_version);
CREATE INDEX IF NOT EXISTS idx_eval_runs_agent_id ON neurondb_agent.eval_runs(agent_id);
CREATE INDEX IF NOT EXISTS idx_eval_runs_started_at ON neurondb_agent.eval_runs(started_at DESC);

-- Eval task results table: Results for individual tasks in a run
CREATE TABLE IF NOT EXISTS neurondb_agent.eval_task_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_run_id UUID NOT NULL REFERENCES neurondb_agent.eval_runs(id) ON DELETE CASCADE,
    eval_task_id UUID NOT NULL REFERENCES neurondb_agent.eval_tasks(id) ON DELETE CASCADE,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    passed BOOLEAN NOT NULL DEFAULT false,
    actual_output TEXT,
    actual_tool_sequence JSONB,
    actual_sql_side_effects JSONB,
    score REAL,  -- Task-specific score (0-1)
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_task_results_eval_run_id ON neurondb_agent.eval_task_results(eval_run_id);
CREATE INDEX IF NOT EXISTS idx_eval_task_results_eval_task_id ON neurondb_agent.eval_task_results(eval_task_id);
CREATE INDEX IF NOT EXISTS idx_eval_task_results_passed ON neurondb_agent.eval_task_results(passed);

-- Retrieval eval results table: Specialized results for retrieval evaluation
CREATE TABLE IF NOT EXISTS neurondb_agent.eval_retrieval_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_task_result_id UUID NOT NULL REFERENCES neurondb_agent.eval_task_results(id) ON DELETE CASCADE,
    recall_at_k REAL,  -- Recall@k score
    mrr REAL,  -- Mean Reciprocal Rank
    grounding_passed BOOLEAN,  -- Whether retrieved chunks were properly cited
    retrieved_chunks JSONB,  -- Retrieved chunks
    relevant_chunks JSONB,  -- Ground truth relevant chunks
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_retrieval_results_task_result_id ON neurondb_agent.eval_retrieval_results(eval_task_result_id);

