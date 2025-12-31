/*-------------------------------------------------------------------------
 *
 * 018_async_tasks.sql
 *    Asynchronous task execution with status tracking and notifications
 *
 * Enables agents to execute long-running tasks asynchronously with
 * status tracking, result storage, and completion notifications.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronAgent/migrations/018_async_tasks.sql
 *
 *-------------------------------------------------------------------------
 */

/* Async tasks table for tracking long-running agent tasks */
CREATE TABLE IF NOT EXISTS neurondb_agent.async_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    task_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    priority INT NOT NULL DEFAULT 0,
    input JSONB NOT NULL DEFAULT '{}',
    result JSONB,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);

/* Indexes for efficient querying */
CREATE INDEX IF NOT EXISTS idx_async_tasks_status ON neurondb_agent.async_tasks(status, priority DESC, created_at);
CREATE INDEX IF NOT EXISTS idx_async_tasks_session ON neurondb_agent.async_tasks(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_async_tasks_agent ON neurondb_agent.async_tasks(agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_async_tasks_type ON neurondb_agent.async_tasks(task_type, status);

/* Task notifications table for tracking notification delivery */
CREATE TABLE IF NOT EXISTS neurondb_agent.task_notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL REFERENCES neurondb_agent.async_tasks(id) ON DELETE CASCADE,
    notification_type TEXT NOT NULL CHECK (notification_type IN ('completion', 'failure', 'progress', 'milestone')),
    channel TEXT NOT NULL CHECK (channel IN ('email', 'webhook', 'push')),
    recipient TEXT NOT NULL,
    sent_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'failed', 'delivered')),
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

/* Indexes for notification queries */
CREATE INDEX IF NOT EXISTS idx_task_notifications_task ON neurondb_agent.task_notifications(task_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_task_notifications_status ON neurondb_agent.task_notifications(status, created_at) WHERE status IN ('pending', 'failed');
CREATE INDEX IF NOT EXISTS idx_task_notifications_type ON neurondb_agent.task_notifications(notification_type, channel);

/* Comments */
COMMENT ON TABLE neurondb_agent.async_tasks IS 'Tracks asynchronous agent tasks with status, results, and metadata. Enables long-running tasks to execute in background with status tracking.';
COMMENT ON COLUMN neurondb_agent.async_tasks.task_type IS 'Type of task (e.g., "agent_execution", "data_processing", "code_execution")';
COMMENT ON COLUMN neurondb_agent.async_tasks.status IS 'Current status: pending, running, completed, failed, or cancelled';
COMMENT ON COLUMN neurondb_agent.async_tasks.priority IS 'Task priority (higher numbers = higher priority)';
COMMENT ON COLUMN neurondb_agent.async_tasks.input IS 'Task input parameters as JSON';
COMMENT ON COLUMN neurondb_agent.async_tasks.result IS 'Task result/output as JSON (null until completion)';
COMMENT ON COLUMN neurondb_agent.async_tasks.error_message IS 'Error message if task failed';

COMMENT ON TABLE neurondb_agent.task_notifications IS 'Tracks notifications sent for task events (completion, failure, progress, milestones)';
COMMENT ON COLUMN neurondb_agent.task_notifications.notification_type IS 'Type of notification: completion, failure, progress, or milestone';
COMMENT ON COLUMN neurondb_agent.task_notifications.channel IS 'Delivery channel: email, webhook, or push';
COMMENT ON COLUMN neurondb_agent.task_notifications.recipient IS 'Recipient identifier (email address, webhook URL, or push token)';
