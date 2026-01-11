/*-------------------------------------------------------------------------
 *
 * 020_task_alerts.sql
 *    Task alert preferences and alert history
 *
 * Enables users to configure alert preferences for task events and
 * tracks alert delivery history.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronAgent/migrations/020_task_alerts.sql
 *
 *-------------------------------------------------------------------------
 */

/* Task alert preferences table for user notification settings */
CREATE TABLE IF NOT EXISTS neurondb_agent.task_alert_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    alert_types TEXT[] NOT NULL DEFAULT ARRAY['completion', 'failure']::TEXT[],
    channels TEXT[] NOT NULL DEFAULT ARRAY['webhook']::TEXT[],
    email_address TEXT,
    webhook_url TEXT,
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(user_id, agent_id)
);

/* Task alerts history table */
CREATE TABLE IF NOT EXISTS neurondb_agent.task_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL REFERENCES neurondb_agent.async_tasks(id) ON DELETE CASCADE,
    alert_type TEXT NOT NULL CHECK (alert_type IN ('completion', 'failure', 'progress', 'milestone')),
    channel TEXT NOT NULL CHECK (channel IN ('email', 'webhook', 'push')),
    recipient TEXT NOT NULL,
    message TEXT,
    sent_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'delivered', 'failed')),
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

/* Indexes */
CREATE INDEX IF NOT EXISTS idx_task_alert_preferences_user ON neurondb_agent.task_alert_preferences(user_id, enabled) WHERE enabled = true;
CREATE INDEX IF NOT EXISTS idx_task_alert_preferences_agent ON neurondb_agent.task_alert_preferences(agent_id, enabled) WHERE enabled = true;
CREATE INDEX IF NOT EXISTS idx_task_alerts_task ON neurondb_agent.task_alerts(task_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_task_alerts_status ON neurondb_agent.task_alerts(status, created_at) WHERE status IN ('pending', 'failed');
CREATE INDEX IF NOT EXISTS idx_task_alerts_type ON neurondb_agent.task_alerts(alert_type, channel);

/* Comments */
COMMENT ON TABLE neurondb_agent.task_alert_preferences IS 'User preferences for task alerts. Configures which alerts to receive and via which channels.';
COMMENT ON COLUMN neurondb_agent.task_alert_preferences.alert_types IS 'Array of alert types to receive: completion, failure, progress, milestone';
COMMENT ON COLUMN neurondb_agent.task_alert_preferences.channels IS 'Array of delivery channels: email, webhook, push';
COMMENT ON COLUMN neurondb_agent.task_alert_preferences.email_address IS 'Email address for email channel notifications';
COMMENT ON COLUMN neurondb_agent.task_alert_preferences.webhook_url IS 'Webhook URL for webhook channel notifications';

COMMENT ON TABLE neurondb_agent.task_alerts IS 'History of task alerts sent to users. Tracks delivery status and errors.';
COMMENT ON COLUMN neurondb_agent.task_alerts.alert_type IS 'Type of alert: completion, failure, progress, or milestone';
COMMENT ON COLUMN neurondb_agent.task_alerts.channel IS 'Delivery channel used: email, webhook, or push';
COMMENT ON COLUMN neurondb_agent.task_alerts.recipient IS 'Recipient identifier (email, webhook URL, or push token)';
COMMENT ON COLUMN neurondb_agent.task_alerts.status IS 'Delivery status: pending, sent, delivered, or failed';
