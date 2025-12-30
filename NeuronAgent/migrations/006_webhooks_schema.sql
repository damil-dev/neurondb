-- Webhooks schema for event notifications
CREATE TABLE IF NOT EXISTS neurondb_agent.webhooks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT NOT NULL,
    events TEXT[] NOT NULL,
    secret TEXT,
    enabled BOOLEAN DEFAULT true,
    timeout_seconds INT DEFAULT 30,
    retry_count INT DEFAULT 3,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_webhooks_enabled ON neurondb_agent.webhooks(enabled) WHERE enabled = true;
CREATE INDEX IF NOT EXISTS idx_webhooks_events ON neurondb_agent.webhooks USING GIN(events);

-- Webhook deliveries for tracking webhook execution
CREATE TABLE IF NOT EXISTS neurondb_agent.webhook_deliveries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    webhook_id UUID NOT NULL REFERENCES neurondb_agent.webhooks(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    payload JSONB NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'success', 'failed', 'retrying')),
    status_code INT,
    response_body TEXT,
    error_message TEXT,
    attempt_count INT DEFAULT 0,
    next_retry_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    delivered_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_webhook ON neurondb_agent.webhook_deliveries(webhook_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_status ON neurondb_agent.webhook_deliveries(status, next_retry_at) WHERE status IN ('pending', 'retrying');









