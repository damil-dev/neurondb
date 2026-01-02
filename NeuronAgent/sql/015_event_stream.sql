-- Event Stream Architecture
-- Provides chronological logging of all agent actions, user messages, and system events
-- Enables context management, event summarization, and audit trails

-- Event stream table for chronological event logging
CREATE TABLE IF NOT EXISTS neurondb_agent.event_stream (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL CHECK (event_type IN ('user_message', 'agent_action', 'tool_execution', 'agent_response', 'error', 'system')),
    actor TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_event_stream_session_id ON neurondb_agent.event_stream(session_id);
CREATE INDEX IF NOT EXISTS idx_event_stream_created_at ON neurondb_agent.event_stream(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_event_stream_event_type ON neurondb_agent.event_stream(event_type);
CREATE INDEX IF NOT EXISTS idx_event_stream_session_time ON neurondb_agent.event_stream(session_id, created_at DESC);

-- Event summaries for compressed event history
CREATE TABLE IF NOT EXISTS neurondb_agent.event_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    start_event_id UUID NOT NULL,
    end_event_id UUID NOT NULL,
    event_count INT NOT NULL,
    summary_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_event_summaries_session_id ON neurondb_agent.event_summaries(session_id);
CREATE INDEX IF NOT EXISTS idx_event_summaries_created_at ON neurondb_agent.event_summaries(created_at DESC);

COMMENT ON TABLE neurondb_agent.event_stream IS 'Chronological log of all user messages, agent actions, tool executions, and system events. Enables event sourcing, context management, and audit trails.';
COMMENT ON TABLE neurondb_agent.event_summaries IS 'Compressed summaries of event ranges for efficient context window management. Created when event count exceeds threshold.';

COMMENT ON COLUMN neurondb_agent.event_stream.event_type IS 'Type of event: user_message (user input), agent_action (agent decision), tool_execution (tool call), agent_response (agent output), error (error event), system (system event)';
COMMENT ON COLUMN neurondb_agent.event_stream.actor IS 'Entity that triggered the event: user ID, agent ID, tool name, or system';
COMMENT ON COLUMN neurondb_agent.event_stream.content IS 'Event content: message text, action description, tool result, error message';
COMMENT ON COLUMN neurondb_agent.event_stream.metadata IS 'Additional event metadata: tool parameters, error details, timing information';
