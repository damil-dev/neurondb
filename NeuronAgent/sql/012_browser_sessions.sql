-- Browser sessions schema for web browser tool
-- Stores browser session state for persistent browsing across requests

CREATE TABLE IF NOT EXISTS neurondb_agent.browser_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL UNIQUE,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    current_url TEXT,
    cookies JSONB DEFAULT '[]',
    local_storage JSONB DEFAULT '{}',
    session_storage JSONB DEFAULT '{}',
    user_agent TEXT,
    viewport_width INT DEFAULT 1920,
    viewport_height INT DEFAULT 1080,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_browser_sessions_session_id ON neurondb_agent.browser_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_browser_sessions_agent_id ON neurondb_agent.browser_sessions(agent_id);
CREATE INDEX IF NOT EXISTS idx_browser_sessions_expires_at ON neurondb_agent.browser_sessions(expires_at) WHERE expires_at IS NOT NULL;

-- Browser snapshots for screenshot storage
CREATE TABLE IF NOT EXISTS neurondb_agent.browser_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    browser_session_id UUID REFERENCES neurondb_agent.browser_sessions(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    screenshot_data BYTEA,
    screenshot_b64 TEXT,
    page_title TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_browser_snapshots_session_id ON neurondb_agent.browser_snapshots(session_id);
CREATE INDEX IF NOT EXISTS idx_browser_snapshots_browser_session_id ON neurondb_agent.browser_snapshots(browser_session_id);
CREATE INDEX IF NOT EXISTS idx_browser_snapshots_created_at ON neurondb_agent.browser_snapshots(created_at DESC);

COMMENT ON TABLE neurondb_agent.browser_sessions IS 'Stores browser session state for web browser tool automation';
COMMENT ON TABLE neurondb_agent.browser_snapshots IS 'Stores browser screenshots and page snapshots';
