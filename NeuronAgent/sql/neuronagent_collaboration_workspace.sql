-- Real-Time Collaboration Workspace Schema
-- Enables multiple users and agents to collaborate on shared tasks
-- Provides workspace management, participant tracking, and real-time updates

-- Collaboration workspaces
CREATE TABLE IF NOT EXISTS neurondb_agent.collaboration_workspaces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    owner_id UUID,
    description TEXT,
    shared_context JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Workspace participants
CREATE TABLE IF NOT EXISTS neurondb_agent.workspace_participants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES neurondb_agent.collaboration_workspaces(id) ON DELETE CASCADE,
    user_id UUID,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    role TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('owner', 'admin', 'member', 'viewer')),
    joined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(workspace_id, COALESCE(user_id::text, ''), COALESCE(agent_id::text, ''))
);

-- Workspace updates for real-time synchronization
CREATE TABLE IF NOT EXISTS neurondb_agent.workspace_updates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES neurondb_agent.collaboration_workspaces(id) ON DELETE CASCADE,
    user_id UUID,
    agent_id UUID REFERENCES neurondb_agent.agents(id) ON DELETE SET NULL,
    update_type TEXT NOT NULL CHECK (update_type IN ('message', 'action', 'state_change', 'file_update', 'context_sync')),
    content TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Workspace sessions linking sessions to workspaces
CREATE TABLE IF NOT EXISTS neurondb_agent.workspace_sessions (
    workspace_id UUID NOT NULL REFERENCES neurondb_agent.collaboration_workspaces(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (workspace_id, session_id)
);

-- Indexes for collaboration_workspaces
CREATE INDEX IF NOT EXISTS idx_collaboration_workspaces_owner_id ON neurondb_agent.collaboration_workspaces(owner_id);
CREATE INDEX IF NOT EXISTS idx_collaboration_workspaces_created_at ON neurondb_agent.collaboration_workspaces(created_at DESC);

-- Indexes for workspace_participants
CREATE INDEX IF NOT EXISTS idx_workspace_participants_workspace_id ON neurondb_agent.workspace_participants(workspace_id);
CREATE INDEX IF NOT EXISTS idx_workspace_participants_user_id ON neurondb_agent.workspace_participants(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_workspace_participants_agent_id ON neurondb_agent.workspace_participants(agent_id) WHERE agent_id IS NOT NULL;

-- Indexes for workspace_updates
CREATE INDEX IF NOT EXISTS idx_workspace_updates_workspace_id ON neurondb_agent.workspace_updates(workspace_id);
CREATE INDEX IF NOT EXISTS idx_workspace_updates_created_at ON neurondb_agent.workspace_updates(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_workspace_updates_update_type ON neurondb_agent.workspace_updates(update_type);

-- Indexes for workspace_sessions
CREATE INDEX IF NOT EXISTS idx_workspace_sessions_workspace_id ON neurondb_agent.workspace_sessions(workspace_id);
CREATE INDEX IF NOT EXISTS idx_workspace_sessions_session_id ON neurondb_agent.workspace_sessions(session_id);

COMMENT ON TABLE neurondb_agent.collaboration_workspaces IS 'Shared workspaces for collaborative agent tasks. Multiple users and agents can work together on shared tasks.';
COMMENT ON TABLE neurondb_agent.workspace_participants IS 'Participants in collaboration workspaces. Tracks users and agents with their roles.';
COMMENT ON TABLE neurondb_agent.workspace_updates IS 'Real-time updates broadcast to all workspace participants. Used for live synchronization.';
COMMENT ON TABLE neurondb_agent.workspace_sessions IS 'Links agent sessions to workspaces for shared context and collaboration.';

COMMENT ON COLUMN neurondb_agent.workspace_participants.role IS 'Participant role: owner (full control), admin (manage participants), member (edit), viewer (read-only).';
COMMENT ON COLUMN neurondb_agent.workspace_updates.update_type IS 'Type of update: message (chat), action (agent action), state_change (workspace state), file_update (file change), context_sync (context update).';
