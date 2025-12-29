-- Migration: Add MCP Chat Threads and Messages tables
-- This stores chat history for MCP conversations per profile

CREATE TABLE IF NOT EXISTS mcp_chat_threads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT 'New chat',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_mcp_threads_profile_id ON mcp_chat_threads(profile_id);
CREATE INDEX IF NOT EXISTS idx_mcp_threads_updated_at ON mcp_chat_threads(updated_at DESC);

CREATE TABLE IF NOT EXISTS mcp_chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id UUID NOT NULL REFERENCES mcp_chat_threads(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    tool_name TEXT,
    data JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_mcp_messages_thread_id ON mcp_chat_messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_mcp_messages_created_at ON mcp_chat_messages(created_at);

