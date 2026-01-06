-- Migration: Add agent templates and workflows tables
-- Created: 2024-2025
-- Description: Store agent templates and workflow definitions

-- Agent Templates Table
CREATE TABLE IF NOT EXISTS agent_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    category TEXT,
    configuration JSONB NOT NULL DEFAULT '{}',
    workflow JSONB,
    popularity INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_templates_category ON agent_templates(category);
CREATE INDEX IF NOT EXISTS idx_agent_templates_popularity ON agent_templates(popularity DESC);

-- Agent Workflows Table (for storing workflow definitions)
CREATE TABLE IF NOT EXISTS agent_workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID, -- References NeuronAgent agent (external, not FK)
    name TEXT NOT NULL,
    workflow_definition JSONB NOT NULL,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_workflows_agent_id ON agent_workflows(agent_id);

-- User Agent Templates (user-created templates from agents)
CREATE TABLE IF NOT EXISTS user_agent_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID, -- Would reference users table if exists
    agent_id TEXT NOT NULL, -- External agent ID from NeuronAgent
    template_id UUID REFERENCES agent_templates(id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    custom_config JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_agent_templates_user_id ON user_agent_templates(user_id);
CREATE INDEX IF NOT EXISTS idx_user_agent_templates_template_id ON user_agent_templates(template_id);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update updated_at
CREATE TRIGGER update_agent_templates_updated_at
    BEFORE UPDATE ON agent_templates
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_workflows_updated_at
    BEFORE UPDATE ON agent_workflows
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();



