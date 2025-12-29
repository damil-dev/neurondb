-- ============================================================================
-- NeuronDesktop Complete Schema
-- ============================================================================
-- This file contains the complete schema for NeuronDesktop.
-- Run this file on your PostgreSQL database to set up NeuronDesktop.
-- 
-- Usage:
--   psql -h localhost -p 5432 -U your_user -d your_database -f neurondesktop.sql
-- 
-- Or from psql:
--   \i neurondesktop.sql
-- ============================================================================

-- Ensure uuid-ossp extension is enabled (if not available, PostgreSQL 13+ has gen_random_uuid() built-in)
-- Try to create extension, but continue if it fails (will use gen_random_uuid() instead)
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
EXCEPTION WHEN OTHERS THEN
    -- Extension not available, will use gen_random_uuid() instead
    NULL;
END $$;

-- Ensure pgcrypto is enabled for gen_random_uuid()
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS "pgcrypto";
EXCEPTION WHEN OTHERS THEN
    -- Extension not available; database might already provide gen_random_uuid()
    NULL;
END $$;

-- ============================================================================
-- Users Table
-- ============================================================================
-- Stores user accounts for authentication
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Ensure new columns exist even if table was created by an older schema
ALTER TABLE users
    ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_is_admin ON users(is_admin);

-- ============================================================================
-- Profiles Table
-- ============================================================================
-- Stores connection profiles for NeuronDB, MCP, and Agent
CREATE TABLE IF NOT EXISTS profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    user_id TEXT NOT NULL,
    profile_username TEXT,
    profile_password_hash TEXT,
    mcp_config JSONB,
    neurondb_dsn TEXT NOT NULL,
    agent_endpoint TEXT,
    agent_api_key TEXT,
    default_collection TEXT,
    is_default BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Ensure new columns exist even if table was created by an older schema
ALTER TABLE profiles
    ADD COLUMN IF NOT EXISTS profile_username TEXT,
    ADD COLUMN IF NOT EXISTS profile_password_hash TEXT;

CREATE INDEX IF NOT EXISTS idx_profiles_user_id ON profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_profiles_default ON profiles(is_default) WHERE is_default = true;
CREATE UNIQUE INDEX IF NOT EXISTS idx_profiles_profile_username_unique ON profiles(profile_username) WHERE profile_username IS NOT NULL;

-- ============================================================================
-- API Keys Table
-- ============================================================================
-- Stores API keys for authentication (legacy support)
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash TEXT NOT NULL,
    key_prefix TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    profile_id UUID REFERENCES profiles(id) ON DELETE SET NULL,
    rate_limit INTEGER NOT NULL DEFAULT 100,
    last_used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(key_prefix);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);

-- ============================================================================
-- Request Logs Table
-- ============================================================================
-- Stores API request logs for monitoring and debugging
CREATE TABLE IF NOT EXISTS request_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id UUID REFERENCES profiles(id) ON DELETE SET NULL,
    endpoint TEXT NOT NULL,
    method TEXT NOT NULL,
    request_body JSONB,
    response_body JSONB,
    status_code INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_request_logs_profile_id ON request_logs(profile_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_created_at ON request_logs(created_at DESC);

-- ============================================================================
-- Model Configurations Table
-- ============================================================================
-- Stores AI model configurations for each profile
CREATE TABLE IF NOT EXISTS model_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id UUID REFERENCES profiles(id) ON DELETE CASCADE,
    model_provider TEXT NOT NULL, -- 'openai', 'anthropic', 'google', 'ollama', 'custom'
    model_name TEXT NOT NULL, -- 'gpt-4', 'claude-3-opus', 'gemini-pro', 'llama2', etc.
    api_key TEXT, -- API key for the model
    base_url TEXT, -- For custom providers or Ollama
    is_default BOOLEAN DEFAULT false,
    is_free BOOLEAN DEFAULT false, -- For free models like Ollama
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(profile_id, model_provider, model_name)
);

CREATE INDEX IF NOT EXISTS idx_model_configs_profile_id ON model_configs(profile_id);
CREATE INDEX IF NOT EXISTS idx_model_configs_default ON model_configs(profile_id, is_default) WHERE is_default = true;

-- ============================================================================
-- App Settings Table
-- ============================================================================
-- Stores application-level configuration
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_app_settings_updated_at ON app_settings(updated_at);

-- ============================================================================
-- Initial Data
-- ============================================================================

-- Insert default admin user (password: neurondb)
-- Note: This uses bcrypt hash. The password hash below is for "neurondb"
-- In production, you should generate a new hash or use the application to create users
DO $$
DECLARE
    admin_exists BOOLEAN;
    admin_user_id UUID;
    admin_profile_exists BOOLEAN;
    admin_profile_id UUID;
BEGIN
    -- Check if admin user already exists
    SELECT EXISTS(SELECT 1 FROM users WHERE username = 'admin') INTO admin_exists;
    
    IF NOT admin_exists THEN
        -- Insert admin user with bcrypt hash for password "neurondb"
        -- This hash was generated with bcrypt.DefaultCost
        -- To generate a new hash, use: bcrypt.GenerateFromPassword([]byte("neurondb"), bcrypt.DefaultCost)
        INSERT INTO users (username, password_hash, is_admin) VALUES (
            'admin',
            '$2a$10$dHqYwh2ZFZZFpITA35/LoOLK697lBFdTJYsMqlRHpKp.XhGvEdS5a', -- bcrypt hash for "neurondb"
            TRUE
        );
        RAISE NOTICE 'Admin user created with username: admin, password: neurondb';
    ELSE
        RAISE NOTICE 'Admin user already exists';
    END IF;

    -- Ensure admin user has is_admin=true
    UPDATE users SET is_admin = TRUE WHERE username = 'admin';

    -- Get admin user id
    SELECT id INTO admin_user_id FROM users WHERE username = 'admin' LIMIT 1;

    -- Create default admin profile (login by admin/neurondb lands on this profile)
    SELECT EXISTS(SELECT 1 FROM profiles WHERE profile_username = 'admin') INTO admin_profile_exists;
    IF NOT admin_profile_exists THEN
        INSERT INTO profiles (
            name,
            user_id,
            profile_username,
            profile_password_hash,
            neurondb_dsn,
            is_default
        ) VALUES (
            'admin',
            admin_user_id::text,
            'admin',
            '$2a$10$dHqYwh2ZFZZFpITA35/LoOLK697lBFdTJYsMqlRHpKp.XhGvEdS5a', -- bcrypt hash for "neurondb"
            format('postgresql://%s@localhost:5432/%s', current_user, current_database()),
            TRUE
        ) RETURNING id INTO admin_profile_id;
        RAISE NOTICE 'Admin profile created (profile_username=admin)';
    ELSE
        SELECT id INTO admin_profile_id FROM profiles WHERE profile_username = 'admin' LIMIT 1;
        RAISE NOTICE 'Admin profile already exists';
    END IF;

    -- Seed default model configs for admin profile (idempotent)
    IF admin_profile_id IS NOT NULL THEN
        INSERT INTO model_configs (profile_id, model_provider, model_name, api_key, base_url, is_default, is_free)
        VALUES
            (admin_profile_id, 'openai', 'gpt-4o', NULL, NULL, TRUE, FALSE),
            (admin_profile_id, 'openai', 'gpt-4-turbo', NULL, NULL, FALSE, FALSE),
            (admin_profile_id, 'openai', 'gpt-4', NULL, NULL, FALSE, FALSE),
            (admin_profile_id, 'openai', 'gpt-3.5-turbo', NULL, NULL, FALSE, FALSE),
            (admin_profile_id, 'anthropic', 'claude-3-5-sonnet-20241022', NULL, NULL, FALSE, FALSE),
            (admin_profile_id, 'anthropic', 'claude-3-opus-20240229', NULL, NULL, FALSE, FALSE),
            (admin_profile_id, 'anthropic', 'claude-3-sonnet-20240229', NULL, NULL, FALSE, FALSE),
            (admin_profile_id, 'anthropic', 'claude-3-haiku-20240307', NULL, NULL, FALSE, FALSE),
            (admin_profile_id, 'google', 'gemini-1.5-pro', NULL, NULL, FALSE, FALSE),
            (admin_profile_id, 'google', 'gemini-pro', NULL, NULL, FALSE, FALSE),
            (admin_profile_id, 'ollama', 'llama2', NULL, 'http://localhost:11434', FALSE, TRUE)
        ON CONFLICT (profile_id, model_provider, model_name) DO NOTHING;
    END IF;
END $$;

-- ============================================================================
-- Schema Verification
-- ============================================================================
-- Create a function to verify schema completeness
CREATE OR REPLACE FUNCTION neurondesktop_verify_schema() RETURNS TABLE(
    tbl_name TEXT,
    tbl_exists BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 'users'::TEXT, EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'users')
    UNION ALL
    SELECT 'profiles'::TEXT, EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'profiles')
    UNION ALL
    SELECT 'api_keys'::TEXT, EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'api_keys')
    UNION ALL
    SELECT 'request_logs'::TEXT, EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'request_logs')
    UNION ALL
    SELECT 'model_configs'::TEXT, EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'model_configs')
    UNION ALL
    SELECT 'app_settings'::TEXT, EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'app_settings');
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Completion Message
-- ============================================================================
DO $$
BEGIN
    RAISE NOTICE '============================================================================';
    RAISE NOTICE 'NeuronDesktop schema has been successfully created!';
    RAISE NOTICE '============================================================================';
    RAISE NOTICE 'Default admin user:';
    RAISE NOTICE '  Username: admin';
    RAISE NOTICE '  Password: neurondb';
    RAISE NOTICE '============================================================================';
    RAISE NOTICE 'To verify the schema, run: SELECT * FROM neurondesktop_verify_schema();';
    RAISE NOTICE '============================================================================';
END $$;

