-- ============================================================================
-- NeuronDesktop Complete Setup Script
-- ============================================================================
-- This script sets up everything needed for NeuronDesktop:
-- - Database schema (tables, indexes)
-- - Default profile with NeuronDB and MCP configuration
-- - Permissions
-- - Model configurations table
-- ============================================================================

\echo '======================================================================'
\echo 'NeuronDesktop Complete Setup'
\echo '======================================================================'
\echo ''

-- ============================================================================
-- Step 1: Create Extensions
-- ============================================================================
\echo 'Step 1: Creating extensions...'
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
\echo '✓ Extensions created'
\echo ''

-- ============================================================================
-- Step 2: Create Profiles Table
-- ============================================================================
\echo 'Step 2: Creating profiles table...'
CREATE TABLE IF NOT EXISTS profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    user_id TEXT NOT NULL,
    mcp_config JSONB,
    neurondb_dsn TEXT NOT NULL,
    agent_endpoint TEXT,
    agent_api_key TEXT,
    default_collection TEXT,
    is_default BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_profiles_user_id ON profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_profiles_default ON profiles(is_default) WHERE is_default = true;

\echo '✓ Profiles table created'
\echo ''

-- ============================================================================
-- Step 3: Create API Keys Table
-- ============================================================================
\echo 'Step 3: Creating API keys table...'
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
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

\echo '✓ API keys table created'
\echo ''

-- ============================================================================
-- Step 4: Create Request Logs Table
-- ============================================================================
\echo 'Step 4: Creating request logs table...'
CREATE TABLE IF NOT EXISTS request_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
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

\echo '✓ Request logs table created'
\echo ''

-- ============================================================================
-- Step 5: Create Model Configs Table
-- ============================================================================
\echo 'Step 5: Creating model configs table...'
CREATE TABLE IF NOT EXISTS model_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    model_provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    api_key TEXT,
    base_url TEXT,
    is_default BOOLEAN NOT NULL DEFAULT false,
    is_free BOOLEAN NOT NULL DEFAULT false,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(profile_id, model_provider, model_name)
);

CREATE INDEX IF NOT EXISTS idx_model_configs_profile_id ON model_configs(profile_id);
CREATE INDEX IF NOT EXISTS idx_model_configs_default ON model_configs(profile_id, is_default) WHERE is_default = true;

\echo '✓ Model configs table created'
\echo ''

-- ============================================================================
-- Step 6: Clean Up Existing Default Profiles
-- ============================================================================
\echo 'Step 6: Cleaning up existing profiles...'
-- Remove any existing default flags
UPDATE profiles SET is_default = false WHERE is_default = true;
\echo '✓ Existing profiles cleaned'
\echo ''

-- ============================================================================
-- Step 7: Create Default Profile
-- ============================================================================
\echo 'Step 7: Creating default profile...'

-- Delete any existing "Default" profile to start fresh
DELETE FROM profiles WHERE name = 'Default' AND user_id = 'nbduser';

-- Insert default profile with complete configuration
INSERT INTO profiles (
    id,
    name,
    user_id,
    mcp_config,
    neurondb_dsn,
    is_default,
    created_at,
    updated_at
) VALUES (
    uuid_generate_v4(),
    'Default',
    'nbduser',
    '{
        "command": "/Users/pgedge/pge/neurondb/NeuronMCP/bin/neurondb-mcp",
        "args": [],
        "env": {
            "NEURONDB_HOST": "localhost",
            "NEURONDB_PORT": "5432",
            "NEURONDB_DATABASE": "neurondb",
            "NEURONDB_USER": "nbduser"
        }
    }'::jsonb,
    'postgresql://nbduser@localhost:5432/neurondb',
    true,
    NOW(),
    NOW()
)
RETURNING id, name, user_id, is_default;

\echo '✓ Default profile created'
\echo ''

-- ============================================================================
-- Step 8: Grant Permissions
-- ============================================================================
\echo 'Step 8: Setting up permissions...'

-- Grant all privileges to current user (neurondesk)
DO $$
BEGIN
    -- Grant table privileges
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO CURRENT_USER;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO CURRENT_USER;
    
    -- Set default privileges for future objects
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO CURRENT_USER;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO CURRENT_USER;
    
    RAISE NOTICE 'Permissions granted';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Permission setup completed (some may already exist)';
END $$;

\echo '✓ Permissions configured'
\echo ''

-- ============================================================================
-- Step 9: Verify Setup
-- ============================================================================
\echo 'Step 9: Verifying setup...'

\echo 'Profiles:'
SELECT 
    id,
    name,
    user_id,
    neurondb_dsn,
    is_default,
    mcp_config->>'command' as mcp_command,
    mcp_config->'env'->>'NEURONDB_USER' as db_user,
    mcp_config->'env'->>'NEURONDB_DATABASE' as db_name
FROM profiles;

\echo ''
\echo 'Tables created:'
SELECT 
    schemaname,
    tablename,
    tableowner
FROM pg_tables
WHERE schemaname = 'public'
    AND tablename IN ('profiles', 'api_keys', 'request_logs', 'model_configs')
ORDER BY tablename;

\echo ''
\echo 'Indexes created:'
SELECT 
    indexname,
    tablename
FROM pg_indexes
WHERE schemaname = 'public'
    AND tablename IN ('profiles', 'api_keys', 'request_logs', 'model_configs')
ORDER BY tablename, indexname;

\echo ''
\echo '======================================================================'
\echo 'Setup Complete!'
\echo '======================================================================'
\echo ''
\echo 'Default Profile Configuration:'
\echo '  - Name: Default'
\echo '  - User: nbduser'
\echo '  - Database: neurondb (localhost:5432)'
\echo '  - MCP Server: /Users/pgedge/pge/neurondb/NeuronMCP/bin/neurondb-mcp'
\echo ''
\echo 'Next Steps:'
\echo '  1. Ensure NeuronDB database exists: createdb neurondb'
\echo '  2. Ensure NeuronDB extension is installed in neurondb database'
\echo '  3. Ensure nbduser can connect to neurondb database'
\echo '  4. Start NeuronDesktop backend'
\echo '  5. Access NeuronDesktop at http://localhost:3000'
\echo ''



