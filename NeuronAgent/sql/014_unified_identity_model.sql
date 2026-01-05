-- ============================================================================
-- Unified Identity Model Migration for NeuronAgent
-- ============================================================================
-- This migration aligns NeuronAgent with the unified identity model
-- used across NeuronDB ecosystem components.
-- ============================================================================

-- ============================================================================
-- Step 1: Add Organizations Table (if not exists)
-- ============================================================================
CREATE TABLE IF NOT EXISTS neurondb_agent.organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_organizations_slug ON neurondb_agent.organizations(slug);

-- ============================================================================
-- Step 2: Add Service Accounts Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS neurondb_agent.service_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    org_id UUID REFERENCES neurondb_agent.organizations(id) ON DELETE CASCADE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_service_accounts_org_id ON neurondb_agent.service_accounts(org_id);

-- ============================================================================
-- Step 3: Update Principals Table to Include Service Account Type
-- ============================================================================
-- Update the check constraint to include 'service_account'
DO $$
BEGIN
    -- Drop existing constraint if it exists
    ALTER TABLE neurondb_agent.principals 
    DROP CONSTRAINT IF EXISTS principals_type_check;
    
    -- Add new constraint with service_account
    ALTER TABLE neurondb_agent.principals 
    ADD CONSTRAINT principals_type_check 
    CHECK (type IN ('user', 'org', 'agent', 'tool', 'dataset', 'service_account'));
END $$;

-- ============================================================================
-- Step 4: Update API Keys Table to Support Unified Model
-- ============================================================================
-- Add principal_type column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'neurondb_agent'
        AND table_name = 'api_keys' 
        AND column_name = 'principal_type'
    ) THEN
        ALTER TABLE neurondb_agent.api_keys 
        ADD COLUMN principal_type TEXT CHECK (principal_type IN ('user', 'org', 'service_account'));
    END IF;
END $$;

-- Migrate existing data to set principal_type
DO $$
BEGIN
    -- Set principal_type based on principal_id lookup
    UPDATE neurondb_agent.api_keys ak
    SET principal_type = p.type
    FROM neurondb_agent.principals p
    WHERE ak.principal_id = p.id
    AND ak.principal_type IS NULL;
    
    -- For API keys with organization_id or user_id but no principal_id
    -- Create principals if needed
    INSERT INTO neurondb_agent.principals (type, name, created_at)
    SELECT DISTINCT 
        CASE 
            WHEN ak.organization_id IS NOT NULL THEN 'org'
            WHEN ak.user_id IS NOT NULL THEN 'user'
        END,
        COALESCE(ak.organization_id, ak.user_id),
        NOW()
    FROM neurondb_agent.api_keys ak
    WHERE ak.principal_id IS NULL
    AND (ak.organization_id IS NOT NULL OR ak.user_id IS NOT NULL)
    ON CONFLICT (type, name) DO NOTHING;
    
    -- Link API keys to principals
    UPDATE neurondb_agent.api_keys ak
    SET 
        principal_id = p.id,
        principal_type = p.type
    FROM neurondb_agent.principals p
    WHERE ak.principal_id IS NULL
    AND (
        (ak.organization_id IS NOT NULL AND p.type = 'org' AND p.name = ak.organization_id)
        OR (ak.user_id IS NOT NULL AND p.type = 'user' AND p.name = ak.user_id)
    );
END $$;

-- ============================================================================
-- Step 5: Add Project Reference to API Keys
-- ============================================================================
-- Note: NeuronAgent doesn't have a projects table, but we can add a reference
-- for future integration with NeuronDesktop profiles
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'neurondb_agent'
        AND table_name = 'api_keys' 
        AND column_name = 'project_id'
    ) THEN
        ALTER TABLE neurondb_agent.api_keys 
        ADD COLUMN project_id TEXT; -- TEXT for now, can be UUID if we link to Desktop
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_api_keys_project_id ON neurondb_agent.api_keys(project_id) WHERE project_id IS NOT NULL;

-- ============================================================================
-- Step 6: Update Audit Log to Include Project ID
-- ============================================================================
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'neurondb_agent'
        AND table_name = 'audit_log' 
        AND column_name = 'project_id'
    ) THEN
        ALTER TABLE neurondb_agent.audit_log 
        ADD COLUMN project_id TEXT;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_audit_log_project_id ON neurondb_agent.audit_log(project_id) WHERE project_id IS NOT NULL;

-- ============================================================================
-- Step 7: Create Views for Convenience
-- ============================================================================

-- View: API keys with principal information
CREATE OR REPLACE VIEW neurondb_agent.api_keys_with_principals AS
SELECT 
    ak.id,
    ak.key_prefix,
    ak.principal_id,
    ak.principal_type,
    ak.project_id,
    ak.rate_limit_per_minute,
    ak.roles,
    ak.last_used_at,
    ak.expires_at,
    ak.created_at,
    p.name AS principal_name,
    p.metadata AS principal_metadata
FROM neurondb_agent.api_keys ak
LEFT JOIN neurondb_agent.principals p ON ak.principal_id = p.id;

-- View: Unified principals view
CREATE OR REPLACE VIEW neurondb_agent.unified_principals AS
SELECT 
    'user' AS principal_type,
    id AS principal_id,
    name AS principal_name,
    metadata,
    created_at,
    updated_at
FROM neurondb_agent.principals
WHERE type = 'user'
UNION ALL
SELECT 
    'org' AS principal_type,
    id AS principal_id,
    name AS principal_name,
    metadata,
    created_at,
    updated_at
FROM neurondb_agent.principals
WHERE type = 'org'
UNION ALL
SELECT 
    'service_account' AS principal_type,
    sa.id AS principal_id,
    sa.name AS principal_name,
    sa.metadata,
    sa.created_at,
    sa.updated_at
FROM neurondb_agent.service_accounts sa;

-- ============================================================================
-- Step 8: Create Helper Functions
-- ============================================================================

-- Function: Get or create principal for user
CREATE OR REPLACE FUNCTION neurondb_agent.get_or_create_user_principal(user_id TEXT)
RETURNS UUID AS $$
DECLARE
    principal_id UUID;
BEGIN
    -- Try to find existing principal
    SELECT id INTO principal_id
    FROM neurondb_agent.principals
    WHERE type = 'user' AND name = user_id;
    
    -- If not found, create it
    IF principal_id IS NULL THEN
        INSERT INTO neurondb_agent.principals (type, name, metadata, created_at)
        VALUES ('user', user_id, jsonb_build_object('user_id', user_id), NOW())
        RETURNING id INTO principal_id;
    END IF;
    
    RETURN principal_id;
END;
$$ LANGUAGE plpgsql;

-- Function: Get principal for API key
CREATE OR REPLACE FUNCTION neurondb_agent.get_principal_for_api_key(api_key_prefix TEXT)
RETURNS TABLE (
    principal_id UUID,
    principal_type TEXT,
    principal_name TEXT,
    project_id TEXT,
    roles TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ak.principal_id,
        ak.principal_type,
        p.name,
        ak.project_id,
        ak.roles
    FROM neurondb_agent.api_keys ak
    LEFT JOIN neurondb_agent.principals p ON ak.principal_id = p.id
    WHERE ak.key_prefix = api_key_prefix;
END;
$$ LANGUAGE plpgsql;

-- Function: Check if principal has permission
CREATE OR REPLACE FUNCTION neurondb_agent.check_principal_permission(
    p_principal_id UUID,
    p_resource_type TEXT,
    p_resource_id TEXT,
    p_permission TEXT
)
RETURNS BOOLEAN AS $$
DECLARE
    has_permission BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1
        FROM neurondb_agent.policies
        WHERE principal_id = p_principal_id
        AND resource_type = p_resource_type
        AND (resource_id = p_resource_id OR resource_id IS NULL)
        AND p_permission = ANY(permissions)
    ) INTO has_permission;
    
    RETURN COALESCE(has_permission, FALSE);
END;
$$ LANGUAGE plpgsql;

