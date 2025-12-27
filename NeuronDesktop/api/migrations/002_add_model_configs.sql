-- Add model configurations table
-- Ensure uuid-ossp extension is enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Add model configurations table (only if profiles table exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'profiles') THEN
        -- Create model_configs table if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'model_configs') THEN
            CREATE TABLE model_configs (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                profile_id UUID REFERENCES profiles(id) ON DELETE CASCADE,
                model_provider TEXT NOT NULL, -- 'openai', 'anthropic', 'google', 'ollama', 'custom'
                model_name TEXT NOT NULL, -- 'gpt-4', 'claude-3-opus', 'gemini-pro', 'llama2', etc.
                api_key TEXT, -- Encrypted or hashed API key
                base_url TEXT, -- For custom providers or Ollama
                is_default BOOLEAN DEFAULT false,
                is_free BOOLEAN DEFAULT false, -- For free models like Ollama
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(profile_id, model_provider, model_name)
            );
        END IF;
        
        -- Create indexes (only if they don't exist)
        IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_model_configs_profile_id') THEN
            CREATE INDEX idx_model_configs_profile_id ON model_configs(profile_id);
        END IF;
        
        IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_model_configs_default') THEN
            CREATE INDEX idx_model_configs_default ON model_configs(profile_id, is_default) WHERE is_default = true;
        END IF;
    END IF;
END $$;

-- Add default_profile flag to profiles (only if profiles table exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'profiles') THEN
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'profiles' AND column_name = 'is_default') THEN
            ALTER TABLE profiles ADD COLUMN is_default BOOLEAN DEFAULT false;
        END IF;
        
        IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_profiles_default') THEN
            CREATE INDEX idx_profiles_default ON profiles(is_default) WHERE is_default = true;
        END IF;
    END IF;
END $$;

