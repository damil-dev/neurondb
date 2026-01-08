/*-------------------------------------------------------------------------
 *
 * setup_neurondb_mcp_schema.sql
 *    Comprehensive NeuronMCP Configuration Schema
 *
 * Creates all tables, indexes, views, triggers, and pre-populates
 * default configurations for NeuronMCP to work seamlessly with NeuronDB.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 *-------------------------------------------------------------------------
 */

-- Ensure required extensions are available
-- Note: The neurondb extension will create the neurondb schema automatically
-- Do NOT create the schema manually here, as it must be owned by the extension
CREATE EXTENSION IF NOT EXISTS neurondb;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================================
-- PART 1: LLM MODELS & PROVIDERS (5 tables)
-- ============================================================================

-- 1. LLM Providers Table
CREATE TABLE IF NOT EXISTS neurondb.llm_providers (
    provider_id SERIAL PRIMARY KEY,
    provider_name TEXT NOT NULL UNIQUE,  -- 'openai', 'anthropic', 'huggingface', 'local', 'openai-compatible'
    display_name TEXT NOT NULL,
    default_base_url TEXT,
    auth_method TEXT NOT NULL DEFAULT 'api_key' CHECK (auth_method IN ('api_key', 'bearer', 'oauth', 'none')),
    default_timeout_ms INTEGER DEFAULT 30000,
    rate_limit_rpm INTEGER,  -- Requests per minute
    rate_limit_tpm INTEGER, -- Tokens per minute
    supports_streaming BOOLEAN DEFAULT false,
    supports_embeddings BOOLEAN DEFAULT false,
    supports_chat BOOLEAN DEFAULT false,
    supports_completion BOOLEAN DEFAULT false,
    metadata JSONB DEFAULT '{}',
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'deprecated', 'disabled')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE neurondb.llm_providers IS 'Master table for LLM providers (OpenAI, Anthropic, HuggingFace, etc.)';

-- 2. LLM Models Table
CREATE TABLE IF NOT EXISTS neurondb.llm_models (
    model_id SERIAL PRIMARY KEY,
    provider_id INTEGER NOT NULL REFERENCES neurondb.llm_providers(provider_id) ON DELETE RESTRICT,
    model_name TEXT NOT NULL,  -- e.g., 'text-embedding-3-small', 'gpt-4'
    model_alias TEXT,  -- Short alias for convenience
    model_type TEXT NOT NULL CHECK (model_type IN ('embedding', 'chat', 'completion', 'rerank', 'multimodal')),
    context_window INTEGER,  -- Max tokens/context length
    embedding_dimension INTEGER,  -- For embedding models
    max_output_tokens INTEGER,
    supports_streaming BOOLEAN DEFAULT false,
    supports_function_calling BOOLEAN DEFAULT false,
    cost_per_1k_tokens_input NUMERIC(10,6),
    cost_per_1k_tokens_output NUMERIC(10,6),
    description TEXT,
    documentation_url TEXT,
    status TEXT DEFAULT 'available' CHECK (status IN ('available', 'disabled', 'deprecated', 'beta')),
    is_default BOOLEAN DEFAULT false,  -- Default model for this type/provider
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    UNIQUE(provider_id, model_name)
);
COMMENT ON TABLE neurondb.llm_models IS 'Catalog of all available LLM models';

-- 3. LLM Model Keys Table (Secure Storage)
CREATE TABLE IF NOT EXISTS neurondb.llm_model_keys (
    key_id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL UNIQUE REFERENCES neurondb.llm_models(model_id) ON DELETE CASCADE,
    api_key_encrypted BYTEA NOT NULL,  -- Encrypted using pgcrypto
    encryption_salt BYTEA NOT NULL,
    key_type TEXT DEFAULT 'api_key' CHECK (key_type IN ('api_key', 'bearer_token', 'oauth_token')),
    expires_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    access_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT DEFAULT CURRENT_USER
);
COMMENT ON TABLE neurondb.llm_model_keys IS 'Secure storage for encrypted API keys';

-- 4. LLM Model Configurations Table
CREATE TABLE IF NOT EXISTS neurondb.llm_model_configs (
    config_id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES neurondb.llm_models(model_id) ON DELETE CASCADE,
    config_name TEXT DEFAULT 'default',
    base_url TEXT,  -- Override provider default
    endpoint_path TEXT,  -- API endpoint path
    default_params JSONB DEFAULT '{}',  -- temperature, top_p, etc.
    request_headers JSONB DEFAULT '{}',  -- Custom headers
    timeout_ms INTEGER,
    retry_config JSONB DEFAULT '{"max_retries": 3, "backoff_ms": 1000}',
    rate_limit_config JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(model_id, config_name)
);
COMMENT ON TABLE neurondb.llm_model_configs IS 'Model-specific configurations';

-- 5. LLM Model Usage Tracking Table
CREATE TABLE IF NOT EXISTS neurondb.llm_model_usage (
    usage_id BIGSERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES neurondb.llm_models(model_id) ON DELETE SET NULL,
    operation_type TEXT NOT NULL CHECK (operation_type IN ('embedding', 'chat', 'completion', 'rerank')),
    tokens_input INTEGER,
    tokens_output INTEGER,
    cost NUMERIC(10,6),
    latency_ms INTEGER,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    user_context TEXT,  -- For multi-tenant scenarios
    created_at TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE neurondb.llm_model_usage IS 'Usage tracking and analytics for LLM models';

-- ============================================================================
-- PART 2: VECTOR INDEX CONFIGURATIONS (2 tables)
-- ============================================================================

-- 6. Index Configurations Table
CREATE TABLE IF NOT EXISTS neurondb.index_configs (
    config_id SERIAL PRIMARY KEY,
    table_name TEXT,
    vector_column TEXT,
    index_type TEXT NOT NULL CHECK (index_type IN ('hnsw', 'ivf', 'flat')),
    hnsw_m INTEGER DEFAULT 16,  -- HNSW: number of connections
    hnsw_ef_construction INTEGER DEFAULT 200,  -- HNSW: construction parameter
    hnsw_ef_search INTEGER DEFAULT 64,  -- HNSW: search parameter
    ivf_lists INTEGER DEFAULT 100,  -- IVF: number of lists
    ivf_probes INTEGER DEFAULT 10,  -- IVF: number of probes
    distance_metric TEXT DEFAULT 'l2' CHECK (distance_metric IN ('l2', 'cosine', 'inner_product')),
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(table_name, vector_column) WHERE table_name IS NOT NULL AND vector_column IS NOT NULL
);
COMMENT ON TABLE neurondb.index_configs IS 'Default index configurations for vector columns';

-- 7. Index Templates Table
CREATE TABLE IF NOT EXISTS neurondb.index_templates (
    template_id SERIAL PRIMARY KEY,
    template_name TEXT NOT NULL UNIQUE,
    description TEXT,
    index_type TEXT NOT NULL CHECK (index_type IN ('hnsw', 'ivf', 'flat')),
    config_json JSONB NOT NULL,  -- Full index configuration
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE neurondb.index_templates IS 'Reusable index templates for common configurations';

-- ============================================================================
-- PART 3: WORKER CONFIGURATIONS (2 tables)
-- ============================================================================

-- 8. Worker Configurations Table
CREATE TABLE IF NOT EXISTS neurondb.worker_configs (
    config_id SERIAL PRIMARY KEY,
    worker_name TEXT NOT NULL UNIQUE,  -- 'neuranq', 'neuranmon', 'neurandefrag'
    display_name TEXT NOT NULL,
    enabled BOOLEAN DEFAULT true,
    naptime_ms INTEGER DEFAULT 1000,  -- Sleep time between iterations
    config_json JSONB DEFAULT '{}',  -- Worker-specific configuration
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE neurondb.worker_configs IS 'Background worker settings and configurations';

-- 9. Worker Schedules Table
CREATE TABLE IF NOT EXISTS neurondb.worker_schedules (
    schedule_id SERIAL PRIMARY KEY,
    worker_name TEXT NOT NULL REFERENCES neurondb.worker_configs(worker_name) ON DELETE CASCADE,
    schedule_name TEXT NOT NULL,
    schedule_type TEXT NOT NULL CHECK (schedule_type IN ('interval', 'cron', 'maintenance_window')),
    schedule_config JSONB NOT NULL,  -- Schedule-specific configuration
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(worker_name, schedule_name)
);
COMMENT ON TABLE neurondb.worker_schedules IS 'Worker scheduling and maintenance windows';

-- ============================================================================
-- PART 4: ML MODEL DEFAULTS (2 tables)
-- ============================================================================

-- 10. ML Default Configurations Table
CREATE TABLE IF NOT EXISTS neurondb.ml_default_configs (
    config_id SERIAL PRIMARY KEY,
    algorithm TEXT NOT NULL UNIQUE,  -- 'linear_regression', 'kmeans', 'svm', etc.
    default_hyperparameters JSONB DEFAULT '{}',
    default_training_settings JSONB DEFAULT '{}',
    use_gpu BOOLEAN DEFAULT false,
    gpu_device INTEGER DEFAULT 0,
    batch_size INTEGER,
    max_iterations INTEGER,
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE neurondb.ml_default_configs IS 'Default ML training configurations per algorithm';

-- 11. ML Model Templates Table
CREATE TABLE IF NOT EXISTS neurondb.ml_model_templates (
    template_id SERIAL PRIMARY KEY,
    template_name TEXT NOT NULL UNIQUE,
    description TEXT,
    algorithm TEXT NOT NULL,
    template_config JSONB NOT NULL,  -- Complete template configuration
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE neurondb.ml_model_templates IS 'Pre-configured ML model templates for quick start';

-- ============================================================================
-- PART 5: TOOL CONFIGURATIONS (1 table)
-- ============================================================================

-- 12. Tool Configurations Table
CREATE TABLE IF NOT EXISTS neurondb.tool_configs (
    config_id SERIAL PRIMARY KEY,
    tool_name TEXT NOT NULL UNIQUE,  -- 'vector_search', 'generate_embedding', 'rag', etc.
    display_name TEXT NOT NULL,
    default_params JSONB DEFAULT '{}',  -- Tool-specific default parameters
    default_limit INTEGER,  -- For search/query tools
    default_timeout_ms INTEGER DEFAULT 30000,
    enabled BOOLEAN DEFAULT true,
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE neurondb.tool_configs IS 'NeuronMCP tool-specific default settings';

-- ============================================================================
-- PART 6: SYSTEM CONFIGURATION (1 table)
-- ============================================================================

-- 13. System Configuration Table
CREATE TABLE IF NOT EXISTS neurondb.system_configs (
    config_id SERIAL PRIMARY KEY,
    config_key TEXT NOT NULL UNIQUE,
    config_value JSONB NOT NULL,
    description TEXT,
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE neurondb.system_configs IS 'System-wide NeuronMCP settings and feature flags';

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- LLM Provider indexes
CREATE INDEX IF NOT EXISTS idx_llm_providers_name ON neurondb.llm_providers(provider_name);
CREATE INDEX IF NOT EXISTS idx_llm_providers_status ON neurondb.llm_providers(status);

-- LLM Model indexes
CREATE INDEX IF NOT EXISTS idx_llm_models_provider ON neurondb.llm_models(provider_id);
CREATE INDEX IF NOT EXISTS idx_llm_models_name ON neurondb.llm_models(model_name);
CREATE INDEX IF NOT EXISTS idx_llm_models_type ON neurondb.llm_models(model_type);
CREATE INDEX IF NOT EXISTS idx_llm_models_status ON neurondb.llm_models(status);
CREATE INDEX IF NOT EXISTS idx_llm_models_default ON neurondb.llm_models(model_type, is_default) WHERE is_default = true;

-- LLM Key indexes
CREATE INDEX IF NOT EXISTS idx_llm_model_keys_model ON neurondb.llm_model_keys(model_id);
CREATE INDEX IF NOT EXISTS idx_llm_model_keys_last_used ON neurondb.llm_model_keys(last_used_at);

-- LLM Config indexes
CREATE INDEX IF NOT EXISTS idx_llm_model_configs_model ON neurondb.llm_model_configs(model_id);
CREATE INDEX IF NOT EXISTS idx_llm_model_configs_active ON neurondb.llm_model_configs(model_id, is_active) WHERE is_active = true;

-- LLM Usage indexes
CREATE INDEX IF NOT EXISTS idx_llm_model_usage_model ON neurondb.llm_model_usage(model_id, created_at);
CREATE INDEX IF NOT EXISTS idx_llm_model_usage_created ON neurondb.llm_model_usage(created_at);
CREATE INDEX IF NOT EXISTS idx_llm_model_usage_type ON neurondb.llm_model_usage(operation_type, created_at);

-- Index Config indexes
CREATE INDEX IF NOT EXISTS idx_index_configs_table ON neurondb.index_configs(table_name, vector_column) WHERE table_name IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_index_templates_name ON neurondb.index_templates(template_name);

-- Worker indexes
CREATE INDEX IF NOT EXISTS idx_worker_configs_name ON neurondb.worker_configs(worker_name);
CREATE INDEX IF NOT EXISTS idx_worker_schedules_worker ON neurondb.worker_schedules(worker_name);

-- ML indexes
CREATE INDEX IF NOT EXISTS idx_ml_default_configs_algorithm ON neurondb.ml_default_configs(algorithm);
CREATE INDEX IF NOT EXISTS idx_ml_model_templates_name ON neurondb.ml_model_templates(template_name);

-- Tool indexes
CREATE INDEX IF NOT EXISTS idx_tool_configs_name ON neurondb.tool_configs(tool_name);

-- System indexes
CREATE INDEX IF NOT EXISTS idx_system_configs_key ON neurondb.system_configs(config_key);

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC TIMESTAMP UPDATES
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION neurondb.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers to all tables with updated_at
CREATE TRIGGER trigger_llm_providers_updated_at
    BEFORE UPDATE ON neurondb.llm_providers
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_llm_models_updated_at
    BEFORE UPDATE ON neurondb.llm_models
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_llm_model_keys_updated_at
    BEFORE UPDATE ON neurondb.llm_model_keys
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_llm_model_configs_updated_at
    BEFORE UPDATE ON neurondb.llm_model_configs
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_index_configs_updated_at
    BEFORE UPDATE ON neurondb.index_configs
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_index_templates_updated_at
    BEFORE UPDATE ON neurondb.index_templates
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_worker_configs_updated_at
    BEFORE UPDATE ON neurondb.worker_configs
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_worker_schedules_updated_at
    BEFORE UPDATE ON neurondb.worker_schedules
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_ml_default_configs_updated_at
    BEFORE UPDATE ON neurondb.ml_default_configs
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_ml_model_templates_updated_at
    BEFORE UPDATE ON neurondb.ml_model_templates
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_tool_configs_updated_at
    BEFORE UPDATE ON neurondb.tool_configs
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_system_configs_updated_at
    BEFORE UPDATE ON neurondb.system_configs
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

-- ============================================================================
-- CONVENIENCE VIEWS
-- ============================================================================

-- Active LLM models view with provider info
CREATE OR REPLACE VIEW neurondb.v_llm_models_active AS
SELECT 
    m.model_id,
    m.model_name,
    m.model_alias,
    p.provider_name,
    p.display_name AS provider_display_name,
    m.model_type,
    m.context_window,
    m.embedding_dimension,
    m.status,
    m.is_default,
    CASE WHEN mk.key_id IS NOT NULL THEN true ELSE false END AS has_api_key,
    mc.config_name,
    mc.base_url,
    mc.default_params,
    m.created_at,
    m.last_used_at
FROM neurondb.llm_models m
JOIN neurondb.llm_providers p ON m.provider_id = p.provider_id
LEFT JOIN neurondb.llm_model_keys mk ON m.model_id = mk.model_id
LEFT JOIN neurondb.llm_model_configs mc ON m.model_id = mc.model_id AND mc.is_active = true
WHERE m.status = 'available' AND p.status = 'active';

COMMENT ON VIEW neurondb.v_llm_models_active IS 'Active LLM models with provider information and key status';

-- Models ready for use (have keys and config)
CREATE OR REPLACE VIEW neurondb.v_llm_models_ready AS
SELECT * FROM neurondb.v_llm_models_active
WHERE has_api_key = true;

COMMENT ON VIEW neurondb.v_llm_models_ready IS 'LLM models ready for use (have API keys configured)';

-- ============================================================================
-- PRE-POPULATE DEFAULT DATA
-- ============================================================================

-- Insert LLM Providers
INSERT INTO neurondb.llm_providers (provider_name, display_name, default_base_url, auth_method, supports_embeddings, supports_chat, supports_completion, supports_streaming)
VALUES
    ('openai', 'OpenAI', 'https://api.openai.com/v1', 'api_key', true, true, true, true),
    ('anthropic', 'Anthropic', 'https://api.anthropic.com', 'api_key', false, true, true, true),
    ('huggingface', 'HuggingFace', 'https://api-inference.huggingface.co', 'api_key', true, false, false, false),
    ('local', 'Local Models', NULL, 'none', true, false, false, false),
    ('openai-compatible', 'OpenAI-Compatible', NULL, 'api_key', true, true, true, true)
ON CONFLICT (provider_name) DO NOTHING;

-- Insert LLM Models (50+ models)
-- Get provider IDs
DO $$
DECLARE
    v_openai_id INTEGER;
    v_anthropic_id INTEGER;
    v_huggingface_id INTEGER;
    v_local_id INTEGER;
BEGIN
    SELECT provider_id INTO v_openai_id FROM neurondb.llm_providers WHERE provider_name = 'openai';
    SELECT provider_id INTO v_anthropic_id FROM neurondb.llm_providers WHERE provider_name = 'anthropic';
    SELECT provider_id INTO v_huggingface_id FROM neurondb.llm_providers WHERE provider_name = 'huggingface';
    SELECT provider_id INTO v_local_id FROM neurondb.llm_providers WHERE provider_name = 'local';

    -- OpenAI Embedding Models
    INSERT INTO neurondb.llm_models (provider_id, model_name, model_type, embedding_dimension, is_default, description)
    VALUES
        (v_openai_id, 'text-embedding-ada-002', 'embedding', 1536, true, 'OpenAI Ada embedding model'),
        (v_openai_id, 'text-embedding-3-small', 'embedding', 1536, false, 'OpenAI small embedding model'),
        (v_openai_id, 'text-embedding-3-large', 'embedding', 3072, false, 'OpenAI large embedding model'),
        (v_openai_id, 'text-embedding-3-small-512', 'embedding', 512, false, 'OpenAI small embedding model (512 dim)'),
        (v_openai_id, 'text-embedding-3-large-256', 'embedding', 256, false, 'OpenAI large embedding model (256 dim)'),
        (v_openai_id, 'text-embedding-3-large-1024', 'embedding', 1024, false, 'OpenAI large embedding model (1024 dim)')
    ON CONFLICT (provider_id, model_name) DO NOTHING;

    -- OpenAI Chat Models
    INSERT INTO neurondb.llm_models (provider_id, model_name, model_type, context_window, supports_streaming, supports_function_calling, is_default, description)
    VALUES
        (v_openai_id, 'gpt-4', 'chat', 8192, true, true, false, 'GPT-4 model'),
        (v_openai_id, 'gpt-4-turbo', 'chat', 128000, true, true, true, 'GPT-4 Turbo model'),
        (v_openai_id, 'gpt-4-turbo-preview', 'chat', 128000, true, true, false, 'GPT-4 Turbo preview'),
        (v_openai_id, 'gpt-3.5-turbo', 'chat', 16385, true, true, false, 'GPT-3.5 Turbo model'),
        (v_openai_id, 'gpt-3.5-turbo-16k', 'chat', 16385, true, true, false, 'GPT-3.5 Turbo 16k context'),
        (v_openai_id, 'gpt-4o', 'chat', 128000, true, true, false, 'GPT-4o model'),
        (v_openai_id, 'gpt-4o-mini', 'chat', 128000, true, true, false, 'GPT-4o mini model'),
        (v_openai_id, 'gpt-4-32k', 'chat', 32768, true, true, false, 'GPT-4 with 32k context')
    ON CONFLICT (provider_id, model_name) DO NOTHING;

    -- Anthropic Models
    INSERT INTO neurondb.llm_models (provider_id, model_name, model_type, context_window, supports_streaming, is_default, description)
    VALUES
        (v_anthropic_id, 'claude-3-opus', 'chat', 200000, true, false, 'Claude 3 Opus model'),
        (v_anthropic_id, 'claude-3-sonnet', 'chat', 200000, true, false, 'Claude 3 Sonnet model'),
        (v_anthropic_id, 'claude-3-haiku', 'chat', 200000, true, false, 'Claude 3 Haiku model'),
        (v_anthropic_id, 'claude-3.5-sonnet', 'chat', 200000, true, true, 'Claude 3.5 Sonnet model'),
        (v_anthropic_id, 'claude-3.5-haiku', 'chat', 200000, true, false, 'Claude 3.5 Haiku model'),
        (v_anthropic_id, 'claude-3-opus-20240229', 'chat', 200000, true, false, 'Claude 3 Opus (versioned)')
    ON CONFLICT (provider_id, model_name) DO NOTHING;

    -- HuggingFace Models
    INSERT INTO neurondb.llm_models (provider_id, model_name, model_type, embedding_dimension, description)
    VALUES
        (v_huggingface_id, 'sentence-transformers/all-MiniLM-L6-v2', 'embedding', 384, 'MiniLM L6 v2 embedding model'),
        (v_huggingface_id, 'sentence-transformers/all-mpnet-base-v2', 'embedding', 768, 'MPNet base v2 embedding model'),
        (v_huggingface_id, 'sentence-transformers/all-MiniLM-L12-v2', 'embedding', 384, 'MiniLM L12 v2 embedding model'),
        (v_huggingface_id, 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'embedding', 384, 'Multilingual MiniLM L12 v2'),
        (v_huggingface_id, 'sentence-transformers/distiluse-base-multilingual-cased', 'embedding', 512, 'DistilUSE multilingual model'),
        (v_huggingface_id, 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 'embedding', 384, 'Multi-QA MiniLM model'),
        (v_huggingface_id, 'sentence-transformers/all-distilroberta-v1', 'embedding', 768, 'DistilRoBERTa model'),
        (v_huggingface_id, 'sentence-transformers/paraphrase-albert-small-v2', 'embedding', 768, 'Paraphrase ALBERT model'),
        (v_huggingface_id, 'sentence-transformers/nli-mpnet-base-v2', 'embedding', 768, 'NLI MPNet model'),
        (v_huggingface_id, 'sentence-transformers/ms-marco-MiniLM-L-6-v2', 'embedding', 384, 'MS MARCO MiniLM model')
    ON CONFLICT (provider_id, model_name) DO NOTHING;

    -- Local Models
    INSERT INTO neurondb.llm_models (provider_id, model_name, model_type, embedding_dimension, is_default, description)
    VALUES
        (v_local_id, 'default', 'embedding', 384, true, 'Generic local embedding model'),
        (v_local_id, 'local-embedding-small', 'embedding', 384, false, 'Small local embedding model'),
        (v_local_id, 'local-embedding-base', 'embedding', 768, false, 'Base local embedding model'),
        (v_local_id, 'local-embedding-large', 'embedding', 1024, false, 'Large local embedding model'),
        (v_local_id, 'local-embedding-multilingual', 'embedding', 512, false, 'Multilingual local embedding model')
    ON CONFLICT (provider_id, model_name) DO NOTHING;

    -- Reranking Models (using OpenAI provider for OpenAI models)
    INSERT INTO neurondb.llm_models (provider_id, model_name, model_type, description)
    VALUES
        (v_openai_id, 'text-search-ada-doc-001', 'rerank', 'OpenAI text search document model'),
        (v_openai_id, 'text-search-ada-query-001', 'rerank', 'OpenAI text search query model')
    ON CONFLICT (provider_id, model_name) DO NOTHING;

    -- HuggingFace Reranking Models
    INSERT INTO neurondb.llm_models (provider_id, model_name, model_type, description)
    VALUES
        (v_huggingface_id, 'cohere/rerank-english-v3.0', 'rerank', 'Cohere reranking model'),
        (v_huggingface_id, 'cross-encoder/ms-marco-MiniLM-L-6-v2', 'rerank', 'Cross-encoder reranking model'),
        (v_huggingface_id, 'BAAI/bge-reranker-base', 'rerank', 'BAAI reranker model')
    ON CONFLICT (provider_id, model_name) DO NOTHING;
END $$;

-- Insert Index Templates
INSERT INTO neurondb.index_templates (template_name, description, index_type, config_json, is_default)
VALUES
    ('hnsw-fast', 'Fast HNSW index for quick searches', 'hnsw', '{"m": 16, "ef_construction": 100, "ef_search": 32}', false),
    ('hnsw-balanced', 'Balanced HNSW index (default)', 'hnsw', '{"m": 16, "ef_construction": 200, "ef_search": 64}', true),
    ('hnsw-precise', 'Precise HNSW index for high recall', 'hnsw', '{"m": 32, "ef_construction": 400, "ef_search": 128}', false),
    ('ivf-fast', 'Fast IVF index', 'ivf', '{"lists": 100, "probes": 10}', false),
    ('ivf-balanced', 'Balanced IVF index', 'ivf', '{"lists": 256, "probes": 32}', true),
    ('ivf-precise', 'Precise IVF index', 'ivf', '{"lists": 512, "probes": 64}', false)
ON CONFLICT (template_name) DO NOTHING;

-- Insert Worker Configurations
INSERT INTO neurondb.worker_configs (worker_name, display_name, enabled, naptime_ms, config_json, is_default)
VALUES
    ('neuranq', 'Queue Executor', true, 1000, '{"queue_depth": 10000, "batch_size": 100, "timeout": 30000, "max_retries": 3}', true),
    ('neuranmon', 'Auto-Tuner', true, 60000, '{"sample_size": 1000, "target_latency": 100.0, "target_recall": 0.95}', true),
    ('neurandefrag', 'Index Maintenance', true, 300000, '{"compact_threshold": 1000, "fragmentation_threshold": 0.3, "maintenance_window": "02:00-04:00"}', true)
ON CONFLICT (worker_name) DO NOTHING;

-- Insert ML Default Configurations
INSERT INTO neurondb.ml_default_configs (algorithm, default_hyperparameters, use_gpu, is_default)
VALUES
    ('linear_regression', '{}', false, true),
    ('logistic_regression', '{}', false, true),
    ('random_forest', '{"n_estimators": 100, "max_depth": 10}', false, true),
    ('svm', '{"C": 1.0, "kernel": "rbf"}', false, true),
    ('kmeans', '{"n_clusters": 8, "max_iter": 300}', false, true),
    ('gmm', '{"n_components": 8}', false, true),
    ('xgboost', '{"n_estimators": 100, "max_depth": 6}', true, true),
    ('naive_bayes', '{}', false, true),
    ('ridge', '{"alpha": 1.0}', false, true),
    ('lasso', '{"alpha": 1.0}', false, true),
    ('knn', '{"n_neighbors": 5}', false, true)
ON CONFLICT (algorithm) DO NOTHING;

-- Insert Tool Configurations
INSERT INTO neurondb.tool_configs (tool_name, display_name, default_params, default_limit, default_timeout_ms, enabled, is_default)
VALUES
    ('vector_search', 'Vector Search', '{"distance_metric": "l2"}', 10, 30000, true, true),
    ('generate_embedding', 'Generate Embedding', '{}', NULL, 30000, true, true),
    ('batch_embedding', 'Batch Embedding', '{}', NULL, 60000, true, true),
    ('rag', 'RAG Operations', '{"top_k": 5}', 5, 30000, true, true),
    ('analytics', 'Analytics Tools', '{}', NULL, 30000, true, true),
    ('ml_training', 'ML Training', '{}', NULL, 3600000, true, true),
    ('ml_prediction', 'ML Prediction', '{}', NULL, 30000, true, true)
ON CONFLICT (tool_name) DO NOTHING;

-- Insert System Configuration
INSERT INTO neurondb.system_configs (config_key, config_value, description, is_default)
VALUES
    ('features', '{"vector": true, "ml": true, "analytics": true, "rag": true}', 'Feature flags', true),
    ('default_timeout_ms', '30000', 'Default timeout for operations', true),
    ('rate_limiting', '{"enabled": false, "requests_per_minute": 60}', 'Rate limiting configuration', true),
    ('caching', '{"enabled": true, "ttl_seconds": 3600}', 'Caching policy', true)
ON CONFLICT (config_key) DO NOTHING;

-- ============================================================================
-- PART 7: PROMPTS (1 table)
-- ============================================================================

-- 14. Prompts Table
CREATE TABLE IF NOT EXISTS neurondb.prompts (
    prompt_id SERIAL PRIMARY KEY,
    prompt_name TEXT NOT NULL UNIQUE,
    description TEXT,
    template TEXT NOT NULL,  -- Prompt template with variables
    variables JSONB DEFAULT '[]',  -- Array of variable definitions
    category TEXT,  -- Optional category for organization
    tags TEXT[],  -- Tags for searchability
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT DEFAULT CURRENT_USER
);
COMMENT ON TABLE neurondb.prompts IS 'MCP prompt templates with variable support';

-- Prompts indexes
CREATE INDEX IF NOT EXISTS idx_prompts_name ON neurondb.prompts(prompt_name);
CREATE INDEX IF NOT EXISTS idx_prompts_category ON neurondb.prompts(category) WHERE category IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_prompts_tags ON neurondb.prompts USING GIN(tags);

-- Trigger for prompts updated_at
CREATE TRIGGER trigger_prompts_updated_at
    BEFORE UPDATE ON neurondb.prompts
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

-- Insert default prompts
INSERT INTO neurondb.prompts (prompt_name, description, template, variables, category, tags, is_default)
VALUES
    ('rag-query', 'RAG query prompt template', 'Context:\n{{context}}\n\nQuestion: {{question}}\n\nAnswer:', '[{"name": "context", "description": "Retrieved context", "required": true}, {"name": "question", "description": "User question", "required": true}]', 'rag', ARRAY['rag', 'query', 'qa'], true),
    ('summarization', 'Text summarization prompt', 'Summarize the following text:\n\n{{text}}', '[{"name": "text", "description": "Text to summarize", "required": true}]', 'text', ARRAY['summarization', 'text'], false),
    ('code-explanation', 'Code explanation prompt', 'Explain the following code:\n\n```{{language}}\n{{code}}\n```', '[{"name": "language", "description": "Programming language", "required": true}, {"name": "code", "description": "Code to explain", "required": true}]', 'code', ARRAY['code', 'explanation'], false)
ON CONFLICT (prompt_name) DO NOTHING;

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE 'NeuronMCP Configuration Schema setup completed successfully!';
    RAISE NOTICE 'Created 14 tables, indexes, views, triggers, and pre-populated default data.';
    RAISE NOTICE 'Next steps:';
    RAISE NOTICE '1. Set API keys using: SELECT neurondb_set_model_key(''model_name'', ''api_key'');';
    RAISE NOTICE '2. Verify setup: SELECT * FROM neurondb.v_llm_models_active;';
    RAISE NOTICE '3. Check ready models: SELECT * FROM neurondb.v_llm_models_ready;';
    RAISE NOTICE '4. List prompts: SELECT * FROM neurondb.prompts;';
END $$;

