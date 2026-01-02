/*-------------------------------------------------------------------------
 *
 * neurondb_mcp_functions.sql
 *    Comprehensive NeuronMCP Management Functions
 *
 * Provides 30+ functions for managing all NeuronMCP configurations:
 * - LLM model and key management
 * - Vector index configurations
 * - Worker settings
 * - ML defaults and templates
 * - Tool configurations
 * - System-wide settings
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 *-------------------------------------------------------------------------
 */

-- ============================================================================
-- HELPER FUNCTIONS FOR ENCRYPTION
-- ============================================================================

-- Get encryption key from GUC or environment
CREATE OR REPLACE FUNCTION neurondb.get_encryption_key()
RETURNS TEXT AS $$
DECLARE
    v_key TEXT;
BEGIN
    -- Try to get from GUC setting first
    BEGIN
        v_key := current_setting('neurondb.encryption_key', true);
    EXCEPTION WHEN OTHERS THEN
        v_key := NULL;
    END;
    
    -- If not found, use a default (in production, should be set via GUC)
    IF v_key IS NULL OR v_key = '' THEN
        v_key := 'neurondb_default_encryption_key_change_in_production';
    END IF;
    
    RETURN v_key;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- LLM MODEL KEY MANAGEMENT FUNCTIONS
-- ============================================================================

-- Set/update API key for a model
CREATE OR REPLACE FUNCTION neurondb_set_model_key(
    p_model_name TEXT,
    p_api_key TEXT,
    p_expires_at TIMESTAMPTZ DEFAULT NULL
)
RETURNS BOOLEAN AS $$
DECLARE
    v_model_id INTEGER;
    v_salt BYTEA;
    v_encrypted_key BYTEA;
    v_encryption_key TEXT;
BEGIN
    -- Get model ID
    SELECT model_id INTO v_model_id
    FROM neurondb.llm_models
    WHERE model_name = p_model_name;
    
    IF v_model_id IS NULL THEN
        RAISE EXCEPTION 'Model % not found', p_model_name;
    END IF;
    
    -- Generate salt and encrypt key
    v_salt := gen_random_bytes(32);
    v_encryption_key := neurondb.get_encryption_key();
    v_encrypted_key := pgp_sym_encrypt(p_api_key, v_encryption_key);
    
    -- Insert or update key
    INSERT INTO neurondb.llm_model_keys (model_id, api_key_encrypted, encryption_salt, expires_at)
    VALUES (v_model_id, v_encrypted_key, v_salt, p_expires_at)
    ON CONFLICT (model_id) DO UPDATE
    SET api_key_encrypted = v_encrypted_key,
        encryption_salt = v_salt,
        expires_at = p_expires_at,
        updated_at = NOW();
    
    RETURN true;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Get decrypted API key (for internal use only)
CREATE OR REPLACE FUNCTION neurondb_get_model_key(p_model_name TEXT)
RETURNS TEXT AS $$
DECLARE
    v_model_id INTEGER;
    v_encrypted_key BYTEA;
    v_decrypted_key TEXT;
    v_encryption_key TEXT;
BEGIN
    -- Get model ID
    SELECT model_id INTO v_model_id
    FROM neurondb.llm_models
    WHERE model_name = p_model_name;
    
    IF v_model_id IS NULL THEN
        RAISE EXCEPTION 'Model % not found', p_model_name;
    END IF;
    
    -- Get encrypted key
    SELECT api_key_encrypted INTO v_encrypted_key
    FROM neurondb.llm_model_keys
    WHERE model_id = v_model_id;
    
    IF v_encrypted_key IS NULL THEN
        RETURN NULL;
    END IF;
    
    -- Check expiration
    IF EXISTS (
        SELECT 1 FROM neurondb.llm_model_keys
        WHERE model_id = v_model_id
        AND expires_at IS NOT NULL
        AND expires_at < NOW()
    ) THEN
        RAISE EXCEPTION 'API key for model % has expired', p_model_name;
    END IF;
    
    -- Decrypt key
    v_encryption_key := neurondb.get_encryption_key();
    v_decrypted_key := pgp_sym_decrypt(v_encrypted_key, v_encryption_key);
    
    -- Update access tracking
    UPDATE neurondb.llm_model_keys
    SET last_used_at = NOW(),
        access_count = access_count + 1
    WHERE model_id = v_model_id;
    
    RETURN v_decrypted_key;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Remove API key from model
CREATE OR REPLACE FUNCTION neurondb_remove_model_key(p_model_name TEXT)
RETURNS BOOLEAN AS $$
DECLARE
    v_model_id INTEGER;
BEGIN
    SELECT model_id INTO v_model_id
    FROM neurondb.llm_models
    WHERE model_name = p_model_name;
    
    IF v_model_id IS NULL THEN
        RAISE EXCEPTION 'Model % not found', p_model_name;
    END IF;
    
    DELETE FROM neurondb.llm_model_keys
    WHERE model_id = v_model_id;
    
    RETURN true;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Rotate API key securely
CREATE OR REPLACE FUNCTION neurondb_rotate_model_key(
    p_model_name TEXT,
    p_new_key TEXT
)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN neurondb_set_model_key(p_model_name, p_new_key);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Internal function for NeuronMCP integration (resolves key)
CREATE OR REPLACE FUNCTION neurondb_resolve_model_key(p_model_name TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN neurondb_get_model_key(p_model_name);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- LLM MODEL MANAGEMENT FUNCTIONS
-- ============================================================================

-- List models with optional filters
CREATE OR REPLACE FUNCTION neurondb_list_models(
    p_provider_name TEXT DEFAULT NULL,
    p_model_type TEXT DEFAULT NULL,
    p_status TEXT DEFAULT NULL
)
RETURNS TABLE (
    model_id INTEGER,
    model_name TEXT,
    provider_name TEXT,
    model_type TEXT,
    status TEXT,
    has_api_key BOOLEAN,
    is_default BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.model_id,
        m.model_name,
        p.provider_name,
        m.model_type,
        m.status::TEXT,
        CASE WHEN mk.key_id IS NOT NULL THEN true ELSE false END AS has_api_key,
        m.is_default
    FROM neurondb.llm_models m
    JOIN neurondb.llm_providers p ON m.provider_id = p.provider_id
    LEFT JOIN neurondb.llm_model_keys mk ON m.model_id = mk.model_id
    WHERE (p_provider_name IS NULL OR p.provider_name = p.provider_name)
      AND (p_model_type IS NULL OR m.model_type = p_model_type)
      AND (p_status IS NULL OR m.status = p_status)
    ORDER BY m.model_name;
END;
$$ LANGUAGE plpgsql;

-- Get complete model configuration
CREATE OR REPLACE FUNCTION neurondb_get_model_config(p_model_name TEXT)
RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'model_id', m.model_id,
        'model_name', m.model_name,
        'provider', p.provider_name,
        'model_type', m.model_type,
        'context_window', m.context_window,
        'embedding_dimension', m.embedding_dimension,
        'has_api_key', CASE WHEN mk.key_id IS NOT NULL THEN true ELSE false END,
        'config', mc.default_params,
        'base_url', mc.base_url
    ) INTO v_result
    FROM neurondb.llm_models m
    JOIN neurondb.llm_providers p ON m.provider_id = p.provider_id
    LEFT JOIN neurondb.llm_model_keys mk ON m.model_id = mk.model_id
    LEFT JOIN neurondb.llm_model_configs mc ON m.model_id = mc.model_id AND mc.is_active = true
    WHERE m.model_name = p_model_name;
    
    RETURN v_result;
END;
$$ LANGUAGE plpgsql;

-- Enable a model
CREATE OR REPLACE FUNCTION neurondb_enable_model(p_model_name TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE neurondb.llm_models
    SET status = 'available'
    WHERE model_name = p_model_name;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Model % not found', p_model_name;
    END IF;
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Disable a model
CREATE OR REPLACE FUNCTION neurondb_disable_model(p_model_name TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE neurondb.llm_models
    SET status = 'disabled'
    WHERE model_name = p_model_name;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Model % not found', p_model_name;
    END IF;
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Set default model for type
CREATE OR REPLACE FUNCTION neurondb_set_default_model(
    p_model_name TEXT,
    p_model_type TEXT
)
RETURNS BOOLEAN AS $$
BEGIN
    -- Clear existing default for this type
    UPDATE neurondb.llm_models
    SET is_default = false
    WHERE model_type = p_model_type;
    
    -- Set new default
    UPDATE neurondb.llm_models
    SET is_default = true
    WHERE model_name = p_model_name AND model_type = p_model_type;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Model % not found or type mismatch', p_model_name;
    END IF;
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Smart model selection
CREATE OR REPLACE FUNCTION neurondb_get_model_for_operation(
    p_operation_type TEXT,
    p_preferred_model TEXT DEFAULT NULL
)
RETURNS TEXT AS $$
DECLARE
    v_model_name TEXT;
BEGIN
    -- If preferred model is specified and available, use it
    IF p_preferred_model IS NOT NULL THEN
        SELECT model_name INTO v_model_name
        FROM neurondb.llm_models m
        JOIN neurondb.llm_providers p ON m.provider_id = p.provider_id
        WHERE m.model_name = p_preferred_model
          AND m.status = 'available'
          AND p.status = 'active'
          AND EXISTS (SELECT 1 FROM neurondb.llm_model_keys WHERE model_id = m.model_id);
        
        IF v_model_name IS NOT NULL THEN
            RETURN v_model_name;
        END IF;
    END IF;
    
    -- Otherwise, get default model for operation type
    SELECT model_name INTO v_model_name
    FROM neurondb.llm_models m
    JOIN neurondb.llm_providers p ON m.provider_id = p.provider_id
    WHERE m.model_type = p_operation_type
      AND m.is_default = true
      AND m.status = 'available'
      AND p.status = 'active'
      AND EXISTS (SELECT 1 FROM neurondb.llm_model_keys WHERE model_id = m.model_id)
    LIMIT 1;
    
    RETURN v_model_name;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- LLM MODEL CONFIGURATION FUNCTIONS
-- ============================================================================

-- Set model configuration
CREATE OR REPLACE FUNCTION neurondb_set_model_config(
    p_model_name TEXT,
    p_config_name TEXT DEFAULT 'default',
    p_config_json JSONB DEFAULT '{}'
)
RETURNS BOOLEAN AS $$
DECLARE
    v_model_id INTEGER;
BEGIN
    SELECT model_id INTO v_model_id
    FROM neurondb.llm_models
    WHERE model_name = p_model_name;
    
    IF v_model_id IS NULL THEN
        RAISE EXCEPTION 'Model % not found', p_model_name;
    END IF;
    
    INSERT INTO neurondb.llm_model_configs (model_id, config_name, default_params)
    VALUES (v_model_id, p_config_name, p_config_json)
    ON CONFLICT (model_id, config_name) DO UPDATE
    SET default_params = p_config_json,
        updated_at = NOW();
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Get model configuration
CREATE OR REPLACE FUNCTION neurondb_get_model_config(
    p_model_name TEXT,
    p_config_name TEXT DEFAULT 'default'
)
RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT mc.default_params INTO v_result
    FROM neurondb.llm_models m
    JOIN neurondb.llm_model_configs mc ON m.model_id = mc.model_id
    WHERE m.model_name = p_model_name
      AND mc.config_name = p_config_name
      AND mc.is_active = true;
    
    RETURN COALESCE(v_result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- Reset model configuration to defaults
CREATE OR REPLACE FUNCTION neurondb_reset_model_config(p_model_name TEXT)
RETURNS BOOLEAN AS $$
DECLARE
    v_model_id INTEGER;
BEGIN
    SELECT model_id INTO v_model_id
    FROM neurondb.llm_models
    WHERE model_name = p_model_name;
    
    IF v_model_id IS NULL THEN
        RAISE EXCEPTION 'Model % not found', p_model_name;
    END IF;
    
    UPDATE neurondb.llm_model_configs
    SET default_params = '{}',
        updated_at = NOW()
    WHERE model_id = v_model_id;
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PROVIDER MANAGEMENT FUNCTIONS
-- ============================================================================

-- Add custom provider
CREATE OR REPLACE FUNCTION neurondb_add_provider(
    p_provider_name TEXT,
    p_config_json JSONB
)
RETURNS INTEGER AS $$
DECLARE
    v_provider_id INTEGER;
BEGIN
    INSERT INTO neurondb.llm_providers (
        provider_name,
        display_name,
        default_base_url,
        auth_method,
        metadata
    )
    VALUES (
        p_provider_name,
        COALESCE(p_config_json->>'display_name', p_provider_name),
        p_config_json->>'default_base_url',
        COALESCE(p_config_json->>'auth_method', 'api_key'),
        p_config_json
    )
    RETURNING provider_id INTO v_provider_id;
    
    RETURN v_provider_id;
END;
$$ LANGUAGE plpgsql;

-- List all providers
CREATE OR REPLACE FUNCTION neurondb_list_providers()
RETURNS TABLE (
    provider_id INTEGER,
    provider_name TEXT,
    display_name TEXT,
    status TEXT,
    supports_embeddings BOOLEAN,
    supports_chat BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p.provider_id,
        p.provider_name,
        p.display_name,
        p.status::TEXT,
        p.supports_embeddings,
        p.supports_chat
    FROM neurondb.llm_providers p
    ORDER BY p.provider_name;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- USAGE & ANALYTICS FUNCTIONS
-- ============================================================================

-- Log model usage
CREATE OR REPLACE FUNCTION neurondb_log_model_usage(
    p_model_name TEXT,
    p_operation_type TEXT,
    p_tokens_input INTEGER DEFAULT NULL,
    p_tokens_output INTEGER DEFAULT NULL,
    p_latency_ms INTEGER DEFAULT NULL,
    p_success BOOLEAN DEFAULT true,
    p_error_message TEXT DEFAULT NULL
)
RETURNS BIGINT AS $$
DECLARE
    v_model_id INTEGER;
    v_usage_id BIGINT;
    v_cost NUMERIC(10,6) := 0;
BEGIN
    SELECT model_id INTO v_model_id
    FROM neurondb.llm_models
    WHERE model_name = p_model_name;
    
    IF v_model_id IS NULL THEN
        RAISE EXCEPTION 'Model % not found', p_model_name;
    END IF;
    
    -- Calculate cost if pricing available
    IF p_tokens_input IS NOT NULL OR p_tokens_output IS NOT NULL THEN
        SELECT 
            COALESCE((p_tokens_input::NUMERIC / 1000.0) * m.cost_per_1k_tokens_input, 0) +
            COALESCE((p_tokens_output::NUMERIC / 1000.0) * m.cost_per_1k_tokens_output, 0)
        INTO v_cost
        FROM neurondb.llm_models m
        WHERE m.model_id = v_model_id;
    END IF;
    
    INSERT INTO neurondb.llm_model_usage (
        model_id,
        operation_type,
        tokens_input,
        tokens_output,
        cost,
        latency_ms,
        success,
        error_message
    )
    VALUES (
        v_model_id,
        p_operation_type,
        p_tokens_input,
        p_tokens_output,
        v_cost,
        p_latency_ms,
        p_success,
        p_error_message
    )
    RETURNING usage_id INTO v_usage_id;
    
    -- Update last_used_at on model
    UPDATE neurondb.llm_models
    SET last_used_at = NOW()
    WHERE model_id = v_model_id;
    
    RETURN v_usage_id;
END;
$$ LANGUAGE plpgsql;

-- Get model statistics
CREATE OR REPLACE FUNCTION neurondb_get_model_stats(
    p_model_name TEXT,
    p_days INTEGER DEFAULT 30
)
RETURNS JSONB AS $$
DECLARE
    v_model_id INTEGER;
    v_result JSONB;
BEGIN
    SELECT model_id INTO v_model_id
    FROM neurondb.llm_models
    WHERE model_name = p_model_name;
    
    IF v_model_id IS NULL THEN
        RAISE EXCEPTION 'Model % not found', p_model_name;
    END IF;
    
    SELECT jsonb_build_object(
        'total_requests', COUNT(*),
        'successful_requests', COUNT(*) FILTER (WHERE success = true),
        'failed_requests', COUNT(*) FILTER (WHERE success = false),
        'total_tokens_input', SUM(tokens_input),
        'total_tokens_output', SUM(tokens_output),
        'total_cost', SUM(cost),
        'avg_latency_ms', AVG(latency_ms),
        'operations', jsonb_object_agg(operation_type, COUNT(*))
    ) INTO v_result
    FROM neurondb.llm_model_usage
    WHERE model_id = v_model_id
      AND created_at >= NOW() - (p_days || ' days')::INTERVAL;
    
    RETURN COALESCE(v_result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- Get cost summary
CREATE OR REPLACE FUNCTION neurondb_get_cost_summary(p_days INTEGER DEFAULT 30)
RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'total_cost', SUM(cost),
        'by_model', (
            SELECT jsonb_object_agg(m.model_name, SUM(u.cost))
            FROM neurondb.llm_model_usage u
            JOIN neurondb.llm_models m ON u.model_id = m.model_id
            WHERE u.created_at >= NOW() - (p_days || ' days')::INTERVAL
            GROUP BY m.model_name
        ),
        'by_operation', (
            SELECT jsonb_object_agg(operation_type, SUM(cost))
            FROM neurondb.llm_model_usage
            WHERE created_at >= NOW() - (p_days || ' days')::INTERVAL
            GROUP BY operation_type
        )
    ) INTO v_result
    FROM neurondb.llm_model_usage
    WHERE created_at >= NOW() - (p_days || ' days')::INTERVAL;
    
    RETURN COALESCE(v_result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VECTOR INDEX MANAGEMENT FUNCTIONS
-- ============================================================================

-- Get index configuration
CREATE OR REPLACE FUNCTION neurondb_get_index_config(
    p_table_name TEXT,
    p_vector_column TEXT
)
RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'index_type', index_type,
        'hnsw_m', hnsw_m,
        'hnsw_ef_construction', hnsw_ef_construction,
        'hnsw_ef_search', hnsw_ef_search,
        'ivf_lists', ivf_lists,
        'ivf_probes', ivf_probes,
        'distance_metric', distance_metric
    ) INTO v_result
    FROM neurondb.index_configs
    WHERE table_name = p_table_name
      AND vector_column = p_vector_column;
    
    RETURN COALESCE(v_result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- Set index configuration
CREATE OR REPLACE FUNCTION neurondb_set_index_config(
    p_table_name TEXT,
    p_vector_column TEXT,
    p_config_json JSONB
)
RETURNS BOOLEAN AS $$
BEGIN
    INSERT INTO neurondb.index_configs (
        table_name,
        vector_column,
        index_type,
        hnsw_m,
        hnsw_ef_construction,
        hnsw_ef_search,
        ivf_lists,
        ivf_probes,
        distance_metric
    )
    VALUES (
        p_table_name,
        p_vector_column,
        COALESCE(p_config_json->>'index_type', 'hnsw'),
        (p_config_json->>'hnsw_m')::INTEGER,
        (p_config_json->>'hnsw_ef_construction')::INTEGER,
        (p_config_json->>'hnsw_ef_search')::INTEGER,
        (p_config_json->>'ivf_lists')::INTEGER,
        (p_config_json->>'ivf_probes')::INTEGER,
        COALESCE(p_config_json->>'distance_metric', 'l2')
    )
    ON CONFLICT (table_name, vector_column) DO UPDATE
    SET index_type = EXCLUDED.index_type,
        hnsw_m = EXCLUDED.hnsw_m,
        hnsw_ef_construction = EXCLUDED.hnsw_ef_construction,
        hnsw_ef_search = EXCLUDED.hnsw_ef_search,
        ivf_lists = EXCLUDED.ivf_lists,
        ivf_probes = EXCLUDED.ivf_probes,
        distance_metric = EXCLUDED.distance_metric,
        updated_at = NOW();
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Apply index template
CREATE OR REPLACE FUNCTION neurondb_apply_index_template(
    p_template_name TEXT,
    p_table_name TEXT,
    p_vector_column TEXT
)
RETURNS BOOLEAN AS $$
DECLARE
    v_config_json JSONB;
BEGIN
    SELECT config_json INTO v_config_json
    FROM neurondb.index_templates
    WHERE template_name = p_template_name;
    
    IF v_config_json IS NULL THEN
        RAISE EXCEPTION 'Template % not found', p_template_name;
    END IF;
    
    RETURN neurondb_set_index_config(p_table_name, p_vector_column, v_config_json);
END;
$$ LANGUAGE plpgsql;

-- List index templates
CREATE OR REPLACE FUNCTION neurondb_list_index_templates()
RETURNS TABLE (
    template_name TEXT,
    description TEXT,
    index_type TEXT,
    is_default BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        t.template_name,
        t.description,
        t.index_type,
        t.is_default
    FROM neurondb.index_templates t
    ORDER BY t.template_name;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- WORKER MANAGEMENT FUNCTIONS
-- ============================================================================

-- Get worker configuration
CREATE OR REPLACE FUNCTION neurondb_get_worker_config(p_worker_name TEXT)
RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'worker_name', worker_name,
        'enabled', enabled,
        'naptime_ms', naptime_ms,
        'config', config_json
    ) INTO v_result
    FROM neurondb.worker_configs
    WHERE worker_name = p_worker_name;
    
    RETURN COALESCE(v_result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- Set worker configuration
CREATE OR REPLACE FUNCTION neurondb_set_worker_config(
    p_worker_name TEXT,
    p_config_json JSONB
)
RETURNS BOOLEAN AS $$
BEGIN
    INSERT INTO neurondb.worker_configs (worker_name, display_name, config_json)
    VALUES (
        p_worker_name,
        COALESCE(p_config_json->>'display_name', p_worker_name),
        p_config_json
    )
    ON CONFLICT (worker_name) DO UPDATE
    SET config_json = EXCLUDED.config_json,
        updated_at = NOW();
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Enable worker
CREATE OR REPLACE FUNCTION neurondb_enable_worker(p_worker_name TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE neurondb.worker_configs
    SET enabled = true
    WHERE worker_name = p_worker_name;
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- Disable worker
CREATE OR REPLACE FUNCTION neurondb_disable_worker(p_worker_name TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE neurondb.worker_configs
    SET enabled = false
    WHERE worker_name = p_worker_name;
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- ML DEFAULTS MANAGEMENT FUNCTIONS
-- ============================================================================

-- Get ML defaults for algorithm
CREATE OR REPLACE FUNCTION neurondb_get_ml_defaults(p_algorithm TEXT)
RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'algorithm', algorithm,
        'hyperparameters', default_hyperparameters,
        'training_settings', default_training_settings,
        'use_gpu', use_gpu,
        'gpu_device', gpu_device,
        'batch_size', batch_size,
        'max_iterations', max_iterations
    ) INTO v_result
    FROM neurondb.ml_default_configs
    WHERE algorithm = p_algorithm;
    
    RETURN COALESCE(v_result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- Set ML defaults
CREATE OR REPLACE FUNCTION neurondb_set_ml_defaults(
    p_algorithm TEXT,
    p_config_json JSONB
)
RETURNS BOOLEAN AS $$
BEGIN
    INSERT INTO neurondb.ml_default_configs (
        algorithm,
        default_hyperparameters,
        default_training_settings,
        use_gpu,
        gpu_device,
        batch_size,
        max_iterations
    )
    VALUES (
        p_algorithm,
        COALESCE(p_config_json->'hyperparameters', '{}'::jsonb),
        COALESCE(p_config_json->'training_settings', '{}'::jsonb),
        COALESCE((p_config_json->>'use_gpu')::BOOLEAN, false),
        COALESCE((p_config_json->>'gpu_device')::INTEGER, 0),
        (p_config_json->>'batch_size')::INTEGER,
        (p_config_json->>'max_iterations')::INTEGER
    )
    ON CONFLICT (algorithm) DO UPDATE
    SET default_hyperparameters = EXCLUDED.default_hyperparameters,
        default_training_settings = EXCLUDED.default_training_settings,
        use_gpu = EXCLUDED.use_gpu,
        gpu_device = EXCLUDED.gpu_device,
        batch_size = EXCLUDED.batch_size,
        max_iterations = EXCLUDED.max_iterations,
        updated_at = NOW();
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Apply ML template
CREATE OR REPLACE FUNCTION neurondb_apply_ml_template(
    p_template_name TEXT,
    p_project_name TEXT
)
RETURNS JSONB AS $$
DECLARE
    v_template_config JSONB;
BEGIN
    SELECT template_config INTO v_template_config
    FROM neurondb.ml_model_templates
    WHERE template_name = p_template_name;
    
    IF v_template_config IS NULL THEN
        RAISE EXCEPTION 'Template % not found', p_template_name;
    END IF;
    
    -- This would typically call a NeuronDB function to create/configure the project
    -- For now, return the template config
    RETURN v_template_config;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TOOL CONFIGURATION FUNCTIONS
-- ============================================================================

-- Get tool configuration
CREATE OR REPLACE FUNCTION neurondb_get_tool_config(p_tool_name TEXT)
RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'tool_name', tool_name,
        'default_params', default_params,
        'default_limit', default_limit,
        'default_timeout_ms', default_timeout_ms,
        'enabled', enabled
    ) INTO v_result
    FROM neurondb.tool_configs
    WHERE tool_name = p_tool_name;
    
    RETURN COALESCE(v_result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- Set tool configuration
CREATE OR REPLACE FUNCTION neurondb_set_tool_config(
    p_tool_name TEXT,
    p_config_json JSONB
)
RETURNS BOOLEAN AS $$
BEGIN
    INSERT INTO neurondb.tool_configs (
        tool_name,
        display_name,
        default_params,
        default_limit,
        default_timeout_ms,
        enabled
    )
    VALUES (
        p_tool_name,
        COALESCE(p_config_json->>'display_name', p_tool_name),
        COALESCE(p_config_json->'default_params', '{}'::jsonb),
        (p_config_json->>'default_limit')::INTEGER,
        COALESCE((p_config_json->>'default_timeout_ms')::INTEGER, 30000),
        COALESCE((p_config_json->>'enabled')::BOOLEAN, true)
    )
    ON CONFLICT (tool_name) DO UPDATE
    SET default_params = EXCLUDED.default_params,
        default_limit = EXCLUDED.default_limit,
        default_timeout_ms = EXCLUDED.default_timeout_ms,
        enabled = EXCLUDED.enabled,
        updated_at = NOW();
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Reset tool configuration
CREATE OR REPLACE FUNCTION neurondb_reset_tool_config(p_tool_name TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE neurondb.tool_configs
    SET default_params = '{}',
        updated_at = NOW()
    WHERE tool_name = p_tool_name;
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SYSTEM CONFIGURATION FUNCTIONS
-- ============================================================================

-- Get system configuration
CREATE OR REPLACE FUNCTION neurondb_get_system_config()
RETURNS JSONB AS $$
DECLARE
    v_result JSONB := '{}'::jsonb;
    v_row RECORD;
BEGIN
    FOR v_row IN SELECT config_key, config_value FROM neurondb.system_configs
    LOOP
        v_result := v_result || jsonb_build_object(v_row.config_key, v_row.config_value);
    END LOOP;
    
    RETURN v_result;
END;
$$ LANGUAGE plpgsql;

-- Set system configuration
CREATE OR REPLACE FUNCTION neurondb_set_system_config(p_config_json JSONB)
RETURNS BOOLEAN AS $$
DECLARE
    v_key TEXT;
    v_value JSONB;
BEGIN
    FOR v_key, v_value IN SELECT * FROM jsonb_each(p_config_json)
    LOOP
        INSERT INTO neurondb.system_configs (config_key, config_value)
        VALUES (v_key, v_value)
        ON CONFLICT (config_key) DO UPDATE
        SET config_value = EXCLUDED.config_value,
            updated_at = NOW();
    END LOOP;
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Get all configurations (unified view)
CREATE OR REPLACE FUNCTION neurondb_get_all_configs()
RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'llm_models', (
            SELECT jsonb_agg(jsonb_build_object(
                'model_name', model_name,
                'provider', provider_name,
                'type', model_type,
                'has_key', CASE WHEN key_id IS NOT NULL THEN true ELSE false END
            ))
            FROM neurondb.v_llm_models_active
        ),
        'index_templates', (
            SELECT jsonb_agg(jsonb_build_object(
                'template_name', template_name,
                'index_type', index_type
            ))
            FROM neurondb.index_templates
        ),
        'workers', (
            SELECT jsonb_agg(jsonb_build_object(
                'worker_name', worker_name,
                'enabled', enabled
            ))
            FROM neurondb.worker_configs
        ),
        'ml_defaults', (
            SELECT jsonb_agg(algorithm)
            FROM neurondb.ml_default_configs
        ),
        'tools', (
            SELECT jsonb_agg(jsonb_build_object(
                'tool_name', tool_name,
                'enabled', enabled
            ))
            FROM neurondb.tool_configs
        ),
        'system', neurondb_get_system_config()
    ) INTO v_result;
    
    RETURN COALESCE(v_result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql;



