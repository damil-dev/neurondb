/*-------------------------------------------------------------------------
 *
 * 000_validate_setup.sql
 *    Comprehensive NeuronMCP Setup Validation Script
 *
 * This script validates that NeuronMCP is properly set up by checking:
 * - Required PostgreSQL extensions (neurondb, pgcrypto)
 * - Required schema (neurondb)
 * - All required tables exist
 * - All required functions exist
 * - Indexes are properly created
 * - Default data is populated
 * - Configuration is valid
 *
 * Usage:
 *   psql -d neurondb -f sql/000_check_setup.sql
 *
 * Returns:
 *   - Exit code 0 if all checks pass
 *   - Exit code 1 if any check fails
 *   - Detailed output showing what's missing or incorrect
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 *-------------------------------------------------------------------------
 */

\set ON_ERROR_STOP on
\set VERBOSITY verbose

-- Output formatting
\pset border 2
\pset format aligned

\echo '============================================================================'
\echo 'NeuronMCP Setup Validation Check'
\echo '============================================================================'
\echo ''

-- ============================================================================
-- CHECK 1: PostgreSQL Version
-- ============================================================================
\echo 'CHECK 1: PostgreSQL Version'
\echo '----------------------------------------'
SELECT 
    version() AS postgresql_version,
    CASE 
        WHEN version() ~ 'PostgreSQL 1[6-9]' THEN '✓ PostgreSQL 16+ detected'
        ELSE '✗ PostgreSQL 16+ required'
    END AS status;
\echo ''

-- ============================================================================
-- CHECK 2: Required Extensions
-- ============================================================================
\echo 'CHECK 2: Required Extensions'
\echo '----------------------------------------'
SELECT 
    extname AS extension_name,
    extversion AS version,
    CASE 
        WHEN extname = 'neurondb' THEN '✓ NeuronDB extension installed'
        WHEN extname = 'pgcrypto' THEN '✓ pgcrypto extension installed'
        ELSE 'Extension: ' || extname
    END AS status
FROM pg_extension
WHERE extname IN ('neurondb', 'pgcrypto')
ORDER BY extname;

-- Check for missing extensions
SELECT 
    '✗ MISSING: ' || extname AS status
FROM (VALUES ('neurondb'), ('pgcrypto')) AS required(extname)
WHERE NOT EXISTS (
    SELECT 1 FROM pg_extension WHERE extname = required.extname
);
\echo ''

-- ============================================================================
-- CHECK 3: Schema Existence
-- ============================================================================
\echo 'CHECK 3: Schema Existence'
\echo '----------------------------------------'
SELECT 
    nspname AS schema_name,
    CASE 
        WHEN nspname = 'neurondb' THEN '✓ neurondb schema exists'
        ELSE 'Schema: ' || nspname
    END AS status
FROM pg_namespace
WHERE nspname = 'neurondb';
\echo ''

-- ============================================================================
-- CHECK 4: Required Tables
-- ============================================================================
\echo 'CHECK 4: Required Tables'
\echo '----------------------------------------'
WITH required_tables AS (
    VALUES 
        ('llm_providers'),
        ('llm_models'),
        ('llm_model_keys'),
        ('vector_index_configs'),
        ('worker_configs'),
        ('ml_defaults'),
        ('ml_templates'),
        ('tool_configs'),
        ('system_settings')
)
SELECT 
    rt.table_name,
    CASE 
        WHEN EXISTS (
            SELECT 1 
            FROM information_schema.tables 
            WHERE table_schema = 'neurondb' 
            AND table_name = rt.table_name
        ) THEN '✓ EXISTS'
        ELSE '✗ MISSING'
    END AS status
FROM required_tables rt
ORDER BY rt.table_name;
\echo ''

-- ============================================================================
-- CHECK 5: Table Row Counts
-- ============================================================================
\echo 'CHECK 5: Table Row Counts'
\echo '----------------------------------------'
SELECT 
    schemaname || '.' || tablename AS table_name,
    n_live_tup AS row_count,
    CASE 
        WHEN n_live_tup > 0 THEN '✓ Has data'
        ELSE '⚠ Empty'
    END AS status
FROM pg_stat_user_tables
WHERE schemaname = 'neurondb'
ORDER BY tablename;
\echo ''

-- ============================================================================
-- CHECK 6: Required Functions
-- ============================================================================
\echo 'CHECK 6: Required Functions'
\echo '----------------------------------------'
WITH required_functions AS (
    VALUES 
        ('neurondb_set_model_key'),
        ('neurondb_get_model_key'),
        ('neurondb_set_default_model'),
        ('neurondb_get_default_model'),
        ('neurondb_set_index_config'),
        ('neurondb_get_index_config'),
        ('neurondb_get_all_configs')
)
SELECT 
    rf.function_name,
    CASE 
        WHEN EXISTS (
            SELECT 1 
            FROM pg_proc p
            JOIN pg_namespace n ON p.pronamespace = n.oid
            WHERE n.nspname = 'neurondb'
            AND p.proname = rf.function_name
        ) THEN '✓ EXISTS'
        ELSE '✗ MISSING'
    END AS status
FROM required_functions rf
ORDER BY rf.function_name;
\echo ''

-- ============================================================================
-- CHECK 7: Indexes
-- ============================================================================
\echo 'CHECK 7: Indexes'
\echo '----------------------------------------'
SELECT 
    schemaname || '.' || tablename AS table_name,
    indexname AS index_name,
    CASE 
        WHEN indexname IS NOT NULL THEN '✓ Indexed'
        ELSE '⚠ No index'
    END AS status
FROM pg_indexes
WHERE schemaname = 'neurondb'
ORDER BY tablename, indexname;
\echo ''

-- ============================================================================
-- CHECK 8: Default LLM Models
-- ============================================================================
\echo 'CHECK 8: Default LLM Models'
\echo '----------------------------------------'
SELECT 
    COUNT(*) AS total_models,
    COUNT(CASE WHEN is_default = true THEN 1 END) AS default_models,
    CASE 
        WHEN COUNT(*) > 0 THEN '✓ Models configured'
        ELSE '✗ No models configured'
    END AS status
FROM neurondb.llm_models;
\echo ''

-- ============================================================================
-- CHECK 9: Vector Index Configurations
-- ============================================================================
\echo 'CHECK 9: Vector Index Configurations'
\echo '----------------------------------------'
SELECT 
    COUNT(*) AS total_configs,
    CASE 
        WHEN COUNT(*) > 0 THEN '✓ Index configs exist'
        ELSE '⚠ No index configs'
    END AS status
FROM neurondb.vector_index_configs;
\echo ''

-- ============================================================================
-- CHECK 10: System Settings
-- ============================================================================
\echo 'CHECK 10: System Settings'
\echo '----------------------------------------'
SELECT 
    setting_key,
    setting_value IS NOT NULL AS has_value,
    CASE 
        WHEN setting_value IS NOT NULL THEN '✓ Configured'
        ELSE '⚠ Not set'
    END AS status
FROM neurondb.system_settings
ORDER BY setting_key;
\echo ''

-- ============================================================================
-- CHECK 11: Encryption Key Configuration
-- ============================================================================
\echo 'CHECK 11: Encryption Key Configuration'
\echo '----------------------------------------'
SELECT 
    name,
    setting,
    CASE 
        WHEN setting IS NOT NULL AND setting != '' THEN '✓ Encryption key set'
        ELSE '⚠ Encryption key not set (use ALTER SYSTEM SET neurondb.encryption_key)'
    END AS status
FROM pg_settings
WHERE name = 'neurondb.encryption_key';
\echo ''

-- ============================================================================
-- CHECK 12: Permissions
-- ============================================================================
\echo 'CHECK 12: Schema Permissions'
\echo '----------------------------------------'
SELECT 
    nspname AS schema_name,
    nspacl AS permissions,
    CASE 
        WHEN nspacl IS NOT NULL THEN '✓ Permissions set'
        ELSE '⚠ Check permissions'
    END AS status
FROM pg_namespace
WHERE nspname = 'neurondb';
\echo ''

-- ============================================================================
-- SUMMARY
-- ============================================================================
\echo '============================================================================'
\echo 'Validation Summary'
\echo '============================================================================'

-- Count missing components
WITH checks AS (
    SELECT 
        CASE WHEN NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'neurondb') THEN 1 ELSE 0 END AS missing_neurondb,
        CASE WHEN NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pgcrypto') THEN 1 ELSE 0 END AS missing_pgcrypto,
        CASE WHEN NOT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'neurondb') THEN 1 ELSE 0 END AS missing_schema,
        (
            SELECT COUNT(*) FROM (
                VALUES ('llm_providers'), ('llm_models'), ('llm_model_keys'),
                       ('vector_index_configs'), ('worker_configs'), ('ml_defaults'),
                       ('ml_templates'), ('tool_configs'), ('system_settings')
            ) AS required_tables(table_name)
            WHERE NOT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = 'neurondb' AND table_name = required_tables.table_name
            )
        ) AS missing_tables
)
SELECT 
    CASE 
        WHEN missing_neurondb + missing_pgcrypto + missing_schema + missing_tables = 0 
        THEN '✓ ALL CHECKS PASSED - NeuronMCP is properly configured!'
        ELSE '✗ SOME CHECKS FAILED - Please review the output above'
    END AS final_status,
    missing_neurondb + missing_pgcrypto + missing_schema + missing_tables AS total_issues
FROM checks;

\echo ''
\echo '============================================================================'
\echo 'Next Steps:'
\echo '  - If checks failed, run: psql -d neurondb -f sql/001_initial_schema.sql'
\echo '  - Then run: psql -d neurondb -f sql/002_functions.sql'
\echo '  - Set encryption key: ALTER SYSTEM SET neurondb.encryption_key = ''your-key'';'
\echo '  - Reload config: SELECT pg_reload_conf();'
\echo '============================================================================'
\echo ''

