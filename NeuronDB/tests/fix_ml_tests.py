#!/usr/bin/env python3
"""
Script to fix all ML test files to match the pattern from 001_linreg_basic.sql
"""

import re
import os
import sys

# List of ML test files to fix
ML_TEST_FILES = [
    '004_svm_basic.sql',
    '005_dt_basic.sql',
    '006_ridge_basic.sql',
    '007_lasso_basic.sql',
    '008_nb_basic.sql',
    '009_knn_basic.sql',
    '010_xgboost_basic.sql',
    '011_catboost_basic.sql',
    '012_lightgbm_basic.sql',
    '014_gmm_basic.sql',
    '015_kmeans_basic.sql',
    '016_minibatch_kmeans_basic.sql',
    '017_hierarchical_basic.sql',
    '018_dbscan_basic.sql',
    '019_pca_basic.sql',
    '021_automl_basic.sql',
    '022_automl_standalone_basic.sql',
]

BASE_DIR = os.path.join(os.path.dirname(__file__), 'sql', 'basic')

def fix_file(filepath):
    """Fix a single test file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # 1. Add SET client_min_messages TO WARNING; after the header comment
    if 'SET client_min_messages TO WARNING;' not in content:
        content = re.sub(
            r'(\\set ON_ERROR_STOP on)',
            r'SET client_min_messages TO WARNING;\n\1',
            content,
            count=1
        )
    
    # 2. Fix GPU configuration block
    content = re.sub(
        r'DO \$\$\s+DECLARE\s+gpu_mode TEXT;\s+current_gpu_enabled TEXT;\s+BEGIN\s+SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = \'gpu_mode\';\s+SELECT current_setting\(\'neurondb\.compute_mode\', true\) INTO current_gpu_enabled;\s+IF gpu_mode = \'gpu\' THEN\s+SELECT neurondb_gpu_enable\(\);\s+END IF;\s+END \$\$;',
        '''DO $$
DECLARE
	compute_mode TEXT;
BEGIN
	-- Get compute_mode from test_settings (set by run_test.py)
	SELECT setting_value INTO compute_mode FROM test_settings WHERE setting_key = 'compute_mode';
	-- Note: compute_mode is set by run_test.py via switch_gpu_mode()
	-- This block is kept for backward compatibility but compute_mode
	-- should be set before running tests via run_test.py
	IF compute_mode = 'gpu' THEN
		PERFORM neurondb_gpu_enable();
	ELSIF compute_mode = 'auto' THEN
		PERFORM neurondb_gpu_enable();
	END IF;
END $$;''',
        content,
        flags=re.MULTILINE | re.DOTALL
    )
    
    # 3. Replace all gpu_mode with compute_mode
    content = content.replace('gpu_mode TEXT;', 'compute_mode TEXT;')
    content = content.replace("setting_key = 'gpu_mode'", "setting_key = 'compute_mode'")
    content = content.replace('INTO gpu_mode', 'INTO compute_mode')
    content = content.replace('IF gpu_mode =', 'IF compute_mode =')
    content = content.replace('ELSIF gpu_mode =', 'ELSIF compute_mode =')
    
    # 4. Replace SELECT neurondb_gpu_enable() with PERFORM
    content = re.sub(
        r'\s+SELECT neurondb_gpu_enable\(\);',
        r'\n\t\tPERFORM neurondb_gpu_enable();',
        content
    )
    
    # 5. Fix evaluation section - add test_count check
    eval_pattern = r'(DO \$\$\s+DECLARE\s+mid integer;\s+metrics_result jsonb;\s+eval_error text;\s+BEGIN\s+SELECT model_id INTO mid FROM gpu_model_temp LIMIT 1;\s+IF mid IS NULL THEN\s+RAISE WARNING \'No model_id found in gpu_model_temp\';\s+INSERT INTO gpu_metrics_temp VALUES \(\'\{"error": "No model_id found"\}\'::jsonb\);\s+RETURN;\s+END IF;\s+BEGIN\s+BEGIN\s+metrics_result := neurondb\.evaluate\(mid, \'test_test_view\', \'features\', \'label\'\);\s+IF metrics_result IS NULL THEN\s+RAISE WARNING \'Evaluation returned NULL\';\s+INSERT INTO gpu_metrics_temp VALUES \(\'\{"error": "Evaluation returned NULL"\}\'::jsonb\);\s+ELSE\s+INSERT INTO gpu_metrics_temp VALUES \(metrics_result\);\s+END IF;\s+EXCEPTION WHEN OTHERS THEN\s+eval_error := SQLERRM;\s+RAISE WARNING \'Evaluation exception: %\', eval_error;\s+eval_error := REPLACE\(REPLACE\(REPLACE\(eval_error, \'"\', \'\\\\"\', E\'\\n\', \' \'\), E\'\\r\', \' \'\);\s+INSERT INTO gpu_metrics_temp VALUES \(jsonb_build_object\(\'error\', eval_error\)\);\s+END;\s+EXCEPTION WHEN OTHERS THEN\s+eval_error := SQLERRM;\s+RAISE WARNING \'Outer evaluation exception: %\', eval_error;\s+eval_error := REPLACE\(REPLACE\(REPLACE\(eval_error, \'"\', \'\\\\"\', E\'\\n\', \' \'\), E\'\\r\', \' \'\);\s+INSERT INTO gpu_metrics_temp VALUES \(jsonb_build_object\(\'error\', eval_error\)\);\s+END;\s+END \$\$;)'
    
    eval_replacement = '''DO $$
DECLARE
	mid integer;
	metrics_result jsonb;
	eval_error text;
	test_count bigint;
BEGIN
	-- Get model_id
	SELECT model_id INTO mid FROM gpu_model_temp LIMIT 1;
	IF mid IS NULL THEN
		RAISE WARNING 'No model_id found in gpu_model_temp';
		INSERT INTO gpu_metrics_temp VALUES ('{"error": "No model_id found"}'::jsonb);
		RETURN;
	END IF;
	
	-- Verify test data is available
	SELECT COUNT(*) INTO test_count 
	FROM test_test_view 
	WHERE features IS NOT NULL AND label IS NOT NULL;
	
	IF test_count < 1 THEN
		RAISE WARNING 'No valid test samples available (count: %)', test_count;
		INSERT INTO gpu_metrics_temp VALUES (jsonb_build_object('error', 'No valid test samples available'));
		RETURN;
	END IF;
	
	RAISE NOTICE 'Evaluating model_id: % on test_test_view with % valid samples', mid, test_count;
	
	BEGIN
		metrics_result := neurondb.evaluate(mid, 'test_test_view', 'features', 'label');
		
		IF metrics_result IS NULL THEN
			RAISE WARNING 'Evaluation returned NULL for model_id: %', mid;
			INSERT INTO gpu_metrics_temp VALUES ('{"error": "Evaluation returned NULL"}'::jsonb);
		ELSE
			RAISE NOTICE 'Evaluation successful for model_id: %', mid;
			INSERT INTO gpu_metrics_temp VALUES (metrics_result);
		END IF;
	EXCEPTION WHEN OTHERS THEN
		eval_error := SQLERRM;
		RAISE WARNING 'Evaluation exception for model_id %: %', mid, eval_error;
		eval_error := REPLACE(REPLACE(REPLACE(eval_error, '"', '\\"'), E'\\n', ' '), E'\\r', ' ');
		INSERT INTO gpu_metrics_temp VALUES (jsonb_build_object('error', eval_error, 'model_id', mid));
	END;
END $$;'''
    
    # This is complex, let's do it more carefully
    # For now, let's just do the simple replacements and handle evaluation separately
    
    # 6. Fix GPU info display blocks
    gpu_info_pattern1 = r'-- Only show GPU info if GPU mode is enabled - never call GPU functions in CPU mode\s+DO \$\$\s+DECLARE\s+gpu_mode TEXT;\s+BEGIN\s+SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = \'gpu_mode\';\s+IF gpu_mode = \'gpu\' THEN\s+-- Display GPU information only when GPU mode is enabled\s+PERFORM NULL; -- Placeholder for GPU info display\s+END IF;\s+END \$\$;'
    
    gpu_info_replacement1 = '''-- Only show GPU info if GPU mode is enabled - never call GPU functions in CPU mode
DO $$
DECLARE
	compute_mode TEXT;
BEGIN
	SELECT setting_value INTO compute_mode FROM test_settings WHERE setting_key = 'compute_mode';
	
	IF compute_mode = 'gpu' THEN
		-- Display GPU information only when GPU mode is enabled
		PERFORM NULL; -- Placeholder for GPU info display
	END IF;
END $$;'''
    
    content = re.sub(gpu_info_pattern1, gpu_info_replacement1, content, flags=re.MULTILINE | re.DOTALL)
    
    gpu_info_pattern2 = r'-- Conditionally display GPU info only in GPU mode\s+DO \$\$\s+DECLARE\s+gpu_mode TEXT;\s+BEGIN\s+SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = \'gpu_mode\';\s+IF gpu_mode = \'gpu\' THEN\s+BEGIN\s+RAISE NOTICE \'GPU Information:\';\s+PERFORM device_id, device_name, total_memory_mb, free_memory_mb,\s+compute_capability_major, compute_capability_minor, is_available\s+FROM neurondb_gpu_info\(\);\s+EXCEPTION WHEN OTHERS THEN\s+RAISE NOTICE \'GPU information not available\';\s+END;\s+ELSE\s+RAISE NOTICE \'CPU mode: GPU information skipped\';\s+END IF;\s+END \$\$;'
    
    gpu_info_replacement2 = '''-- Conditionally display GPU info only in GPU mode
DO $$
DECLARE
	compute_mode TEXT;
	rec RECORD;
BEGIN
	SELECT setting_value INTO compute_mode FROM test_settings WHERE setting_key = 'compute_mode';
	
	IF compute_mode = 'gpu' THEN
		BEGIN
			RAISE NOTICE 'GPU Information:';
			-- Display GPU info by looping through results
			FOR rec IN SELECT device_id, device_name, total_memory_mb, free_memory_mb, 
					compute_capability_major, compute_capability_minor, is_available
			FROM neurondb_gpu_info()
			LOOP
				RAISE NOTICE '  Device %: % (Memory: %/% MB, Compute: %.%, Available: %)',
					rec.device_id, rec.device_name, rec.free_memory_mb, rec.total_memory_mb,
					rec.compute_capability_major, rec.compute_capability_minor, rec.is_available;
			END LOOP;
		EXCEPTION WHEN OTHERS THEN
			RAISE NOTICE 'GPU information not available';
		END;
	ELSE
		RAISE NOTICE 'CPU mode: GPU information skipped';
	END IF;
END $$;'''
    
    content = re.sub(gpu_info_pattern2, gpu_info_replacement2, content, flags=re.MULTILINE | re.DOTALL)
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

if __name__ == '__main__':
    fixed_count = 0
    for filename in ML_TEST_FILES:
        filepath = os.path.join(BASE_DIR, filename)
        if os.path.exists(filepath):
            if fix_file(filepath):
                print(f"Fixed: {filename}")
                fixed_count += 1
            else:
                print(f"No changes needed: {filename}")
        else:
            print(f"File not found: {filename}")
    
    print(f"\nFixed {fixed_count} files")

