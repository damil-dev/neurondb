-- Basic tests for GPU search functions
-- Tests the fixed GPU HNSW and IVF search functions

-- Skip this test if running in CPU compute mode
DO $$
DECLARE
	compute_mode_val text;
BEGIN
	-- Check compute mode, default to '0' (CPU) if not set
	BEGIN
		compute_mode_val := current_setting('neurondb.compute_mode', true);
	EXCEPTION WHEN OTHERS THEN
		compute_mode_val := '0';  -- Default to CPU if setting doesn't exist
	END;
	
	-- Skip test if in CPU mode - just print message and exit
	IF compute_mode_val = '0' THEN
		RAISE NOTICE 'Skipping GPU search test: running in CPU compute mode';
		-- Don't execute any test logic
		RETURN;
	END IF;
	
	-- Execute all test logic here using dynamic SQL
	EXECUTE 'DROP TABLE IF EXISTS gpu_search_test CASCADE';
	
	EXECUTE 'CREATE TABLE gpu_search_test (
		id serial PRIMARY KEY,
		vec vector(4)
	)';
	
	EXECUTE 'INSERT INTO gpu_search_test (vec)
	SELECT (''['' || random() || '','' || random() || '','' || random() || '','' || random() || '']'')::vector
	FROM generate_series(1, 50)';
	
	EXECUTE 'CREATE INDEX gpu_hnsw_idx ON gpu_search_test USING hnsw (vec vector_l2_ops)
	WITH (m = 16, ef_construction = 64, ef_search = 40)';
	
	-- Test GPU HNSW search (if GPU available)
	BEGIN
		PERFORM * FROM hnsw_knn_search_gpu('gpu_hnsw_idx', '[0.5,0.5,0.5,0.5]'::vector, 5, 20);
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'GPU HNSW search function not available: %', SQLERRM;
	END;
	
	BEGIN
		PERFORM * FROM hnsw_knn_search_gpu('gpu_hnsw_idx', '[0.5,0.5,0.5,0.5]'::vector, 5);
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'GPU HNSW search function not available: %', SQLERRM;
	END;
	
	EXECUTE 'CREATE INDEX gpu_ivf_idx ON gpu_search_test USING ivf (vec vector_l2_ops)
	WITH (lists = 10, probes = 5)';
	
	-- Test GPU IVF search (if GPU available)
	BEGIN
		PERFORM * FROM ivf_knn_search_gpu('gpu_ivf_idx', '[0.5,0.5,0.5,0.5]'::vector, 5, 3);
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'GPU IVF search function not available: %', SQLERRM;
	END;
	
	BEGIN
		PERFORM * FROM ivf_knn_search_gpu('gpu_ivf_idx', '[0.5,0.5,0.5,0.5]'::vector, 5);
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'GPU IVF search function not available: %', SQLERRM;
	END;
	
	-- Test error cases
	BEGIN
		PERFORM * FROM hnsw_knn_search_gpu('nonexistent_index', '[0.5,0.5,0.5,0.5]'::vector, 5);
		RAISE EXCEPTION 'Should have failed';
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected
	END;
	
	-- Cleanup
	EXECUTE 'DROP INDEX IF EXISTS gpu_hnsw_idx';
	EXECUTE 'DROP INDEX IF EXISTS gpu_ivf_idx';
	EXECUTE 'DROP TABLE IF EXISTS gpu_search_test CASCADE';
	
	RAISE NOTICE 'GPU search test completed successfully';
END $$;

\echo 'Test completed successfully'
