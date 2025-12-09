#!/bin/bash
# Script to fix all ML test files to match the pattern from 001_linreg_basic.sql

cd "$(dirname "$0")/sql/basic" || exit 1

FILES=(
    "006_ridge_basic.sql"
    "007_lasso_basic.sql"
    "008_nb_basic.sql"
    "009_knn_basic.sql"
    "010_xgboost_basic.sql"
    "011_catboost_basic.sql"
    "012_lightgbm_basic.sql"
    "014_gmm_basic.sql"
    "015_kmeans_basic.sql"
    "016_minibatch_kmeans_basic.sql"
    "017_hierarchical_basic.sql"
    "018_dbscan_basic.sql"
    "019_pca_basic.sql"
    "021_automl_basic.sql"
    "022_automl_standalone_basic.sql"
)

for file in "${FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "File not found: $file"
        continue
    fi
    
    echo "Processing $file..."
    
    # 1. Add SET client_min_messages TO WARNING; if not present
    if ! grep -q "SET client_min_messages TO WARNING;" "$file"; then
        sed -i '/^\\set ON_ERROR_STOP on/i SET client_min_messages TO WARNING;' "$file"
    fi
    
    # 2. Replace gpu_mode with compute_mode in variable declarations
    sed -i 's/gpu_mode TEXT;/compute_mode TEXT;/g' "$file"
    sed -i "s/setting_key = 'gpu_mode'/setting_key = 'compute_mode'/g" "$file"
    sed -i 's/INTO gpu_mode/INTO compute_mode/g' "$file"
    sed -i 's/IF gpu_mode =/IF compute_mode =/g' "$file"
    sed -i 's/ELSIF gpu_mode =/ELSIF compute_mode =/g' "$file"
    
    # 3. Fix GPU configuration block
    sed -i '/SELECT neurondb_gpu_enable();/c\		PERFORM neurondb_gpu_enable();' "$file"
    
    # 4. Fix GPU config block structure (add ELSIF for auto mode)
    # This is complex, so we'll handle it with a more specific pattern
    sed -i '/IF compute_mode = '\''gpu'\'' THEN/,/END IF;/ {
        /IF compute_mode = '\''gpu'\'' THEN/a\
	ELSIF compute_mode = '\''auto'\'' THEN\
		PERFORM neurondb_gpu_enable();
    }' "$file"
    
    # Note: Complex replacements like evaluation section and GPU info display
    # need to be done manually or with more sophisticated tools
    # For now, the basic replacements are done
    
    echo "  Basic fixes applied to $file"
done

echo "Done! Note: Complex sections (evaluation, GPU info display) may need manual fixes."

