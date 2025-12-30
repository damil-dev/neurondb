#!/usr/bin/env python3
"""
Renumber and flatten test files from subdirectories into tests/sql/basic/
Removes "pgvector" from filenames and ensures sequential numbering 001-077
"""

import os
import re
import shutil
from pathlib import Path

# Base directory
TESTS_SQL_DIR = Path(__file__).parent / "sql"
BASIC_DIR = TESTS_SQL_DIR / "basic"
CRASH_PREVENTION_DIR = TESTS_SQL_DIR / "crash_prevention"

# Define the order of directories and files
FILE_ORDER = [
    # Core tests (11 files)
    ("basic/core", [
        "001_index.sql",
        "002_ivf_index.sql",
        "003_core.sql",
        "004_worker.sql",
        "005_storage.sql",
        "006_scan.sql",
        "007_util.sql",
        "008_planner.sql",
        "009_tenant.sql",
        "010_types.sql",
        "011_metrics.sql",
    ]),
    # Vector tests (23 files)
    ("basic/vector", [
        "001_vector_ops.sql",
        "002_vector.sql",
        "003_sparse_vectors.sql",
        "004_quantization_fp8.sql",
        "005_embeddings_text.sql",
        "006_embeddings_batch.sql",
        "007_embeddings_config.sql",
        "008_embeddings_hf_models.sql",
        "009_embeddings_multimodal.sql",
        "010_pgvector_vector_type.sql",
        "011_pgvector_cast.sql",
        "012_pgvector_hnsw_vector.sql",
        "013_pgvector_ivfflat_vector.sql",
        "014_pgvector_hnsw_halfvec.sql",
        "015_pgvector_hnsw_sparsevec.sql",
        "016_pgvector_hnsw_bit.sql",
        "017_pgvector_ivfflat_halfvec.sql",
        "018_pgvector_ivfflat_bit.sql",
        "019_pgvector_halfvec.sql",
        "020_pgvector_sparsevec.sql",
        "021_pgvector_bit.sql",
        "022_pgvector_copy.sql",
        "023_pgvector_btree.sql",
    ]),
    # ML tests (24 files)
    ("basic/ml", [
        "001_linreg.sql",
        "002_logreg.sql",
        "003_rf.sql",
        "004_svm.sql",
        "005_dt.sql",
        "006_ridge.sql",
        "007_lasso.sql",
        "008_nb.sql",
        "009_knn.sql",
        "010_xgboost.sql",
        "011_catboost.sql",
        "012_lightgbm.sql",
        "013_neural_network.sql",
        "014_gmm.sql",
        "015_kmeans.sql",
        "016_minibatch_kmeans.sql",
        "017_hierarchical.sql",
        "018_dbscan.sql",
        "019_pca.sql",
        "020_timeseries.sql",
        "021_automl.sql",
        "022_automl_standalone.sql",
        "023_recommender.sql",
        "024_arima.sql",
    ]),
    # RAG tests (3 files)
    ("basic/rag", [
        "001_rag.sql",
        "002_hybrid_search.sql",
        "003_reranking_flash.sql",
    ]),
    # GPU tests (3 files)
    ("basic/gpu", [
        "001_gpu_info.sql",
        "002_gpu_search.sql",
        "003_onnx.sql",
    ]),
    # Other tests (3 files)
    ("basic/other", [
        "001_crash_prevention.sql",
        "002_multimodal.sql",
        "003_opq_pq.sql",
    ]),
    # Crash prevention tests (5 files)
    ("crash_prevention", [
        "001_null_parameters.sql",
        "002_invalid_models.sql",
        "003_spi_failures.sql",
        "004_memory_contexts.sql",
        "005_array_bounds.sql",
    ]),
    # Existing basic/ root files (5 files)
    ("basic", [
        "061_pgvector_ivfflat_vector.sql",
        "062_pgvector_hnsw_halfvec.sql",
        "063_pgvector_hnsw_sparsevec.sql",
        "067_pgvector_halfvec.sql",
        "068_pgvector_sparsevec.sql",
    ]),
]


def remove_pgvector_from_name(filename):
    """Remove 'pgvector' from filename, preserving the rest"""
    # Remove the numeric prefix
    match = re.match(r'^(\d{3})_(.+)\.sql$', filename)
    if match:
        num = match.group(1)
        rest = match.group(2)
        # Remove 'pgvector' (case insensitive) and any surrounding underscores
        rest = re.sub(r'_?pgvector_?', '_', rest, flags=re.IGNORECASE)
        # Clean up multiple underscores
        rest = re.sub(r'_+', '_', rest)
        # Remove leading/trailing underscores
        rest = rest.strip('_')
        return f"{num}_{rest}.sql"
    return filename


def main():
    """Main renumbering function"""
    print("Starting test file renumbering and flattening...")
    print(f"Target directory: {BASIC_DIR}")
    
    # Create a backup/mapping of all operations
    operations = []
    new_number = 1
    
    # Process files in order
    for dir_path, files in FILE_ORDER:
        source_dir = TESTS_SQL_DIR / dir_path
        
        for old_filename in files:
            source_file = source_dir / old_filename
            
            if not source_file.exists():
                print(f"WARNING: File not found: {source_file}")
                continue
            
            # Extract descriptive name (remove number prefix)
            match = re.match(r'^\d{3}_(.+)\.sql$', old_filename)
            if match:
                desc_name = match.group(1)
            else:
                desc_name = old_filename.replace('.sql', '')
            
            # Remove "pgvector" from descriptive name
            desc_name = re.sub(r'_?pgvector_?', '_', desc_name, flags=re.IGNORECASE)
            desc_name = re.sub(r'_+', '_', desc_name)
            desc_name = desc_name.strip('_')
            
            # Create new filename with sequential number
            new_filename = f"{new_number:03d}_{desc_name}.sql"
            target_file = BASIC_DIR / new_filename
            
            operations.append({
                'source': source_file,
                'target': target_file,
                'old_name': old_filename,
                'new_name': new_filename,
                'number': new_number
            })
            
            new_number += 1
    
    print(f"\nTotal files to process: {len(operations)}")
    print(f"Expected final count: {new_number - 1}")
    
    # Check for conflicts
    conflicts = []
    for op in operations:
        if op['target'].exists() and op['source'] != op['target']:
            conflicts.append(op['target'])
    
    if conflicts:
        print(f"\nWARNING: {len(conflicts)} target files already exist:")
        for cf in conflicts[:10]:
            print(f"  - {cf}")
        if len(conflicts) > 10:
            print(f"  ... and {len(conflicts) - 10} more")
        response = input("\nContinue anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
    
    # Perform the moves/renames
    print("\nMoving and renaming files...")
    moved = 0
    for op in operations:
        try:
            # If source and target are the same, just rename if needed
            if op['source'] == op['target']:
                if op['source'].name != op['new_name']:
                    # Just rename in place
                    new_path = op['source'].parent / op['new_name']
                    op['source'].rename(new_path)
                    print(f"  Renamed: {op['source'].name} -> {op['new_name']}")
                    moved += 1
            else:
                # Move and rename
                shutil.move(str(op['source']), str(op['target']))
                print(f"  [{op['number']:03d}] Moved: {op['source']} -> {op['target'].name}")
                moved += 1
        except Exception as e:
            print(f"  ERROR moving {op['source']}: {e}")
    
    print(f"\nSuccessfully processed {moved} files.")
    
    # Clean up empty directories
    print("\nCleaning up empty directories...")
    dirs_to_remove = [
        BASIC_DIR / "core",
        BASIC_DIR / "vector",
        BASIC_DIR / "ml",
        BASIC_DIR / "rag",
        BASIC_DIR / "gpu",
        BASIC_DIR / "other",
        CRASH_PREVENTION_DIR,
    ]
    
    for dir_path in dirs_to_remove:
        if dir_path.exists() and dir_path.is_dir():
            try:
                # Check if directory is empty
                if not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    print(f"  Removed empty directory: {dir_path}")
                else:
                    remaining = list(dir_path.iterdir())
                    print(f"  WARNING: Directory not empty: {dir_path} ({len(remaining)} items)")
            except Exception as e:
                print(f"  ERROR removing {dir_path}: {e}")
    
    print("\nRenumbering complete!")
    print(f"All test files should now be in {BASIC_DIR} with sequential numbering 001-{new_number-1:03d}")


if __name__ == "__main__":
    main()





