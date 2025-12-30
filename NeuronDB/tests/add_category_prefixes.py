#!/usr/bin/env python3
"""
Add category prefixes to test filenames:
- Core tests: ???_core_...
- Vector tests: ???_vector_...
- ML tests: ???_ml_...
- RAG tests: ???_rag_...
- GPU tests: ???_gpu_...
- Other tests: ???_other_...
- Crash prevention tests: ???_crash_...
"""

import os
import re
from pathlib import Path

BASIC_DIR = Path(__file__).parent / "sql" / "basic"

# Define the mapping: (start_num, end_num, category_prefix)
CATEGORY_MAPPING = [
    (1, 11, "core"),      # Core tests: 001-011
    (12, 34, "vector"),   # Vector tests: 012-034
    (35, 58, "ml"),       # ML tests: 035-058
    (59, 61, "rag"),      # RAG tests: 059-061
    (62, 64, "gpu"),      # GPU tests: 062-064
    (65, 67, "other"),    # Other tests: 065-067
    (68, 72, "crash"),    # Crash prevention tests: 068-072
    (73, 77, "vector"),   # Existing vector tests: 073-077
]

def get_category_prefix(num):
    """Get the category prefix for a given file number"""
    for start, end, prefix in CATEGORY_MAPPING:
        if start <= num <= end:
            return prefix
    return None

def rename_file(old_path, new_path):
    """Rename a file"""
    try:
        old_path.rename(new_path)
        return True
    except Exception as e:
        print(f"ERROR renaming {old_path.name} to {new_path.name}: {e}")
        return False

def main():
    """Main renaming function"""
    print("Adding category prefixes to test files...")
    print(f"Directory: {BASIC_DIR}\n")
    
    # Get all SQL files
    sql_files = sorted(BASIC_DIR.glob("*.sql"))
    print(f"Found {len(sql_files)} SQL files\n")
    
    renamed = 0
    skipped = 0
    
    for file_path in sql_files:
        filename = file_path.name
        # Extract number and rest of name
        match = re.match(r'^(\d{3})_(.+)$', filename)
        if not match:
            print(f"SKIP: {filename} (doesn't match pattern)")
            skipped += 1
            continue
        
        num = int(match.group(1))
        rest = match.group(2)
        
        # Get category prefix
        category = get_category_prefix(num)
        if not category:
            print(f"SKIP: {filename} (no category mapping for {num:03d})")
            skipped += 1
            continue
        
        # Check if already has the prefix
        if rest.startswith(f"{category}_"):
            print(f"SKIP: {filename} (already has {category}_ prefix)")
            skipped += 1
            continue
        
        # Create new filename
        new_filename = f"{num:03d}_{category}_{rest}"
        new_path = BASIC_DIR / new_filename
        
        # Check if target already exists
        if new_path.exists():
            print(f"SKIP: {filename} -> {new_filename} (target exists)")
            skipped += 1
            continue
        
        # Rename
        if rename_file(file_path, new_path):
            print(f"  [{num:03d}] {filename} -> {new_filename}")
            renamed += 1
    
    print(f"\nRenamed: {renamed} files")
    print(f"Skipped: {skipped} files")
    print("Done!")

if __name__ == "__main__":
    main()





