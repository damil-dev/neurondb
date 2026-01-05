#!/bin/bash

# Script to rename all .md files to lowercase except README.md files

cd /Users/pgedge/pge/neurondb2 || exit 1

# Find all .md files except README.md and rename them to lowercase
find . -name "*.md" -not -name "README.md" -type f | while read -r file; do
  dir=$(dirname "$file")
  basename=$(basename "$file")
  lowercase=$(echo "$basename" | tr '[:upper:]' '[:lower:]')
  
  if [ "$basename" != "$lowercase" ]; then
    newfile="$dir/$lowercase"
    echo "Renaming: $file -> $newfile"
    git mv "$file" "$newfile" 2>/dev/null || mv "$file" "$newfile"
  fi
done

echo "Done renaming files."

