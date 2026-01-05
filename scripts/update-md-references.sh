#!/bin/bash

# Script to update references to renamed .md files

cd /Users/pgedge/pge/neurondb2 || exit 1

# Get list of renamed files from git
git status --short | grep -E "^R" | while read -r status oldfile newfile; do
  oldname=$(basename "$oldfile")
  newname=$(basename "$newfile")
  
  if [ "$oldname" != "$newname" ]; then
    echo "Updating references: $oldname -> $newname"
    
    # Find and replace in all files
    find . -type f \( -name "*.md" -o -name "*.txt" -o -name "*.sh" -o -name "*.go" -o -name "*.py" -o -name "*.ts" -o -name "*.tsx" -o -name "*.js" -o -name "*.json" \) \
      ! -path "./.git/*" \
      -exec sed -i '' "s|$oldname|$newname|g" {} \; 2>/dev/null || \
      find . -type f \( -name "*.md" -o -name "*.txt" -o -name "*.sh" -o -name "*.go" -o -name "*.py" -o -name "*.ts" -o -name "*.tsx" -o -name "*.js" -o -name "*.json" \) \
        ! -path "./.git/*" \
        -exec sed -i "s|$oldname|$newname|g" {} \;
  fi
done

echo "Done updating references."

