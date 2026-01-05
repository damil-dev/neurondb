#!/bin/bash

# Fix broken markdown formatting in blog files

cd /Users/pgedge/pge/neurondb2/blog || exit 1

echo "Fixing markdown formatting..."

# Fix escaped backticks in code blocks
for file in *.md; do
  if [ -f "$file" ]; then
    echo "Processing: $file"
    
    # Replace \`\`\` with ```
    sed -i '' 's/\\`\\`\\`/```/g' "$file"
    
    # Fix any remaining escaped backticks
    sed -i '' 's/\\`/`/g' "$file"
  fi
done

echo "Done fixing markdown."

