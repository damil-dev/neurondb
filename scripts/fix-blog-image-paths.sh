#!/bin/bash

# Script to fix image paths in blog markdown files
# Updates paths from /blog/... to assets/... format

cd /Users/pgedge/pge/neurondb2 || exit 1

for blog_file in blog/*.md; do
  if [ -f "$blog_file" ]; then
    blog_slug=$(basename "$blog_file" .md)
    echo "Processing: $blog_slug"
    
    # Update image paths from /blog/slug/ to assets/slug/
    # Pattern: /blog/slug/image.svg -> assets/slug/image.svg
    sed -i '' "s|/blog/$blog_slug/|assets/$blog_slug/|g" "$blog_file"
    
    # Also handle paths that might have query strings
    sed -i '' "s|/blog/$blog_slug/\([^?]*\)?v=[0-9]*|assets/$blog_slug/\1|g" "$blog_file"
    
    # Handle any remaining /blog/ references
    sed -i '' "s|/blog/$blog_slug/|assets/$blog_slug/|g" "$blog_file"
  fi
done

echo "Done fixing image paths."

