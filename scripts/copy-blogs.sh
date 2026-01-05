#!/bin/bash

# Script to copy blogs from neurondb-www to neurondb2
# Extracts markdown content and copies assets

SOURCE_DIR="/Users/pgedge/pge/neurondb-www"
TARGET_DIR="/Users/pgedge/pge/neurondb2"
BLOG_SOURCE="$SOURCE_DIR/app/blog"
BLOG_TARGET="$TARGET_DIR/blog"
ASSETS_SOURCE="$SOURCE_DIR/public/blog"
ASSETS_TARGET="$TARGET_DIR/blog/assets"

echo "📝 Copying blogs from neurondb-www to neurondb2..."
echo ""

# Create target directories
mkdir -p "$BLOG_TARGET"
mkdir -p "$ASSETS_TARGET"

# Blog slugs from config
BLOGS=(
  "neurondb"
  "neurondb-semantic-search-guide"
  "neurondb-vectors"
  "neurondb-mcp-server"
  "rag-complete-guide"
  "rag-architectures-ai-builders-should-understand"
  "agentic-ai"
  "postgresql-vector-database"
  "ai-with-database-on-prem"
)

# Function to extract markdown from page.tsx
extract_markdown() {
  local page_file="$1"
  local output_file="$2"
  
  # Extract content between const markdown = ` and `; (handling multi-line)
  # This is a simplified extraction - may need adjustment based on actual format
  awk '
    /^const markdown = `$/ { 
      in_markdown=1
      next
    }
    in_markdown {
      if (/^`;?$/) {
        in_markdown=0
        next
      }
      print
    }
  ' "$page_file" > "$output_file"
  
  # Alternative: use sed to extract between markers
  # sed -n '/^const markdown = `$/,/^`;$/p' "$page_file" | sed '1d;$d' > "$output_file"
}

# Copy each blog
for blog in "${BLOGS[@]}"; do
  echo "📄 Processing: $blog"
  
  # Source files
  PAGE_FILE="$BLOG_SOURCE/$blog/page.tsx"
  MARKDOWN_FILE="$BLOG_TARGET/$blog.md"
  ASSET_DIR_SOURCE="$ASSETS_SOURCE/$blog"
  ASSET_DIR_TARGET="$ASSETS_TARGET/$blog"
  
  # Extract markdown if page exists
  if [ -f "$PAGE_FILE" ]; then
    echo "  ✓ Extracting markdown from $PAGE_FILE"
    extract_markdown "$PAGE_FILE" "$MARKDOWN_FILE"
    
    # Check if extraction worked (file should have content)
    if [ ! -s "$MARKDOWN_FILE" ]; then
      echo "  ⚠️  Warning: Markdown extraction may have failed for $blog"
      echo "  → Trying alternative extraction method..."
      
      # Alternative: extract between backticks more carefully
      grep -A 10000 "^const markdown = \`" "$PAGE_FILE" | \
        sed '1d' | \
        sed '/^\`;$/,$d' > "$MARKDOWN_FILE" || true
    fi
    
    # Count lines extracted
    LINES=$(wc -l < "$MARKDOWN_FILE" 2>/dev/null || echo "0")
    echo "  ✓ Extracted $LINES lines to $MARKDOWN_FILE"
  else
    echo "  ❌ Page file not found: $PAGE_FILE"
  fi
  
  # Copy assets
  if [ -d "$ASSET_DIR_SOURCE" ]; then
    echo "  📦 Copying assets from $ASSET_DIR_SOURCE"
    mkdir -p "$ASSET_DIR_TARGET"
    cp -r "$ASSET_DIR_SOURCE"/* "$ASSET_DIR_TARGET/" 2>/dev/null || true
    ASSET_COUNT=$(find "$ASSET_DIR_TARGET" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "  ✓ Copied $ASSET_COUNT asset files"
  else
    echo "  ⚠️  No assets directory found: $ASSET_DIR_SOURCE"
  fi
  
  echo ""
done

# Create blog index/README
echo "📋 Creating blog index..."
cat > "$BLOG_TARGET/README.md" << 'EOF'
# NeuronDB Blog Posts

This directory contains all blog posts in Markdown format.

## Blog Posts

EOF

for blog in "${BLOGS[@]}"; do
  if [ -f "$BLOG_TARGET/$blog.md" ]; then
    # Extract title from first line (usually # Title)
    TITLE=$(head -n 1 "$BLOG_TARGET/$blog.md" | sed 's/^# //' | sed 's/^!\[.*\]//' | xargs)
    if [ -z "$TITLE" ]; then
      TITLE="$blog"
    fi
    echo "- [$TITLE]($blog.md)" >> "$BLOG_TARGET/README.md"
  fi
done

cat >> "$BLOG_TARGET/README.md" << 'EOF'

## Assets

Blog assets (images, SVGs, etc.) are stored in the `assets/` subdirectory, organized by blog slug.

## Source

These blogs were extracted from the neurondb-www repository and converted to Markdown format for easier version control and editing.
EOF

echo "✅ Blog copying complete!"
echo ""
echo "Summary:"
echo "  - Blog markdown files: $BLOG_TARGET/"
echo "  - Blog assets: $ASSETS_TARGET/"
echo "  - Blog index: $BLOG_TARGET/README.md"
echo ""

