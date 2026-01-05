#!/usr/bin/env python3
"""
Extract blog markdown content from neurondb-www React components
and copy to neurondb2/blog directory
"""

import os
import re
import shutil
from pathlib import Path

SOURCE_DIR = Path("/Users/pgedge/pge/neurondb-www")
TARGET_DIR = Path("/Users/pgedge/pge/neurondb2")

BLOG_SOURCE = SOURCE_DIR / "app" / "blog"
BLOG_TARGET = TARGET_DIR / "blog"
ASSETS_SOURCE = SOURCE_DIR / "public" / "blog"
ASSETS_TARGET = TARGET_DIR / "blog" / "assets"

# Blog slugs
BLOGS = [
    "neurondb",
    "neurondb-semantic-search-guide",
    "neurondb-vectors",
    "neurondb-mcp-server",
    "rag-complete-guide",
    "rag-architectures-ai-builders-should-understand",
    "agentic-ai",
    "postgresql-vector-database",
    "ai-with-database-on-prem",
]

def extract_markdown_from_tsx(file_path):
    """Extract markdown content from React component file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try regex first (more reliable for template literals)
        pattern = r'const\s+markdown\s*=\s*`(.*?)`;'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # Fallback: line-by-line extraction
        lines = content.split('\n')
        
        # Find the line with "const markdown = `"
        start_idx = None
        for i, line in enumerate(lines):
            if 'const markdown = `' in line or 'const markdown=`' in line or 'const markdown =`' in line:
                start_idx = i
                break
        
        if start_idx is None:
            return None
        
        # Find the closing `; (must be on its own line or end of line)
        end_idx = None
        for i in range(start_idx + 1, len(lines)):
            line = lines[i].strip()
            if line == '`;' or line == '`' or line.endswith('`;') or line.endswith('`'):
                end_idx = i
                break
        
        if end_idx is None:
            return None
        
        # Extract markdown content
        # First line might have the opening backtick, so we need to handle it
        first_line = lines[start_idx]
        if '`' in first_line:
            # Extract everything after the backtick
            markdown_start = first_line.split('`', 1)[1] if '`' in first_line else ''
        else:
            markdown_start = ''
        
        # Middle lines (full content)
        middle_lines = '\n'.join(lines[start_idx + 1:end_idx])
        
        # Last line might have the closing backtick, extract before it
        last_line = lines[end_idx] if end_idx < len(lines) else ''
        if '`' in last_line:
            markdown_end = last_line.split('`')[0]
        else:
            markdown_end = last_line
        
        markdown = markdown_start + ('\n' if markdown_start and middle_lines else '') + middle_lines + ('\n' if middle_lines and markdown_end else '') + markdown_end
        
        return markdown.strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    print("📝 Extracting blogs from neurondb-www to neurondb2...\n")
    
    # Create target directories
    BLOG_TARGET.mkdir(parents=True, exist_ok=True)
    ASSETS_TARGET.mkdir(parents=True, exist_ok=True)
    
    blog_list = []
    
    for blog in BLOGS:
        print(f"📄 Processing: {blog}")
        
        # Source files
        page_file = BLOG_SOURCE / blog / "page.tsx"
        markdown_file = BLOG_TARGET / f"{blog}.md"
        asset_dir_source = ASSETS_SOURCE / blog
        asset_dir_target = ASSETS_TARGET / blog
        
        # Extract markdown
        if page_file.exists():
            markdown_content = extract_markdown_from_tsx(page_file)
            
            if markdown_content:
                # Write markdown file
                with open(markdown_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                
                lines = len(markdown_content.split('\n'))
                print(f"  ✓ Extracted {lines} lines to {markdown_file.name}")
                
                # Extract title for index
                title_match = re.search(r'^#\s+(.+)$', markdown_content, re.MULTILINE)
                title = title_match.group(1).strip() if title_match else blog.replace('-', ' ').title()
                blog_list.append((blog, title))
            else:
                print(f"  ⚠️  Could not extract markdown from {page_file}")
                blog_list.append((blog, blog.replace('-', ' ').title()))
        else:
            print(f"  ❌ Page file not found: {page_file}")
            blog_list.append((blog, blog.replace('-', ' ').title()))
        
        # Copy assets
        if asset_dir_source.exists():
            if asset_dir_target.exists():
                shutil.rmtree(asset_dir_target)
            shutil.copytree(asset_dir_source, asset_dir_target)
            asset_count = len(list(asset_dir_target.rglob('*')))
            print(f"  📦 Copied {asset_count} asset files")
        else:
            print(f"  ⚠️  No assets directory: {asset_dir_source}")
        
        print()
    
    # Create README
    readme_content = """# NeuronDB Blog Posts

This directory contains all blog posts in Markdown format.

## Blog Posts

"""
    
    for blog, title in blog_list:
        readme_content += f"- [{title}]({blog}.md)\n"
    
    readme_content += """
## Assets

Blog assets (images, SVGs, etc.) are stored in the `assets/` subdirectory, organized by blog slug.

## Source

These blogs were extracted from the neurondb-www repository and converted to Markdown format for easier version control and editing.
"""
    
    readme_file = BLOG_TARGET / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("✅ Blog extraction complete!")
    print(f"\nSummary:")
    print(f"  - Blog markdown files: {BLOG_TARGET}/")
    print(f"  - Blog assets: {ASSETS_TARGET}/")
    print(f"  - Blog index: {readme_file}")
    print(f"\nTotal blogs: {len(blog_list)}")

if __name__ == "__main__":
    main()

