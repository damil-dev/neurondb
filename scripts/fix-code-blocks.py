#!/usr/bin/env python3
"""
Fix code block formatting in blog markdown files
Code blocks should have language identifier on the same line as opening backticks,
but the code content should start on the next line.
"""

import re
from pathlib import Path

blog_dir = Path("/Users/pgedge/pge/neurondb2/blog")

def fix_code_blocks(content):
    """Fix code blocks where code starts on the same line as opening backticks"""
    
    # Pattern: ```language CODE_HERE
    # Should be: ```language\nCODE_HERE
    
    # Fix when there's code on the same line as opening backticks
    # Match ```language followed by non-newline content
    pattern = r'```([a-zA-Z]+)\s+(.+?)```'
    
    def replace_func(match):
        language = match.group(1)
        code = match.group(2)
        # Add newline after language identifier
        return f'```{language}\n{code}\n```'
    
    # Handle multiline code blocks
    content = re.sub(pattern, replace_func, content, flags=re.DOTALL)
    
    return content

# Process each markdown file
for md_file in blog_dir.glob("*.md"):
    if md_file.name == "README.md":
        continue
    
    print(f"Checking: {md_file.name}")
    
    with open(md_file, 'r', encoding='utf-8') as f:
        original = f.read()
    
    fixed = fix_code_blocks(original)
    
    if fixed != original:
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(fixed)
        print(f"  ✓ Fixed code blocks in {md_file.name}")
    else:
        print(f"  - No issues in {md_file.name}")

print("\nDone fixing code blocks.")

