#!/usr/bin/env python3
"""
Convert HTML blog to markdown format
"""

import re
from pathlib import Path

blog_file = Path("/Users/pgedge/pge/neurondb2/blog/ai-with-database-on-prem.md")

with open(blog_file, 'r') as f:
    content = f.read()

# Convert HTML to Markdown
# <h1> -> # 
content = re.sub(r'<h1>(.*?)</h1>', r'# \1', content)
content = re.sub(r'<h2>(.*?)</h2>', r'## \1', content)
content = re.sub(r'<h3>(.*?)</h3>', r'### \1', content)
content = re.sub(r'<h4>(.*?)</h4>', r'#### \1', content)

# <p> -> paragraph
content = re.sub(r'<p>(.*?)</p>', r'\1\n', content)

# <img> -> ![alt](src)
content = re.sub(r'<img src="([^"]*)" alt="([^"]*)" />', r'![\2](\1)', content)

# <ul><li> -> - 
content = re.sub(r'<ul>', r'', content)
content = re.sub(r'</ul>', r'', content)
content = re.sub(r'<li>(.*?)</li>', r'- \1', content)
content = re.sub(r'  <li>', r'- ', content)

# <code> -> `code`
content = re.sub(r'<code>(.*?)</code>', r'`\1`', content)

# <pre><code> -> ```
content = re.sub(r'<pre><code>(.*?)</code></pre>', r'```\n\1\n```', content, flags=re.DOTALL)

# <strong> -> **bold**
content = re.sub(r'<strong>(.*?)</strong>', r'**\1**', content)

# <em> -> *italic*
content = re.sub(r'<em>(.*?)</em>', r'*\1*', content)

# Clean up extra newlines
content = re.sub(r'\n{3,}', '\n\n', content)

with open(blog_file, 'w') as f:
    f.write(content)

print(f"Converted {blog_file} to markdown format")

