#!/bin/bash
#
# NeuronDB Blog Management Script
# Self-sufficient script for all blog maintenance operations
#
# Usage:
#   ./neurondb-blogs.sh COMMAND [OPTIONS]
#
# Commands:
#   fix-markdown      Fix broken markdown formatting
#   fix-image-paths   Fix image paths in blog files
#   fix-code-blocks   Fix code block formatting
#   copy              Copy blogs from neurondb-www
#   convert-html      Convert HTML blog to markdown
#   extract           Extract blog content from React components
#   svg-to-png        Convert SVG images to PNG

set -euo pipefail

#=========================================================================
# SELF-SUFFICIENT CONFIGURATION - NO EXTERNAL DEPENDENCIES
#=========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_NAME=$(basename "$0")

# Colors (inline - no external dependency)
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Default configuration
COMMAND=""
VERBOSE=false
DRY_RUN=false
BLOG_DIR="${PROJECT_ROOT}/blog"

#=========================================================================
# SELF-SUFFICIENT LOGGING FUNCTIONS
#=========================================================================

log_info() {
    echo -e "${CYAN}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

#=========================================================================
# HELP FUNCTION
#=========================================================================

show_help() {
    cat << EOF
${BOLD}NeuronDB Blog Management${NC}

${BOLD}Usage:${NC}
    ${SCRIPT_NAME} COMMAND [OPTIONS]

${BOLD}Commands:${NC}
    fix-markdown      Fix broken markdown formatting (escaped backticks)
    fix-image-paths   Fix image paths from /blog/... to assets/... format
    fix-code-blocks   Fix code block formatting
    copy              Copy blogs from neurondb-www repository
    convert-html      Convert HTML blog to markdown format
    extract           Extract blog content from React components
    svg-to-png        Convert SVG images to PNG format

${BOLD}Options:${NC}
    --blog-dir DIR    Blog directory (default: ./blog)
    --source DIR      Source directory for copy/extract commands
    --input FILE      Input file for convert-html
    --output DIR      Output directory
    --width WIDTH     PNG width for svg-to-png (default: 1200)
    --height HEIGHT   PNG height for svg-to-png (default: 800)
    --force           Force operation even if target exists
    --dry-run         Preview changes without applying
    -h, --help        Show this help message
    -v, --verbose     Enable verbose output
    -V, --version     Show version information

${BOLD}Examples:${NC}
    # Fix markdown formatting
    ${SCRIPT_NAME} fix-markdown

    # Fix image paths
    ${SCRIPT_NAME} fix-image-paths

    # Convert SVGs to PNGs
    ${SCRIPT_NAME} svg-to-png --width 1920 --height 1080

    # Copy blogs from neurondb-www
    ${SCRIPT_NAME} copy --source /path/to/neurondb-www

EOF
}

#=========================================================================
# COMMAND HANDLERS
#=========================================================================

fix_markdown_command() {
    shift
    
    log_info "Fixing markdown formatting..."
    
    if [[ ! -d "${BLOG_DIR}" ]]; then
        log_error "Blog directory not found: ${BLOG_DIR}"
        exit 1
    fi
    
    cd "${BLOG_DIR}" || exit 1
    
    local fixed_count=0
    for file in *.md; do
        if [[ ! -f "$file" ]]; then
            continue
        fi
        
        if [[ "${DRY_RUN}" == "true" ]]; then
            if grep -q '\\`' "$file" 2>/dev/null; then
                log_info "[DRY RUN] Would fix: $file"
                ((fixed_count++))
            fi
        else
            [[ "$VERBOSE" == "true" ]] && log_info "Processing: $file"
            
            # Determine sed command based on OS
            if [[ "$OSTYPE" == "darwin"* ]]; then
                sed -i '' 's/\\`\\`\\`/```/g' "$file" 2>/dev/null || true
                sed -i '' 's/\\`/`/g' "$file" 2>/dev/null || true
            else
                sed -i 's/\\`\\`\\`/```/g' "$file" 2>/dev/null || true
                sed -i 's/\\`/`/g' "$file" 2>/dev/null || true
            fi
            ((fixed_count++))
        fi
    done
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would fix $fixed_count file(s)"
    else
        log_success "Fixed markdown in $fixed_count file(s)"
    fi
}

fix_image_paths_command() {
    shift
    
    log_info "Fixing image paths..."
    
    if [[ ! -d "${BLOG_DIR}" ]]; then
        log_error "Blog directory not found: ${BLOG_DIR}"
        exit 1
    fi
    
    local fixed_count=0
    for blog_file in "${BLOG_DIR}"/*.md; do
        if [[ ! -f "$blog_file" ]]; then
            continue
        fi
        
        local blog_slug=$(basename "$blog_file" .md)
        
        if [[ "${DRY_RUN}" == "true" ]]; then
            if grep -q "/blog/${blog_slug}/" "$blog_file" 2>/dev/null; then
                log_info "[DRY RUN] Would fix: $blog_slug"
                ((fixed_count++))
            fi
        else
            [[ "$VERBOSE" == "true" ]] && log_info "Processing: $blog_slug"
            
            # Determine sed command based on OS
            if [[ "$OSTYPE" == "darwin"* ]]; then
                sed -i '' "s|/blog/$blog_slug/|assets/$blog_slug/|g" "$blog_file" 2>/dev/null || true
                sed -i '' "s|/blog/$blog_slug/\([^?]*\)?v=[0-9]*|assets/$blog_slug/\1|g" "$blog_file" 2>/dev/null || true
            else
                sed -i "s|/blog/$blog_slug/|assets/$blog_slug/|g" "$blog_file" 2>/dev/null || true
                sed -i "s|/blog/$blog_slug/\([^?]*\)?v=[0-9]*|assets/$blog_slug/\1|g" "$blog_file" 2>/dev/null || true
            fi
            ((fixed_count++))
        fi
    done
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would fix $fixed_count file(s)"
    else
        log_success "Fixed image paths in $fixed_count file(s)"
    fi
}

fix_code_blocks_command() {
    shift
    
    log_info "Fixing code block formatting..."
    
    if [[ ! -d "${BLOG_DIR}" ]]; then
        log_error "Blog directory not found: ${BLOG_DIR}"
        exit 1
    fi
    
    # Inline Python script (self-sufficient)
    local python_script='
import re
import sys
from pathlib import Path

blog_dir = Path("'"${BLOG_DIR}"'")

def fix_code_blocks(content):
    pattern = r"```([a-zA-Z]+)\s+(.+?)```"
    def replace_func(match):
        language = match.group(1)
        code = match.group(2)
        return f"```{language}\n{code}\n```"
    return re.sub(pattern, replace_func, content, flags=re.DOTALL)

for md_file in blog_dir.glob("*.md"):
    if md_file.name == "README.md":
        continue
    
    try:
        with open(md_file, "r", encoding="utf-8") as f:
            original = f.read()
        fixed = fix_code_blocks(original)
        if fixed != original:
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(fixed)
            print(f"Fixed: {md_file.name}")
    except Exception as e:
        print(f"Error processing {md_file.name}: {e}", file=sys.stderr)
'
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would fix code blocks"
    else
        cd "${PROJECT_ROOT}" && python3 -c "$python_script"
        log_success "Code blocks fixed"
    fi
}

copy_command() {
    shift
    
    local source_dir="${1:-/Users/pgedge/pge/neurondb-www}"
    
    log_info "Copying blogs from neurondb-www..."
    
    if [[ ! -d "$source_dir" ]]; then
        log_error "Source directory not found: $source_dir"
        exit 1
    fi
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would copy blogs from $source_dir"
        return 0
    fi
    
    # Inline copy logic (self-sufficient)
    local blog_source="$source_dir/app/blog"
    local blog_target="${BLOG_DIR}"
    local assets_source="$source_dir/public/blog"
    local assets_target="${BLOG_DIR}/assets"
    
    mkdir -p "$blog_target"
    mkdir -p "$assets_target"
    
    # Blog slugs
    local blogs=(
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
    
    for blog in "${blogs[@]}"; do
        local page_file="$blog_source/$blog/page.tsx"
        local markdown_file="$blog_target/$blog.md"
        local asset_dir_source="$assets_source/$blog"
        local asset_dir_target="$assets_target/$blog"
        
        if [[ -f "$page_file" ]]; then
            log_info "Extracting: $blog"
            # Extract markdown from React component
            awk '/^const markdown = `$/,/^`;?$/' "$page_file" | sed '1d;$d' > "$markdown_file" 2>/dev/null || true
        fi
        
        if [[ -d "$asset_dir_source" ]]; then
            log_info "Copying assets: $blog"
            mkdir -p "$asset_dir_target"
            cp -r "$asset_dir_source"/* "$asset_dir_target/" 2>/dev/null || true
        fi
    done
    
    log_success "Blogs copied"
}

convert_html_command() {
    shift
    
    local input_file=""
    local output_dir="${BLOG_DIR}"
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --input)
                input_file="$2"
                shift 2
                ;;
            --output)
                output_dir="$2"
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done
    
    if [[ -z "$input_file" ]]; then
        log_error "Input file required (--input)"
        exit 1
    fi
    
    if [[ ! -f "$input_file" ]]; then
        log_error "Input file not found: $input_file"
        exit 1
    fi
    
    log_info "Converting HTML to markdown..."
    
    # Inline Python conversion (self-sufficient)
    local python_script="
import re
import sys

with open('$input_file', 'r') as f:
    content = f.read()

# Convert HTML to Markdown
content = re.sub(r'<h1>(.*?)</h1>', r'# \1', content)
content = re.sub(r'<h2>(.*?)</h2>', r'## \1', content)
content = re.sub(r'<h3>(.*?)</h3>', r'### \1', content)
content = re.sub(r'<p>(.*?)</p>', r'\1\n', content)
content = re.sub(r'<img src=\"([^\"]*)\" alt=\"([^\"]*)\" />', r'![\2](\1)', content)
content = re.sub(r'<ul>', r'', content)
content = re.sub(r'</ul>', r'', content)
content = re.sub(r'<li>(.*?)</li>', r'- \1', content)
content = re.sub(r'<code>(.*?)</code>', r'\`\1\`', content)
content = re.sub(r'<pre><code>(.*?)</code></pre>', r'\`\`\`\n\1\n\`\`\`', content, flags=re.DOTALL)
content = re.sub(r'<strong>(.*?)</strong>', r'**\1**', content)
content = re.sub(r'<em>(.*?)</em>', r'*\1*', content)
content = re.sub(r'\n{3,}', '\n\n', content)

output_file = '$input_file'.replace('.html', '.md')
with open(output_file, 'w') as f:
    f.write(content)
print(f'Converted: {output_file}')
"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would convert $input_file"
    else
        python3 -c "$python_script"
        log_success "HTML converted to markdown"
    fi
}

extract_command() {
    shift
    
    local source_dir="${1:-/Users/pgedge/pge/neurondb-www}"
    
    log_info "Extracting blog content from React components..."
    
    if [[ ! -d "$source_dir" ]]; then
        log_error "Source directory not found: $source_dir"
        exit 1
    fi
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would extract from $source_dir"
        return 0
    fi
    
    # Inline Python extraction (self-sufficient)
    local python_script="
import os
import re
import shutil
from pathlib import Path

source_dir = Path('$source_dir')
target_dir = Path('${PROJECT_ROOT}/blog')
assets_source = source_dir / 'public' / 'blog'
assets_target = target_dir / 'assets'

target_dir.mkdir(exist_ok=True)
assets_target.mkdir(exist_ok=True)

blogs = [
    'neurondb', 'neurondb-semantic-search-guide', 'neurondb-vectors',
    'neurondb-mcp-server', 'rag-complete-guide',
    'rag-architectures-ai-builders-should-understand', 'agentic-ai',
    'postgresql-vector-database', 'ai-with-database-on-prem'
]

for blog in blogs:
    page_file = source_dir / 'app' / 'blog' / blog / 'page.tsx'
    if page_file.exists():
        with open(page_file, 'r', encoding='utf-8') as f:
            content = f.read()
        pattern = r'const\s+markdown\s*=\s*\`(.*?)\`;'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            markdown = match.group(1).strip()
            output_file = target_dir / f'{blog}.md'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown)
            print(f'Extracted: {blog}')
    
    asset_dir = assets_source / blog
    if asset_dir.exists():
        shutil.copytree(asset_dir, assets_target / blog, dirs_exist_ok=True)
        print(f'Copied assets: {blog}')
"
    
    python3 -c "$python_script"
    log_success "Blogs extracted"
}

svg_to_png_command() {
    shift
    
    local width=1200
    local height=800
    local force=false
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --width)
                width="$2"
                shift 2
                ;;
            --height)
                height="$2"
                shift 2
                ;;
            --force)
                force=true
                shift
                ;;
            *)
                break
                ;;
        esac
    done
    
    log_info "Converting SVGs to PNGs..."
    
    # Check for converter
    local converter=""
    if command -v rsvg-convert &> /dev/null; then
        converter="rsvg-convert"
    elif command -v inkscape &> /dev/null; then
        converter="inkscape"
    elif command -v convert &> /dev/null; then
        converter="imagemagick"
    else
        log_error "No SVG converter found. Install rsvg-convert, inkscape, or ImageMagick"
        log_info "On macOS: brew install librsvg"
        exit 1
    fi
    
    local assets_dir="${BLOG_DIR}/assets"
    if [[ ! -d "$assets_dir" ]]; then
        log_error "Assets directory not found: $assets_dir"
        exit 1
    fi
    
    local converted=0
    local skipped=0
    
    while IFS= read -r -d '' svg_file; do
        png_file="${svg_file%.svg}.png"
        
        if [[ "$force" == "true" ]] || [[ ! -f "$png_file" ]] || [[ "$svg_file" -nt "$png_file" ]]; then
            [[ "$VERBOSE" == "true" ]] && log_info "Converting: $(basename "$svg_file")"
            
            case $converter in
                rsvg-convert)
                    rsvg-convert -w "$width" -h "$height" "$svg_file" > "$png_file" 2>/dev/null
                    ;;
                inkscape)
                    inkscape "$svg_file" --export-filename="$png_file" --export-width="$width" --export-height="$height" 2>/dev/null
                    ;;
                imagemagick)
                    convert -background none -resize "${width}x${height}" "$svg_file" "$png_file" 2>/dev/null
                    ;;
            esac
            
            [[ $? -eq 0 ]] && ((converted++)) || log_warning "Failed: $(basename "$svg_file")"
        else
            ((skipped++))
        fi
    done < <(find "$assets_dir" -name "*.svg" -type f -print0 2>/dev/null)
    
    log_success "Conversion complete: $converted converted, $skipped skipped"
}

#=========================================================================
# ARGUMENT PARSING
#=========================================================================

parse_arguments() {
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi
    
    COMMAND="$1"
    shift
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --blog-dir)
                BLOG_DIR="$2"
                shift 2
                ;;
            --source)
                # Handled by commands
                shift 2
                ;;
            --input)
                # Handled by commands
                shift 2
                ;;
            --output)
                # Handled by commands
                shift 2
                ;;
            --width)
                # Handled by commands
                shift 2
                ;;
            --height)
                # Handled by commands
                shift 2
                ;;
            --force)
                # Handled by commands
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -V|--version)
                echo "${SCRIPT_NAME} version 2.0.0"
                exit 0
                ;;
            *)
                # Remaining arguments passed to command
                break
                ;;
        esac
    done
}

#=========================================================================
# MAIN FUNCTION
#=========================================================================

main() {
    parse_arguments "$@"
    
    case "${COMMAND}" in
        fix-markdown)
            fix_markdown_command "$@"
            ;;
        fix-image-paths)
            fix_image_paths_command "$@"
            ;;
        fix-code-blocks)
            fix_code_blocks_command "$@"
            ;;
        copy)
            copy_command "$@"
            ;;
        convert-html)
            convert_html_command "$@"
            ;;
        extract)
            extract_command "$@"
            ;;
        svg-to-png)
            svg_to_png_command "$@"
            ;;
        *)
            log_error "Unknown command: ${COMMAND}"
            show_help
            exit 1
            ;;
    esac
}

main "$@"

