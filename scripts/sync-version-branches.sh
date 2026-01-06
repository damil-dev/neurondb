#!/bin/bash
# Sync REL1_STABLE and main branches with version differences
# REL1_STABLE: 1.0.0, main: 2.0.0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Syncing REL1_STABLE and main branches with version differences${NC}"

# Check if we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo -e "${YELLOW}Warning: Not on main branch. Switching to main...${NC}"
    git checkout main
fi

# Fetch latest
echo -e "${GREEN}Fetching latest changes...${NC}"
git fetch origin

# Check if REL1_STABLE exists locally
if git show-ref --verify --quiet refs/heads/REL1_STABLE; then
    echo -e "${GREEN}REL1_STABLE branch exists locally${NC}"
    git checkout REL1_STABLE
    git merge origin/main --no-edit || {
        echo -e "${RED}Merge conflict detected. Please resolve manually.${NC}"
        exit 1
    }
else
    echo -e "${GREEN}Creating REL1_STABLE branch from main...${NC}"
    git checkout -b REL1_STABLE origin/main || git checkout -b REL1_STABLE main
fi

# Function to replace versions in files
replace_version() {
    local file="$1"
    local from_version="$2"
    local to_version="$3"
    
    if [ -f "$file" ]; then
        # Use sed with backup for macOS compatibility
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/${from_version}/${to_version}/g" "$file"
        else
            sed -i "s/${from_version}/${to_version}/g" "$file"
        fi
        echo "  Updated: $file"
    fi
}

echo -e "${GREEN}Replacing 2.0.0 with 1.0.0 in REL1_STABLE...${NC}"

# Update Helm chart
replace_version "helm/neurondb/Chart.yaml" "version: 2.0.0" "version: 1.0.0"
replace_version "helm/neurondb/Chart.yaml" 'appVersion: "2.0.0"' 'appVersion: "1.0.0"'

# Update Helm values files - be careful with image tags
find helm/neurondb -name "values*.yaml" -type f | while read file; do
    # Replace image tags
    replace_version "$file" 'tag: "2.0.0"' 'tag: "1.0.0"'
    replace_version "$file" 'tag: "2.0.0-pg17-cpu"' 'tag: "1.0.0-pg17-cpu"'
    replace_version "$file" 'tag: "2.0.0-pg17' 'tag: "1.0.0-pg17'
done

# Update package.json files (these should be 1.0.0 in REL1_STABLE, 2.0.0 in main)
replace_version "NeuronMCP/package.json" '"version": "2.0.0"' '"version": "1.0.0"'
replace_version "NeuronDesktop/frontend/package.json" '"version": "2.0.0"' '"version": "1.0.0"'

# Update Docker files
find dockers -name "Dockerfile*" -o -name "*.md" | while read file; do
    if [ -f "$file" ]; then
        replace_version "$file" "VERSION=2.0.0" "VERSION=1.0.0"
        replace_version "$file" "PACKAGE_VERSION=2.0.0" "PACKAGE_VERSION=1.0.0"
        replace_version "$file" "2.0.0.beta" "1.0.0.beta"
    fi
done

# Update packaging scripts
find packaging -name "*.sh" -o -name "*.md" | while read file; do
    if [ -f "$file" ]; then
        replace_version "$file" "VERSION=2.0.0" "VERSION=1.0.0"
        replace_version "$file" "2.0.0" "1.0.0"
    fi
done

# Update README and documentation
replace_version "README.md" "2.0.0" "1.0.0"
replace_version "VERSION_2.0.0.md" "2.0.0" "1.0.0" || true

# Update docker-compose files
find . -maxdepth 1 -name "docker-compose*.yml" | while read file; do
    replace_version "$file" "2.0.0" "1.0.0"
done

# Update NeuronDB control file - ensure it's 1.0
replace_version "NeuronDB/neurondb.control" "default_version = '2.0'" "default_version = '1.0'"

# Check for any remaining 2.0.0 references (excluding this script and git files)
echo -e "${YELLOW}Checking for remaining 2.0.0 references...${NC}"
REMAINING=$(grep -r "2\.0\.0" --exclude-dir=.git --exclude="*.sh" --exclude="sync-version-branches.sh" . 2>/dev/null | grep -v "1\.0\.0" | wc -l | tr -d ' ')
if [ "$REMAINING" -gt 0 ]; then
    echo -e "${YELLOW}Found $REMAINING remaining 2.0.0 references. Review manually:${NC}"
    grep -r "2\.0\.0" --exclude-dir=.git --exclude="*.sh" --exclude="sync-version-branches.sh" . 2>/dev/null | head -20
fi

# Commit changes to REL1_STABLE
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${GREEN}Committing version changes to REL1_STABLE...${NC}"
    git add -A
    git commit -m "NeuronDB: Update all modules to version 1.0.0 for REL1_STABLE branch.

This commit synchronizes REL1_STABLE with main branch while
maintaining version 1.0.0 for all components. All functionality
remains identical to main branch except for version numbers."
    
    echo -e "${GREEN}Pushing REL1_STABLE to origin...${NC}"
    git push origin REL1_STABLE || echo -e "${YELLOW}Push failed. You may need to force push with --force-with-lease${NC}"
else
    echo -e "${GREEN}No changes needed in REL1_STABLE${NC}"
fi

# Switch back to main and ensure it has 2.0.0
echo -e "${GREEN}Switching back to main and ensuring 2.0.0 versions...${NC}"
git checkout main

# Verify main has 2.0.0 in key files
echo -e "${GREEN}Verifying main branch has 2.0.0 versions...${NC}"

# Check Helm chart
if grep -q "version: 2.0.0" helm/neurondb/Chart.yaml; then
    echo -e "${GREEN}✓ Helm chart version is 2.0.0${NC}"
else
    echo -e "${RED}✗ Helm chart version is not 2.0.0${NC}"
    replace_version "helm/neurondb/Chart.yaml" "version: 1.0.0" "version: 2.0.0"
    replace_version "helm/neurondb/Chart.yaml" 'appVersion: "1.0.0"' 'appVersion: "2.0.0"'
fi

# Update package.json files to 2.0.0 in main
if grep -q '"version": "1.0.0"' NeuronMCP/package.json; then
    echo -e "${YELLOW}Updating NeuronMCP package.json to 2.0.0...${NC}"
    replace_version "NeuronMCP/package.json" '"version": "1.0.0"' '"version": "2.0.0"'
fi

if grep -q '"version": "1.0.0"' NeuronDesktop/frontend/package.json; then
    echo -e "${YELLOW}Updating NeuronDesktop frontend package.json to 2.0.0...${NC}"
    replace_version "NeuronDesktop/frontend/package.json" '"version": "1.0.0"' '"version": "2.0.0"'
fi

# Update Helm values image tags to 2.0.0 in main
find helm/neurondb -name "values*.yaml" -type f | while read file; do
    if grep -q 'tag: "1.0.0"' "$file"; then
        echo -e "${YELLOW}Updating image tags in $file to 2.0.0...${NC}"
        replace_version "$file" 'tag: "1.0.0"' 'tag: "2.0.0"'
        replace_version "$file" 'tag: "1.0.0-pg17-cpu"' 'tag: "2.0.0-pg17-cpu"'
        replace_version "$file" 'tag: "1.0.0-pg17' 'tag: "2.0.0-pg17'
    fi
done

# Commit any changes to main
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${GREEN}Committing version updates to main...${NC}"
    git add -A
    git commit -m "NeuronDB: Ensure all modules are at version 2.0.0 in main branch.

This commit ensures consistency of version 2.0.0 across all
components in the main branch."
    
    echo -e "${GREEN}Pushing main to origin...${NC}"
    git push origin main || echo -e "${YELLOW}Push failed. You may need to force push with --force-with-lease${NC}"
else
    echo -e "${GREEN}Main branch already has correct 2.0.0 versions${NC}"
fi

echo -e "${GREEN}✓ Version branch sync complete!${NC}"
echo -e "${GREEN}  REL1_STABLE: 1.0.0${NC}"
echo -e "${GREEN}  main: 2.0.0${NC}"

