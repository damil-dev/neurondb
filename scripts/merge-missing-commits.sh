#!/bin/bash
# Merge missing commits from dev/DEV2 into main and REL1_STABLE
# This ensures all code changes are synchronized

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Merging missing commits from dev branches...${NC}"

# Ensure we're on main
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo -e "${YELLOW}Switching to main branch...${NC}"
    git checkout main
fi

git fetch --all

# Merge DEV2 commits if they exist
if git show-ref --verify --quiet refs/heads/DEV2; then
    echo -e "${GREEN}Checking DEV2 for missing commits...${NC}"
    
    # Get commits in DEV2 not in main (excluding version-only)
    MISSING_COMMITS=$(git log --oneline DEV2 --not origin/main | grep -v -i "version\|chore.*2\.0\.0\|docs.*2\.0\.0" | awk '{print $1}')
    
    if [ -n "$MISSING_COMMITS" ]; then
        echo -e "${YELLOW}Found missing commits in DEV2:${NC}"
        git log --oneline DEV2 --not origin/main | grep -v -i "version\|chore.*2\.0\.0\|docs.*2\.0\.0"
        
        read -p "Merge these commits into main? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # Cherry-pick non-version commits
            for commit in $MISSING_COMMITS; do
                echo -e "${GREEN}Cherry-picking $commit...${NC}"
                git cherry-pick "$commit" || {
                    echo -e "${RED}Conflict in $commit. Resolve manually.${NC}"
                    exit 1
                }
            done
        fi
    else
        echo -e "${GREEN}✓ No missing commits in DEV2${NC}"
    fi
fi

# Merge origin/dev commits if they exist
if git show-ref --verify --quiet refs/remotes/origin/dev; then
    echo -e "${GREEN}Checking origin/dev for missing commits...${NC}"
    
    # Get commits in dev not in main (excluding version-only)
    MISSING_COMMITS=$(git log --oneline origin/dev --not origin/main | grep -v -i "version\|chore.*2\.0\.0\|docs.*2\.0\.0" | awk '{print $1}')
    
    if [ -n "$MISSING_COMMITS" ]; then
        echo -e "${YELLOW}Found missing commits in origin/dev:${NC}"
        git log --oneline origin/dev --not origin/main | grep -v -i "version\|chore.*2\.0\.0\|docs.*2\.0\.0"
        
        read -p "Merge these commits into main? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # Cherry-pick non-version commits
            for commit in $MISSING_COMMITS; do
                echo -e "${GREEN}Cherry-picking $commit...${NC}"
                git cherry-pick "$commit" || {
                    echo -e "${RED}Conflict in $commit. Resolve manually.${NC}"
                    exit 1
                }
            done
        fi
    else
        echo -e "${GREEN}✓ No missing commits in origin/dev${NC}"
    fi
fi

echo -e "${GREEN}Merge complete!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Review merged changes"
echo -e "2. Run sync-version-branches.sh to update REL1_STABLE"
echo -e "3. Push changes to origin"



