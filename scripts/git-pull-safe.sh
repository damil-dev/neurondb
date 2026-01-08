#!/bin/bash
# Safe git pull that avoids merge commits
# Always uses rebase to maintain linear history

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Fetching latest changes...${NC}"
git fetch origin

# Check if we have uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${YELLOW}You have uncommitted changes. Stashing...${NC}"
    git stash push -m "Auto-stash before pull $(date +%Y-%m-%d_%H:%M:%S)"
    STASHED=true
else
    STASHED=false
fi

# Rebase instead of merge
echo -e "${GREEN}Rebasing on origin/$(git branch --show-current)...${NC}"
if git rebase "origin/$(git branch --show-current)"; then
    echo -e "${GREEN}✓ Successfully rebased${NC}"
else
    echo -e "${YELLOW}Rebase had conflicts. Resolve them and run: git rebase --continue${NC}"
    if [ "$STASHED" = true ]; then
        echo -e "${YELLOW}Your stashed changes are available with: git stash pop${NC}"
    fi
    exit 1
fi

# Restore stashed changes if any
if [ "$STASHED" = true ]; then
    echo -e "${GREEN}Restoring stashed changes...${NC}"
    git stash pop || {
        echo -e "${YELLOW}Could not automatically restore stashed changes. Run: git stash pop${NC}"
    }
fi

echo -e "${GREEN}Done!${NC}"



