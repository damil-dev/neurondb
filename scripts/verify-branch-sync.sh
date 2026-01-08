#!/bin/bash
# Verify that all code changes from dev/DEV2 are in main and REL1_STABLE
# Only version numbers should differ between branches

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Verifying branch synchronization...${NC}"

# Fetch all branches
echo -e "${GREEN}Fetching all branches...${NC}"
git fetch --all

# Check DEV2 branch
if git show-ref --verify --quiet refs/heads/DEV2; then
    echo -e "${GREEN}Checking DEV2 branch...${NC}"
    
    # Get commits in DEV2 not in main (excluding version-only commits)
    DEV2_COMMITS=$(git log --oneline DEV2 --not origin/main | grep -v -i "version\|chore.*2\.0\.0\|docs.*2\.0\.0" | wc -l | tr -d ' ')
    
    if [ "$DEV2_COMMITS" -gt 0 ]; then
        echo -e "${YELLOW}Found $DEV2_COMMITS non-version commits in DEV2 not in main:${NC}"
        git log --oneline DEV2 --not origin/main | grep -v -i "version\|chore.*2\.0\.0\|docs.*2\.0\.0" | head -10
    else
        echo -e "${GREEN}✓ DEV2 has no code changes missing from main${NC}"
    fi
    
    # Check file differences (excluding version files)
    DEV2_FILES=$(git diff --name-only DEV2 origin/main | grep -v -E "(Chart\.yaml|package\.json|values.*\.yaml|VERSION|\.control|Dockerfile.*package)" | wc -l | tr -d ' ')
    
    if [ "$DEV2_FILES" -gt 0 ]; then
        echo -e "${YELLOW}Found $DEV2_FILES non-version files different in DEV2:${NC}"
        git diff --name-only DEV2 origin/main | grep -v -E "(Chart\.yaml|package\.json|values.*\.yaml|VERSION|\.control|Dockerfile.*package)" | head -20
    else
        echo -e "${GREEN}✓ DEV2 has no code file differences from main (excluding versions)${NC}"
    fi
else
    echo -e "${YELLOW}DEV2 branch not found locally${NC}"
fi

# Check origin/dev branch
if git show-ref --verify --quiet refs/remotes/origin/dev; then
    echo -e "${GREEN}Checking origin/dev branch...${NC}"
    
    # Get commits in dev not in main
    DEV_COMMITS=$(git log --oneline origin/dev --not origin/main | grep -v -i "version\|chore.*2\.0\.0\|docs.*2\.0\.0" | wc -l | tr -d ' ')
    
    if [ "$DEV_COMMITS" -gt 0 ]; then
        echo -e "${YELLOW}Found $DEV_COMMITS non-version commits in origin/dev not in main:${NC}"
        git log --oneline origin/dev --not origin/main | grep -v -i "version\|chore.*2\.0\.0\|docs.*2\.0\.0" | head -10
    else
        echo -e "${GREEN}✓ origin/dev has no code changes missing from main${NC}"
    fi
    
    # Check file differences
    DEV_FILES=$(git diff --name-only origin/dev origin/main | grep -v -E "(Chart\.yaml|package\.json|values.*\.yaml|VERSION|\.control|Dockerfile.*package)" | wc -l | tr -d ' ')
    
    if [ "$DEV_FILES" -gt 0 ]; then
        echo -e "${YELLOW}Found $DEV_FILES non-version files different in origin/dev:${NC}"
        git diff --name-only origin/dev origin/main | grep -v -E "(Chart\.yaml|package\.json|values.*\.yaml|VERSION|\.control|Dockerfile.*package)" | head -20
    else
        echo -e "${GREEN}✓ origin/dev has no code file differences from main (excluding versions)${NC}"
    fi
else
    echo -e "${YELLOW}origin/dev branch not found${NC}"
fi

# Compare main and REL1_STABLE (should only differ in versions)
echo -e "${GREEN}Comparing main and REL1_STABLE...${NC}"

MAIN_REL1_FILES=$(git diff --name-only origin/main origin/REL1_STABLE | grep -v -E "(Chart\.yaml|package\.json|values.*\.yaml|VERSION|\.control|Dockerfile.*package|README\.md|\.md$)" | wc -l | tr -d ' ')

if [ "$MAIN_REL1_FILES" -gt 0 ]; then
    echo -e "${RED}⚠ Found $MAIN_REL1_FILES code files different between main and REL1_STABLE:${NC}"
    git diff --name-only origin/main origin/REL1_STABLE | grep -v -E "(Chart\.yaml|package\.json|values.*\.yaml|VERSION|\.control|Dockerfile.*package|README\.md|\.md$)" | head -20
    echo -e "${YELLOW}These should only differ in version numbers!${NC}"
else
    echo -e "${GREEN}✓ main and REL1_STABLE differ only in version files${NC}"
fi

# Check for missing commits
echo -e "${GREEN}Checking for missing commits...${NC}"

# Commits in main not in REL1_STABLE (should be minimal, mostly version updates)
MAIN_NOT_REL1=$(git log --oneline origin/main --not origin/REL1_STABLE | wc -l | tr -d ' ')
REL1_NOT_MAIN=$(git log --oneline origin/REL1_STABLE --not origin/main | wc -l | tr -d ' ')

echo -e "Commits in main not in REL1_STABLE: $MAIN_NOT_REL1"
echo -e "Commits in REL1_STABLE not in main: $REL1_NOT_MAIN"

if [ "$MAIN_NOT_REL1" -gt 20 ] || [ "$REL1_NOT_MAIN" -gt 20 ]; then
    echo -e "${YELLOW}⚠ Large number of divergent commits. Review needed.${NC}"
else
    echo -e "${GREEN}✓ Commit divergence is within expected range${NC}"
fi

echo -e "${GREEN}Verification complete!${NC}"



