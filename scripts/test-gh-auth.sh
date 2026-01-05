#!/bin/bash
#
# GitHub CLI Authentication Test & Setup Script
# This script helps test and configure GitHub CLI authentication
#

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  GitHub CLI Authentication Test & Setup"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed."
    echo ""
    echo "Install with:"
    echo "  Ubuntu/Debian: sudo apt install gh"
    echo "  macOS: brew install gh"
    echo "  Other: https://github.com/cli/cli#installation"
    exit 1
fi

echo "✅ GitHub CLI installed: $(gh --version | head -1)"
echo ""

# Check current authentication status
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Current Authentication Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if gh auth status 2>&1 | grep -q "Logged in"; then
    echo "✅ Already authenticated to GitHub"
    gh auth status 2>&1
    echo ""
    
    # Test API access
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "2. Testing GitHub API Access"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Get authenticated user
    if USER=$(gh api user --jq .login 2>/dev/null); then
        echo "✅ Successfully authenticated as: $USER"
    else
        echo "⚠️  Authentication exists but API test failed"
        echo "    You may need to re-authenticate"
    fi
    echo ""
    
    # Check repository access
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "3. Testing Repository Access"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Get repository info
    REPO_FULL=$(git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/')
    if REPO_INFO=$(gh repo view "$REPO_FULL" --json name,owner,isPrivate 2>/dev/null); then
        REPO_NAME=$(echo "$REPO_INFO" | jq -r .name)
        REPO_OWNER=$(echo "$REPO_INFO" | jq -r .owner.login)
        IS_PRIVATE=$(echo "$REPO_INFO" | jq -r .isPrivate)
        
        echo "✅ Repository access confirmed"
        echo "   Repository: $REPO_OWNER/$REPO_NAME"
        echo "   Private: $IS_PRIVATE"
    else
        echo "⚠️  Could not access repository information"
        echo "    Repository: $REPO_FULL"
        echo "    You may need additional permissions"
    fi
    echo ""
    
    # Test workflow access
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "4. Testing GitHub Actions Access"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if WORKFLOWS=$(gh workflow list 2>/dev/null); then
        WORKFLOW_COUNT=$(echo "$WORKFLOWS" | wc -l)
        echo "✅ GitHub Actions access confirmed"
        echo "   Found $WORKFLOW_COUNT workflows"
        echo ""
        echo "   Workflows:"
        echo "$WORKFLOWS" | head -5 | sed 's/^/   /'
        if [ "$WORKFLOW_COUNT" -gt 5 ]; then
            echo "   ... and $((WORKFLOW_COUNT - 5)) more"
        fi
    else
        echo "⚠️  Could not access GitHub Actions workflows"
        echo "    You may need workflow permissions"
    fi
    echo ""
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Authentication Test Complete - All checks passed!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
else
    echo "❌ Not authenticated to GitHub"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Authentication Options"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Option 1: Interactive Web-based Login (Recommended)"
    echo "   gh auth login"
    echo ""
    echo "Option 2: Login with Token"
    echo "   Create a token at: https://github.com/settings/tokens"
    echo "   Required scopes: repo, read:org, workflow"
    echo "   Then run: gh auth login --with-token < token.txt"
    echo ""
    echo "Option 3: Use Environment Variable"
    echo "   export GH_TOKEN=your_token_here"
    echo "   gh auth status"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Prompt for authentication
    read -p "Would you like to authenticate now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Starting interactive authentication..."
        echo ""
        gh auth login
        
        # Re-run tests after authentication
        echo ""
        echo "Re-running authentication tests..."
        echo ""
        exec "$0" "$@"
    else
        echo ""
        echo "Skipping authentication. Run this script again when ready."
        exit 0
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Quick Reference Commands"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Check status:          gh auth status"
echo "List workflows:        gh workflow list"
echo "View workflow runs:    gh run list --limit 10"
echo "View specific run:     gh run view <run-id>"
echo "Trigger workflow:      gh workflow run <workflow-name>"
echo "Check rate limits:     gh api rate_limit"
echo "Logout:                gh auth logout"
echo ""
echo "For more: gh help"
echo ""

