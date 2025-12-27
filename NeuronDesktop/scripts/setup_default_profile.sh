#!/bin/bash
# Setup default profile for NeuronDesktop
# Auto-detects NeuronMCP binary and creates default profile with proper configuration

set -e

# Colors for output
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}Setting up default profile for NeuronDesktop...${NC}"

# Get project root (assuming script is in NeuronDesktop/scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NEURONDESKTOP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Database connection from environment variables
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-neurondesk}"
DB_USER="${DB_USER:-neurondesk}"
DB_PASSWORD="${DB_PASSWORD:-neurondesk}"

# NeuronDB connection (for MCP config)
NEURONDB_HOST="${NEURONDB_HOST:-localhost}"
NEURONDB_PORT="${NEURONDB_PORT:-5432}"
NEURONDB_DATABASE="${NEURONDB_DATABASE:-neurondb}"
NEURONDB_USER="${NEURONDB_USER:-neurondb}"
NEURONDB_PASSWORD="${NEURONDB_PASSWORD:-neurondb}"

# Agent endpoint
AGENT_ENDPOINT="${NEURONAGENT_ENDPOINT:-http://localhost:8080}"
AGENT_API_KEY="${NEURONAGENT_API_KEY:-}"

# User ID for default profile
USER_ID="${USER_ID:-default}"

# Function to find NeuronMCP binary
find_neurondb_mcp() {
    local mcp_binary=""
    
    # Check if path is explicitly set
    if [ -n "$NEURONMCP_BINARY_PATH" ]; then
        if [ -f "$NEURONMCP_BINARY_PATH" ] && [ -x "$NEURONMCP_BINARY_PATH" ]; then
            echo "$NEURONMCP_BINARY_PATH"
            return 0
        else
            echo -e "${YELLOW}Warning: NEURONMCP_BINARY_PATH set but binary not found or not executable${NC}" >&2
        fi
    fi
    
    # Check relative to project root: NeuronMCP/bin/neurondb-mcp
    local relative_path="$PROJECT_ROOT/NeuronMCP/bin/neurondb-mcp"
    if [ -f "$relative_path" ] && [ -x "$relative_path" ]; then
        echo "$relative_path"
        return 0
    fi
    
    # Check relative to NeuronDesktop: ../NeuronMCP/bin/neurondb-mcp
    local desktop_relative="$NEURONDESKTOP_ROOT/../NeuronMCP/bin/neurondb-mcp"
    if [ -f "$desktop_relative" ] && [ -x "$desktop_relative" ]; then
        echo "$desktop_relative"
        return 0
    fi
    
    # Try to build it if source exists
    local mcp_dir="$PROJECT_ROOT/NeuronMCP"
    if [ -d "$mcp_dir" ] && [ -f "$mcp_dir/cmd/neurondb-mcp/main.go" ]; then
        echo -e "${CYAN}Building NeuronMCP binary...${NC}"
        cd "$mcp_dir"
        if make build 2>/dev/null || go build -o bin/neurondb-mcp ./cmd/neurondb-mcp 2>/dev/null; then
            if [ -f "$mcp_dir/bin/neurondb-mcp" ] && [ -x "$mcp_dir/bin/neurondb-mcp" ]; then
                echo "$mcp_dir/bin/neurondb-mcp"
                return 0
            fi
        fi
    fi
    
    # Check PATH
    if command -v neurondb-mcp >/dev/null 2>&1; then
        echo "$(command -v neurondb-mcp)"
        return 0
    fi
    
    # Not found
    return 1
}

# Find NeuronMCP binary
echo -e "${CYAN}Detecting NeuronMCP binary...${NC}"
MCP_BINARY=$(find_neurondb_mcp || echo "")

if [ -z "$MCP_BINARY" ]; then
    echo -e "${YELLOW}Warning: NeuronMCP binary not found. Profile will be created without MCP config.${NC}"
    echo -e "${YELLOW}You can set NEURONMCP_BINARY_PATH environment variable to specify the path.${NC}"
    MCP_CONFIG="null"
else
    echo -e "${GREEN}✓ Found NeuronMCP binary: $MCP_BINARY${NC}"
    
    # Create MCP config JSON
    MCP_CONFIG=$(cat <<EOF
{
    "command": "$MCP_BINARY",
    "args": [],
    "env": {
        "NEURONDB_HOST": "$NEURONDB_HOST",
        "NEURONDB_PORT": "$NEURONDB_PORT",
        "NEURONDB_DATABASE": "$NEURONDB_DATABASE",
        "NEURONDB_USER": "$NEURONDB_USER",
        "NEURONDB_PASSWORD": "$NEURONDB_PASSWORD"
    }
}
EOF
)
fi

# Create NeuronDB DSN
NEURONDB_DSN="postgresql://${NEURONDB_USER}:${NEURONDB_PASSWORD}@${NEURONDB_HOST}:${NEURONDB_PORT}/${NEURONDB_DATABASE}"

# Check if profile already exists
echo -e "${CYAN}Checking for existing default profile...${NC}"
EXISTING_PROFILE=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT id FROM profiles WHERE name = 'Default' AND user_id = '$USER_ID' LIMIT 1" 2>/dev/null | xargs || echo "")

if [ -n "$EXISTING_PROFILE" ]; then
    echo -e "${YELLOW}Default profile already exists. Updating...${NC}"
    
    # Update existing profile
    if [ "$MCP_CONFIG" != "null" ]; then
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" <<EOF
UPDATE profiles 
SET 
    mcp_config = '$MCP_CONFIG'::jsonb,
    neurondb_dsn = '$NEURONDB_DSN',
    agent_endpoint = $(if [ -n "$AGENT_ENDPOINT" ]; then echo "'$AGENT_ENDPOINT'"; else echo "NULL"; fi),
    agent_api_key = $(if [ -n "$AGENT_API_KEY" ]; then echo "'$AGENT_API_KEY'"; else echo "NULL"; fi),
    is_default = true,
    updated_at = NOW()
WHERE id = '$EXISTING_PROFILE';

-- Unset other defaults
UPDATE profiles SET is_default = false WHERE id != '$EXISTING_PROFILE';
EOF
    else
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" <<EOF
UPDATE profiles 
SET 
    neurondb_dsn = '$NEURONDB_DSN',
    agent_endpoint = $(if [ -n "$AGENT_ENDPOINT" ]; then echo "'$AGENT_ENDPOINT'"; else echo "NULL"; fi),
    agent_api_key = $(if [ -n "$AGENT_API_KEY" ]; then echo "'$AGENT_API_KEY'"; else echo "NULL"; fi),
    is_default = true,
    updated_at = NOW()
WHERE id = '$EXISTING_PROFILE';

-- Unset other defaults
UPDATE profiles SET is_default = false WHERE id != '$EXISTING_PROFILE';
EOF
    fi
    
    echo -e "${GREEN}✓ Default profile updated${NC}"
else
    echo -e "${CYAN}Creating new default profile...${NC}"
    
    # Create new profile
    if [ "$MCP_CONFIG" != "null" ]; then
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" <<EOF
-- Unset any existing defaults
UPDATE profiles SET is_default = false;

-- Insert new default profile
INSERT INTO profiles (
    name,
    user_id,
    mcp_config,
    neurondb_dsn,
    agent_endpoint,
    agent_api_key,
    is_default,
    created_at,
    updated_at
) VALUES (
    'Default',
    '$USER_ID',
    '$MCP_CONFIG'::jsonb,
    '$NEURONDB_DSN',
    $(if [ -n "$AGENT_ENDPOINT" ]; then echo "'$AGENT_ENDPOINT'"; else echo "NULL"; fi),
    $(if [ -n "$AGENT_API_KEY" ]; then echo "'$AGENT_API_KEY'"; else echo "NULL"; fi),
    true,
    NOW(),
    NOW()
);
EOF
    else
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" <<EOF
-- Unset any existing defaults
UPDATE profiles SET is_default = false;

-- Insert new default profile
INSERT INTO profiles (
    name,
    user_id,
    neurondb_dsn,
    agent_endpoint,
    agent_api_key,
    is_default,
    created_at,
    updated_at
) VALUES (
    'Default',
    '$USER_ID',
    '$NEURONDB_DSN',
    $(if [ -n "$AGENT_ENDPOINT" ]; then echo "'$AGENT_ENDPOINT'"; else echo "NULL"; fi),
    $(if [ -n "$AGENT_API_KEY" ]; then echo "'$AGENT_API_KEY'"; else echo "NULL"; fi),
    true,
    NOW(),
    NOW()
);
EOF
    fi
    
    echo -e "${GREEN}✓ Default profile created${NC}"
fi

# Verify profile
echo -e "${CYAN}Verifying default profile...${NC}"
PROFILE_INFO=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -A -F'|' <<EOF
SELECT 
    id,
    name,
    is_default,
    mcp_config->>'command' as mcp_command
FROM profiles 
WHERE name = 'Default' AND user_id = '$USER_ID' AND is_default = true
LIMIT 1;
EOF
)

if [ -n "$PROFILE_INFO" ]; then
    echo -e "${GREEN}✓ Default profile verified${NC}"
    echo -e "${CYAN}Profile details:${NC}"
    echo "$PROFILE_INFO" | while IFS='|' read -r id name is_default mcp_command; do
        echo "  ID: $id"
        echo "  Name: $name"
        echo "  Is Default: $is_default"
        if [ -n "$mcp_command" ]; then
            echo "  MCP Command: $mcp_command"
        fi
    done
else
    echo -e "${RED}✗ Failed to verify default profile${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Default profile setup complete!${NC}"

