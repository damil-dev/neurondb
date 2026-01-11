#!/bin/bash
# Create a sample NeuronAgent via API
# Checks NeuronAgent health and creates a sample assistant agent

set -e

# Colors for output
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}Creating sample NeuronAgent...${NC}"

# Configuration from environment variables
AGENT_ENDPOINT="${NEURONAGENT_ENDPOINT:-http://localhost:8080}"
AGENT_API_KEY="${NEURONAGENT_API_KEY:-}"

# Sample agent configuration
AGENT_NAME="${SAMPLE_AGENT_NAME:-sample-assistant}"
AGENT_DESCRIPTION="${SAMPLE_AGENT_DESCRIPTION:-General purpose assistant for answering questions and helping with tasks}"
AGENT_SYSTEM_PROMPT="${SAMPLE_AGENT_SYSTEM_PROMPT:-You are a helpful, harmless, and honest assistant. Answer questions accurately and helpfully.}"
AGENT_MODEL="${SAMPLE_AGENT_MODEL:-gpt-4}"
AGENT_TOOLS="${SAMPLE_AGENT_TOOLS:-sql,http}"

# Check if NeuronAgent is running
echo -e "${CYAN}Checking NeuronAgent health...${NC}"
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$AGENT_ENDPOINT/health" 2>/dev/null || echo "000")

if [ "$HEALTH_RESPONSE" != "200" ]; then
    echo -e "${YELLOW}Warning: NeuronAgent is not responding at $AGENT_ENDPOINT${NC}"
    echo -e "${YELLOW}Status code: $HEALTH_RESPONSE${NC}"
    echo -e "${YELLOW}Skipping sample agent creation.${NC}"
    echo -e "${YELLOW}Make sure NeuronAgent is running and set NEURONAGENT_ENDPOINT if using a different URL.${NC}"
    exit 0
fi

echo -e "${GREEN}✓ NeuronAgent is healthy${NC}"

# Check if API key is provided
if [ -z "$AGENT_API_KEY" ]; then
    echo -e "${YELLOW}Warning: NEURONAGENT_API_KEY not set.${NC}"
    echo -e "${YELLOW}Attempting to create agent without API key (may fail if authentication is required)...${NC}"
    AUTH_HEADER=""
else
    AUTH_HEADER="Authorization: Bearer $AGENT_API_KEY"
    echo -e "${GREEN}✓ API key provided${NC}"
fi

# Check if agent already exists
echo -e "${CYAN}Checking if sample agent already exists...${NC}"
if [ -n "$AUTH_HEADER" ]; then
    EXISTING_AGENTS=$(curl -s -H "$AUTH_HEADER" "$AGENT_ENDPOINT/api/v1/agents" 2>/dev/null || echo "[]")
else
    EXISTING_AGENTS=$(curl -s "$AGENT_ENDPOINT/api/v1/agents" 2>/dev/null || echo "[]")
fi

# Check if agent with this name already exists
if echo "$EXISTING_AGENTS" | grep -q "\"name\":\"$AGENT_NAME\""; then
    echo -e "${YELLOW}Sample agent '$AGENT_NAME' already exists. Skipping creation.${NC}"
    exit 0
fi

# Create agent
echo -e "${CYAN}Creating sample agent: $AGENT_NAME...${NC}"

# Convert tools string to JSON array
IFS=',' read -ra TOOLS_ARRAY <<< "$AGENT_TOOLS"
TOOLS_JSON="["
for i in "${!TOOLS_ARRAY[@]}"; do
    if [ $i -gt 0 ]; then
        TOOLS_JSON+=","
    fi
    TOOLS_JSON+="\"${TOOLS_ARRAY[$i]}\""
done
TOOLS_JSON+="]"

# Create agent payload
AGENT_PAYLOAD=$(cat <<EOF
{
    "name": "$AGENT_NAME",
    "description": "$AGENT_DESCRIPTION",
    "system_prompt": "$AGENT_SYSTEM_PROMPT",
    "model_name": "$AGENT_MODEL",
    "enabled_tools": $TOOLS_JSON,
    "config": {
        "temperature": 0.7,
        "max_tokens": 1000
    }
}
EOF
)

# Make API call
if [ -n "$AUTH_HEADER" ]; then
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
        -H "$AUTH_HEADER" \
        -H "Content-Type: application/json" \
        -d "$AGENT_PAYLOAD" \
        "$AGENT_ENDPOINT/api/v1/agents" 2>/dev/null || echo -e "\n000")
else
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d "$AGENT_PAYLOAD" \
        "$AGENT_ENDPOINT/api/v1/agents" 2>/dev/null || echo -e "\n000")
fi

# Extract HTTP status code (last line)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

# Check response
if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
    AGENT_ID=$(echo "$BODY" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4 || echo "")
    if [ -n "$AGENT_ID" ]; then
        echo -e "${GREEN}✓ Sample agent created successfully!${NC}"
        echo -e "${CYAN}Agent details:${NC}"
        echo "  ID: $AGENT_ID"
        echo "  Name: $AGENT_NAME"
        echo "  Model: $AGENT_MODEL"
        echo "  Tools: $AGENT_TOOLS"
        echo "  Endpoint: $AGENT_ENDPOINT"
    else
        echo -e "${GREEN}✓ Agent created (ID extraction failed, but creation succeeded)${NC}"
        echo "$BODY" | head -20
    fi
elif [ "$HTTP_CODE" = "401" ] || [ "$HTTP_CODE" = "403" ]; then
    echo -e "${RED}✗ Authentication failed. Please set NEURONAGENT_API_KEY environment variable.${NC}"
    exit 1
elif [ "$HTTP_CODE" = "000" ]; then
    echo -e "${RED}✗ Failed to connect to NeuronAgent at $AGENT_ENDPOINT${NC}"
    exit 1
else
    echo -e "${RED}✗ Failed to create agent (HTTP $HTTP_CODE)${NC}"
    echo -e "${RED}Response: $BODY${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Sample agent creation complete!${NC}"

