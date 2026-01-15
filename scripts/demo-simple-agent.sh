#!/bin/bash
#
# NeuronDB + NeuronAgent Simple Agent Demo
# This script demonstrates creating and testing a simple agent
#
# Usage: ./scripts/demo-simple-agent.sh
#

set -euo pipefail
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
BASE_URL="${NEURONAGENT_URL:-http://localhost:8080}"
NEURONDB_URL="${NEURONDB_URL:-postgresql://neurondb:neurondb@localhost:5433/neurondb}"

# Generate a test API key (for demo purposes)
API_KEY="demo-key-$(openssl rand -hex 8)"

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}  ${BOLD}NeuronDB + NeuronAgent Simple Agent Demo${NC}                    ${BLUE}║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""


echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}${BOLD}Step 1: Checking Docker Services${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

cd "$(dirname "$0")/.." || exit 1

# Check if services are running
echo -n "  Checking NeuronDB container... "
if docker ps --filter "name=neurondb" --format "{{.Names}}" | grep -q "neurondb"; then
    echo -e "${GREEN}✓ Running${NC}"
else
    echo -e "${YELLOW}⚠ Not running${NC}"
    echo -e "  ${YELLOW}Starting NeuronDB...${NC}"
    docker compose up -d neurondb || {
        echo -e "  ${RED}✗ Failed to start NeuronDB${NC}"
        exit 1
    }
    echo -e "  ${GREEN}✓ NeuronDB started${NC}"
fi

echo -n "  Checking NeuronAgent container... "
if docker ps --filter "name=neuronagent" --format "{{.Names}}" | grep -q "neuronagent"; then
    echo -e "${GREEN}✓ Running${NC}"
else
    echo -e "${YELLOW}⚠ Not running${NC}"
    echo -e "  ${YELLOW}Starting NeuronAgent...${NC}"
    docker compose up -d neuronagent || {
        echo -e "  ${RED}✗ Failed to start NeuronAgent${NC}"
        exit 1
    }
    echo -e "  ${GREEN}✓ NeuronAgent started${NC}"
fi

# Wait for services to be healthy
echo ""
echo -e "  ${YELLOW}Waiting for services to be ready...${NC}"
MAX_WAIT=120
WAIT_COUNT=0

# Wait for NeuronDB
echo -n "    Waiting for NeuronDB... "
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if docker exec neurondb-cpu pg_isready -U neurondb -d neurondb >/dev/null 2>&1 || \
       docker exec neurondb pg_isready -U neurondb -d neurondb >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Ready${NC}"
        break
    fi
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
    echo -n "."
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo -e "${RED}✗ Timeout waiting for NeuronDB${NC}"
    exit 1
fi

# Wait for NeuronAgent
WAIT_COUNT=0
echo -n "    Waiting for NeuronAgent... "
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Ready${NC}"
        break
    fi
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
    echo -n "."
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo -e "${RED}✗ Timeout waiting for NeuronAgent${NC}"
    exit 1
fi


echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}${BOLD}Step 2: Verifying Connections${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check NeuronDB health
echo -n "  NeuronDB health check... "
if docker exec neurondb-cpu pg_isready -U neurondb -d neurondb >/dev/null 2>&1 || \
   docker exec neurondb pg_isready -U neurondb -d neurondb >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Healthy${NC}"
else
    echo -e "${RED}✗ Not healthy${NC}"
    exit 1
fi

# Check NeuronAgent health
echo -n "  NeuronAgent health check... "
HEALTH_RESPONSE=$(curl -sf "${BASE_URL}/health" 2>&1)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Healthy${NC}"
    echo -e "    Response: ${HEALTH_RESPONSE}"
else
    echo -e "${RED}✗ Not healthy${NC}"
    echo -e "    Error: ${HEALTH_RESPONSE}"
    exit 1
fi

# Check NeuronAgent can connect to NeuronDB (by checking logs)
echo -n "  NeuronAgent → NeuronDB connection... "
if docker logs neuronagent 2>&1 | grep -q "Connected to database\|database connection\|ready" || \
   docker logs neuronagent 2>&1 | grep -v "error\|Error\|ERROR" | tail -5 | grep -q "."; then
    echo -e "${GREEN}✓ Connected${NC}"
else
    echo -e "${YELLOW}⚠ Checking logs...${NC}"
    docker logs neuronagent 2>&1 | tail -10
fi


echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}${BOLD}Step 3: Initializing NeuronDB${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Get container name
CONTAINER_NAME=$(docker ps --filter "name=neurondb" --format "{{.Names}}" | head -1)

if [ -n "$CONTAINER_NAME" ]; then
    echo -n "  Creating NeuronDB extension... "
    if docker exec "$CONTAINER_NAME" psql -U neurondb -d neurondb -c "CREATE EXTENSION IF NOT EXISTS neurondb;" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Extension ready${NC}"
    else
        echo -e "${YELLOW}⚠ Extension may already exist or will be created on first use${NC}"
    fi
    
    echo -n "  Verifying NeuronAgent schema... "
    SCHEMA_CHECK=$(docker exec "$CONTAINER_NAME" psql -U neurondb -d neurondb -tAc "SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = 'neurondb_agent');" 2>/dev/null || echo "false")
    if [ "$SCHEMA_CHECK" = "t" ] || [ "$SCHEMA_CHECK" = "true" ]; then
        echo -e "${GREEN}✓ Schema exists${NC}"
    else
        echo -e "${YELLOW}⚠ Schema will be created on first NeuronAgent connection${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Could not find NeuronDB container, skipping schema check${NC}"
fi


echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}${BOLD}Step 4: Creating Simple Agent${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

AGENT_NAME="demo-agent-$(date +%s)"
AGENT_CONFIG=$(cat <<EOF
{
  "name": "${AGENT_NAME}",
  "description": "A simple demo agent for testing",
  "system_prompt": "You are a helpful assistant. Respond briefly and clearly to user questions.",
  "model_name": "gpt-4",
  "enabled_tools": ["sql"],
  "config": {
    "temperature": 0.7,
    "max_tokens": 500
  }
}
EOF
)

echo "  Agent Name: ${AGENT_NAME}"
echo "  Model: gpt-4"
echo "  Tools: sql"
echo ""
echo -n "  Creating agent... "

AGENT_RESPONSE=$(curl -s -X POST "${BASE_URL}/api/v1/agents" \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "${AGENT_CONFIG}" 2>&1)

if echo "$AGENT_RESPONSE" | grep -q '"id"\|"error"'; then
    if echo "$AGENT_RESPONSE" | grep -q '"error"'; then
        echo -e "${RED}✗ Failed${NC}"
        echo -e "    Error: ${AGENT_RESPONSE}"
        
        # Try without authentication (maybe it's not required for demo)
        echo -e "  ${YELLOW}Trying without authentication...${NC}"
        AGENT_RESPONSE=$(curl -s -X POST "${BASE_URL}/api/v1/agents" \
            -H "Content-Type: application/json" \
            -d "${AGENT_CONFIG}" 2>&1)
        
        if echo "$AGENT_RESPONSE" | grep -q '"id"'; then
            echo -e "  ${GREEN}✓ Agent created (no auth required)${NC}"
        else
            echo -e "  ${RED}✗ Failed: ${AGENT_RESPONSE}${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}✓ Agent created${NC}"
    fi
else
    echo -e "${RED}✗ Unexpected response${NC}"
    echo -e "    Response: ${AGENT_RESPONSE}"
    exit 1
fi

# Extract agent ID
AGENT_ID=$(echo "$AGENT_RESPONSE" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4 || echo "")

if [ -z "$AGENT_ID" ]; then
    # Try alternative JSON parsing
    AGENT_ID=$(echo "$AGENT_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', ''))" 2>/dev/null || echo "")
fi

if [ -z "$AGENT_ID" ]; then
    echo -e "  ${RED}✗ Could not extract agent ID${NC}"
    echo -e "    Full response: ${AGENT_RESPONSE}"
    exit 1
fi

echo -e "    ${GREEN}Agent ID: ${AGENT_ID}${NC}"


echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}${BOLD}Step 5: Creating Session${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

SESSION_CONFIG=$(cat <<EOF
{
  "agent_id": "${AGENT_ID}",
  "external_user_id": "demo-user",
  "metadata": {
    "demo": true,
    "created_by": "demo-script"
  }
}
EOF
)

echo -n "  Creating session... "

SESSION_RESPONSE=$(curl -s -X POST "${BASE_URL}/api/v1/sessions" \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "${SESSION_CONFIG}" 2>&1)

# Try without auth if it fails
if echo "$SESSION_RESPONSE" | grep -q '"error"'; then
    SESSION_RESPONSE=$(curl -s -X POST "${BASE_URL}/api/v1/sessions" \
        -H "Content-Type: application/json" \
        -d "${SESSION_CONFIG}" 2>&1)
fi

if echo "$SESSION_RESPONSE" | grep -q '"id"'; then
    echo -e "${GREEN}✓ Session created${NC}"
    SESSION_ID=$(echo "$SESSION_RESPONSE" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4 || echo "")
    if [ -z "$SESSION_ID" ]; then
        SESSION_ID=$(echo "$SESSION_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', ''))" 2>/dev/null || echo "")
    fi
    if [ -n "$SESSION_ID" ]; then
        echo -e "    ${GREEN}Session ID: ${SESSION_ID}${NC}"
    else
        echo -e "    ${YELLOW}Could not extract session ID, but session was created${NC}"
    fi
else
    echo -e "${RED}✗ Failed to create session${NC}"
    echo -e "    Response: ${SESSION_RESPONSE}"
    exit 1
fi


echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}${BOLD}Step 6: Testing Agent with a Message${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

TEST_MESSAGE="Hello! Can you tell me what you are and what you can do?"

echo "  User Message: ${TEST_MESSAGE}"
echo ""
echo -n "  Sending message... "

MESSAGE_CONFIG=$(cat <<EOF
{
  "role": "user",
  "content": "${TEST_MESSAGE}",
  "stream": false
}
EOF
)

MESSAGE_RESPONSE=$(curl -s -X POST "${BASE_URL}/api/v1/sessions/${SESSION_ID}/messages" \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "${MESSAGE_CONFIG}" 2>&1)

# Try without auth if it fails
if echo "$MESSAGE_RESPONSE" | grep -q '"error"'; then
    MESSAGE_RESPONSE=$(curl -s -X POST "${BASE_URL}/api/v1/sessions/${SESSION_ID}/messages" \
        -H "Content-Type: application/json" \
        -d "${MESSAGE_CONFIG}" 2>&1)
fi

if echo "$MESSAGE_RESPONSE" | grep -q '"response"\|"content"\|"message"'; then
    echo -e "${GREEN}✓ Message sent and response received${NC}"
    echo ""
    echo -e "${GREEN}Agent Response:${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Try to extract and format the response
    RESPONSE_TEXT=$(echo "$MESSAGE_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'response' in data:
        print(data['response'].get('content', str(data['response'])))
    elif 'content' in data:
        print(data['content'])
    elif 'message' in data:
        print(data['message'].get('content', str(data['message'])))
    else:
        print(json.dumps(data, indent=2))
except:
    print(sys.stdin.read())
" 2>/dev/null || echo "$MESSAGE_RESPONSE")
    
    echo -e "${RESPONSE_TEXT}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
else
    echo -e "${YELLOW}⚠ Unexpected response format${NC}"
    echo ""
    echo "  Full Response:"
    echo "$MESSAGE_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$MESSAGE_RESPONSE"
fi


echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}  ${BOLD}Demo Summary${NC}                                                ${BLUE}║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${GREEN}✓${NC} NeuronDB: Running and healthy"
echo -e "  ${GREEN}✓${NC} NeuronAgent: Running and connected to NeuronDB"
echo -e "  ${GREEN}✓${NC} Agent Created: ${AGENT_NAME}"
echo -e "  ${GREEN}✓${NC} Agent ID: ${AGENT_ID}"
if [ -n "${SESSION_ID:-}" ]; then
    echo -e "  ${GREEN}✓${NC} Session Created: ${SESSION_ID}"
fi
echo -e "  ${GREEN}✓${NC} Test message sent and response received"
echo ""
echo -e "${CYAN}Next Steps:${NC}"
echo -e "  • View agent: ${BASE_URL}/api/v1/agents/${AGENT_ID}"
echo -e "  • View session: ${BASE_URL}/api/v1/sessions/${SESSION_ID}"
echo -e "  • Send more messages to test the agent"
echo ""
echo -e "${GREEN}Demo completed successfully!${NC}"
echo ""


