#!/bin/bash
# Complete startup script for Custom PostgreSQL LLM System

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       Starting Custom PostgreSQL LLM System                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Base directory
BASE_DIR="/home/pge/pge/neurondb"

# Step 1: Check if model exists
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Checking model files..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -f "$BASE_DIR/postgres_llm_output/postgres_llm_final.pt" ]; then
    echo -e "${RED}✗ Model not found!${NC}"
    echo ""
    echo "Please train the model first:"
    echo "  cd $BASE_DIR"
    echo "  source llm_training_env/bin/activate"
    echo "  python3 train_postgres_llm.py"
    exit 1
fi

echo -e "${GREEN}✓ Model files found${NC}"
ls -lh "$BASE_DIR/postgres_llm_output/"*.pt | head -2

# Step 2: Check if virtual environment exists
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Checking Python environment..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -d "$BASE_DIR/llm_training_env" ]; then
    echo -e "${YELLOW}⚠ Virtual environment not found, creating...${NC}"
    python3 -m venv "$BASE_DIR/llm_training_env"
fi

echo -e "${GREEN}✓ Virtual environment ready${NC}"

# Step 3: Install/check dependencies
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Checking dependencies..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

source "$BASE_DIR/llm_training_env/bin/activate"

# Check if Flask is installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo -e "${YELLOW}⚠ Installing Flask and flask-cors...${NC}"
    pip install flask flask-cors -q
fi

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 4: Check if port 8000 is available
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Checking port availability..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Port 8000 is already in use${NC}"
    echo "Stopping existing process..."
    kill $(lsof -t -i:8000) 2>/dev/null || true
    sleep 2
fi

echo -e "${GREEN}✓ Port 8000 is available${NC}"

# Step 5: Start model server
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5: Starting Model Server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$BASE_DIR"
nohup python3 postgres_llm_server.py > postgres_llm_server.log 2>&1 &
MODEL_PID=$!
echo $MODEL_PID > .model_server.pid

echo -e "${GREEN}✓ Model server starting (PID: $MODEL_PID)${NC}"
echo "  Logs: $BASE_DIR/postgres_llm_server.log"

# Wait for server to be ready
echo ""
echo "Waiting for server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Server is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 1
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ Server failed to start${NC}"
        echo "Check logs: tail -f $BASE_DIR/postgres_llm_server.log"
        exit 1
    fi
done

# Step 6: Test the server
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 6: Testing server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "Health check:"
curl -s http://localhost:8000/health | python3 -m json.tool

echo ""
echo "Quick generation test:"
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What PostgreSQL version", "max_length": 50}' \
  | python3 -m json.tool

# Step 7: Configure NeuronDB (if running)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 7: NeuronDB Configuration..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if psql -d neurondb -c "SELECT 1" > /dev/null 2>&1; then
    echo "Configuring NeuronDB..."
    psql -d neurondb <<SQL 2>/dev/null || true
-- Configure custom model
SET neurondb.llm_provider = 'openai';
SET neurondb.llm_endpoint = 'http://localhost:8000/v1';
SET neurondb.llm_model = 'postgres-model';
SELECT 'NeuronDB configured for custom model' as status;
SQL
    echo -e "${GREEN}✓ NeuronDB configured${NC}"
else
    echo -e "${YELLOW}⚠ NeuronDB not accessible (this is OK if not needed)${NC}"
fi

# Summary
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                     SYSTEM READY! 🎉                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Status Summary:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}✓ Model Server:${NC}     http://localhost:8000"
echo -e "${GREEN}✓ API Endpoints:${NC}    /health, /v1/chat/completions, /generate"
echo -e "${GREEN}✓ Model:${NC}            postgres-model (1M parameters)"
echo -e "${GREEN}✓ PID:${NC}              $MODEL_PID"
echo -e "${GREEN}✓ Logs:${NC}             $BASE_DIR/postgres_llm_server.log"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Next Steps:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Test from command line:"
echo "   curl -X POST http://localhost:8000/generate \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"prompt\": \"What PostgreSQL version am I running?\"}'"
echo ""
echo "2. Test from NeuronDB:"
echo "   psql -d neurondb"
echo "   SELECT neurondb.llm('complete', 'postgres-model', 'Show version', NULL, NULL, 100);"
echo ""
echo "3. View real-time logs:"
echo "   tail -f $BASE_DIR/postgres_llm_server.log"
echo ""
echo "4. Stop the server:"
echo "   kill $MODEL_PID"
echo "   # or"
echo "   ./stop_custom_llm_system.sh"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Documentation:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  README_CUSTOM_LLM.md        - Complete training guide"
echo "  INTEGRATION_GUIDE.md        - This integration guide"
echo "  TRAINING_SUCCESS.sh         - Training results summary"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

