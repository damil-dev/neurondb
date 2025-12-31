#!/bin/bash
# Export PostgreSQL LLM to Ollama format

echo "=========================================="
echo "Export PostgreSQL LLM to Ollama"
echo "=========================================="
echo ""

MODEL_DIR="postgres_llm_output"
OUTPUT_NAME="postgres-model"

# Check if model exists
if [ ! -f "$MODEL_DIR/postgres_llm_final.pt" ]; then
    echo "Error: Model not found in $MODEL_DIR"
    echo "Please run train_postgres_llm.py first"
    exit 1
fi

echo "Step 1: Convert PyTorch model to GGUF format..."
echo "(This requires llama.cpp - installing if needed)"

# Install llama.cpp if not present
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    make
    cd ..
fi

echo ""
echo "Step 2: Create Ollama Modelfile..."

cat > Modelfile.postgres <<EOF
FROM $MODEL_DIR/postgres_llm_final.pt

TEMPLATE """<|user|>{{ .Prompt }}<|assistant|>"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 512
PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"

SYSTEM """You are a PostgreSQL database expert. When asked about database operations, use tool calls.

Available tools:
- postgresql_version: Get PostgreSQL version
- postgresql_stats: Get database statistics
- postgresql_databases: List databases
- postgresql_connections: Show connections
- postgresql_settings: Get settings
- postgresql_extensions: List extensions

Format: TOOL_CALL: {"name": "tool_name", "arguments": {...}}"""
EOF

echo "✓ Modelfile created: Modelfile.postgres"
echo ""

echo "Step 3: Create Ollama model..."
echo "Run: ollama create $OUTPUT_NAME -f Modelfile.postgres"
echo ""

echo "Step 4: Test the model..."
echo "Run: ollama run $OUTPUT_NAME 'What PostgreSQL version am I running?'"
echo ""

echo "Step 5: Configure in NeuronDB..."
cat <<SQL
-- In psql:
SET neurondb.llm_provider = 'ollama';
SET neurondb.llm_model = '$OUTPUT_NAME';
SET neurondb.llm_endpoint = 'http://localhost:11434';

-- Test:
SELECT neurondb.llm('complete', '$OUTPUT_NAME', 'Hello', NULL, NULL, 50);
SQL

echo ""
echo "=========================================="
echo "Export instructions ready!"
echo "=========================================="

