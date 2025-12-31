#!/bin/bash
# Stop the Custom PostgreSQL LLM System

echo "Stopping Custom PostgreSQL LLM System..."

BASE_DIR="/home/pge/pge/neurondb"
PID_FILE="$BASE_DIR/.model_server.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "Stopping model server (PID: $PID)..."
        kill $PID
        sleep 2
        
        # Force kill if still running
        if ps -p $PID > /dev/null 2>&1; then
            echo "Force stopping..."
            kill -9 $PID
        fi
        
        rm "$PID_FILE"
        echo "✓ Model server stopped"
    else
        echo "⚠ Model server not running (stale PID file)"
        rm "$PID_FILE"
    fi
else
    # Try to find by port
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "Stopping process on port 8000..."
        kill $(lsof -t -i:8000) 2>/dev/null
        echo "✓ Server stopped"
    else
        echo "⚠ No server running on port 8000"
    fi
fi

echo ""
echo "System stopped."

