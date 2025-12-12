#!/bin/bash
# Script to analyze PostgreSQL core dumps

CORE_DIR="/tmp/core"
POSTGRES_BIN="/usr/local/pgsql.18-pge/bin/postgres"
OUTPUT_DIR="./core_analysis"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Analyzing core dumps in $CORE_DIR..."
echo "======================================"
echo ""

# Check if postgres binary exists
if [ ! -f "$POSTGRES_BIN" ]; then
    echo "Warning: Postgres binary not found at $POSTGRES_BIN"
    echo "Attempting to find postgres binary..."
    POSTGRES_BIN=$(which postgres 2>/dev/null || find /usr -name postgres -type f 2>/dev/null | head -1)
    if [ -z "$POSTGRES_BIN" ]; then
        echo "Error: Could not find postgres binary"
        exit 1
    fi
    echo "Using: $POSTGRES_BIN"
fi

# Process each core dump
for core_file in "$CORE_DIR"/postgres.postgres.*; do
    if [ ! -f "$core_file" ]; then
        continue
    fi
    
    core_name=$(basename "$core_file")
    output_file="$OUTPUT_DIR/${core_name}.txt"
    
    echo "Analyzing: $core_name"
    echo "Output: $output_file"
    
    # Extract PID from filename (format: postgres.postgres.PID)
    pid=$(echo "$core_name" | sed 's/postgres\.postgres\.//')
    
    # Use gdb to get stack trace and info
    gdb -batch -ex "set pagination off" \
        -ex "file $POSTGRES_BIN" \
        -ex "core-file $core_file" \
        -ex "bt" \
        -ex "info registers" \
        -ex "info locals" \
        -ex "thread apply all bt" \
        -ex "quit" \
        2>&1 > "$output_file"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Analysis complete"
    else
        echo "  ✗ Analysis failed"
    fi
    echo ""
done

echo "Summary of all core dumps:"
echo "=========================="
for output_file in "$OUTPUT_DIR"/*.txt; do
    if [ -f "$output_file" ]; then
        echo ""
        echo "File: $(basename "$output_file")"
        echo "---"
        # Extract the first stack frame (most relevant)
        head -30 "$output_file" | grep -A 20 "#0\|#1\|#2" | head -20
        echo ""
    fi
done

echo ""
echo "Full analysis files saved in: $OUTPUT_DIR"
echo "To view a specific analysis: cat $OUTPUT_DIR/<core_file>.txt"



