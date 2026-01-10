#!/bin/bash
# Timeout wrapper for test execution
# Ensures tests never hang indefinitely

set -euo pipefail

TIMEOUT="${1:-60}"
COMMAND="${@:2}"

if [ -z "$COMMAND" ]; then
    echo "Usage: $0 <timeout_seconds> <command> [args...]"
    exit 1
fi

# Create a temporary file for the command output
OUTPUT_FILE=$(mktemp)
EXIT_CODE=124  # Timeout exit code

# Run command with timeout
timeout "$TIMEOUT" bash -c "$COMMAND" > "$OUTPUT_FILE" 2>&1 || EXIT_CODE=$?

# Check if timeout occurred
if [ $EXIT_CODE -eq 124 ]; then
    echo "ERROR: Test timed out after ${TIMEOUT} seconds"
    echo "Command: $COMMAND"
    echo "Output (last 100 lines):"
    tail -100 "$OUTPUT_FILE"
    rm -f "$OUTPUT_FILE"
    exit 124
elif [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Test failed with exit code $EXIT_CODE"
    echo "Command: $COMMAND"
    echo "Output:"
    cat "$OUTPUT_FILE"
    rm -f "$OUTPUT_FILE"
    exit $EXIT_CODE
else
    # Success - show output if verbose
    if [ "${VERBOSE:-0}" = "1" ]; then
        cat "$OUTPUT_FILE"
    fi
    rm -f "$OUTPUT_FILE"
    exit 0
fi


