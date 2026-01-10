#!/bin/bash
# Quick launcher for comprehensive tests

cd "$(dirname "$0")/.."

echo "Starting comprehensive test suite..."
echo ""

# Run the comprehensive test runner
bash scripts/comprehensive_test_runner.sh "$@"


