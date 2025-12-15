#!/bin/bash
# Run SVM test with core dump enabled

# Source the core dump setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/setup_core_dump.sh"

# Database connection parameters (adjust as needed)
DB_NAME="${DB_NAME:-neurondb}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-pge}"

echo "Running SVM test with core dump enabled..."
echo "Database: $DB_NAME on $DB_HOST:$DB_PORT"
echo ""

# Run the test SQL file
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
     -f "$SCRIPT_DIR/sql/basic/004_svm_basic.sql"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "Test failed with exit code: $EXIT_CODE"
    echo "Checking for core dump files..."
    
    # Look for core dump files
    CORE_FILES=$(find /tmp /var/crash "$HOME" "$SCRIPT_DIR" -name "core.*" -type f 2>/dev/null | head -5)
    
    if [ -n "$CORE_FILES" ]; then
        echo "Found core dump files:"
        echo "$CORE_FILES"
        echo ""
        echo "To analyze with gdb:"
        echo "  gdb /usr/lib/postgresql/*/bin/postgres <core_file>"
        echo "  or"
        echo "  gdb /path/to/postgres/bin/postgres <core_file>"
    else
        echo "No core dump files found."
        echo "Make sure core dumps are enabled and check:"
        echo "  - ulimit -c (should show 'unlimited')"
        echo "  - /proc/sys/kernel/core_pattern"
        echo "  - Check system logs: dmesg | tail"
    fi
else
    echo ""
    echo "Test completed successfully (no crash)."
fi

exit $EXIT_CODE




