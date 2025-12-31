#!/bin/bash
# Audit script to find unsafe string functions that need replacement
# Usage: ./scripts/audit-unsafe-functions.sh

set -e

echo "=== NeuronDB Unsafe Function Audit ==="
echo ""

cd "$(dirname "$0")/.."

echo "Searching for unsafe string functions in C code..."
echo ""

# Count occurrences
STRCPY_COUNT=$(grep -rn "strcpy" NeuronDB/src --include="*.c" | wc -l || echo "0")
STRCAT_COUNT=$(grep -rn "strcat" NeuronDB/src --include="*.c" | wc -l || echo "0")
SPRINTF_COUNT=$(grep -rn "sprintf" NeuronDB/src --include="*.c" | wc -l || echo "0")
GETS_COUNT=$(grep -rn "gets" NeuronDB/src --include="*.c" | wc -l || echo "0")

echo "Found unsafe function usage:"
echo "  strcpy: $STRCPY_COUNT occurrences"
echo "  strcat: $STRCAT_COUNT occurrences"
echo "  sprintf: $SPRINTF_COUNT occurrences"
echo "  gets: $GETS_COUNT occurrences"
echo ""
echo "Total unsafe calls: $((STRCPY_COUNT + STRCAT_COUNT + SPRINTF_COUNT + GETS_COUNT))"
echo ""

echo "=== Recommended Replacements ==="
echo "strcpy(dest, src) -> snprintf(dest, sizeof(dest), \"%s\", src)"
echo "strcat(dest, src) -> snprintf(dest + strlen(dest), sizeof(dest) - strlen(dest), \"%s\", src)"
echo "sprintf(buf, fmt, ...) -> snprintf(buf, sizeof(buf), fmt, ...)"
echo "gets(buf) -> fgets(buf, sizeof(buf), stdin)"
echo ""

echo "=== Detailed Report ==="
echo ""
echo "strcpy occurrences:"
grep -rn "strcpy" NeuronDB/src --include="*.c" | head -20 || echo "  (none)"
echo ""
echo "strcat occurrences:"
grep -rn "strcat" NeuronDB/src --include="*.c" | head -20 || echo "  (none)"
echo ""
echo "sprintf occurrences:"
grep -rn "sprintf" NeuronDB/src --include="*.c" | head -20 || echo "  (none)"

