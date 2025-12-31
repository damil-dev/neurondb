#!/bin/bash
# Generate detailed TODO report with context
# Part of Phase 1: Assessment for TODO Remediation Plan
# Usage: ./scripts/generate-todo-inventory.sh > todo_inventory.txt

set -e

echo "=== NeuronDB TODO/FIXME/HACK Inventory ==="
echo "Generated on: $(date)"
echo ""

cd "$(dirname "$0")/.."

# Count by type
TODO_COUNT=$(grep -rn "TODO" NeuronDB/ NeuronAgent/ NeuronMCP/ NeuronDesktop/ --include="*.c" --include="*.go" --include="*.ts" --include="*.tsx" 2>/dev/null | wc -l || echo "0")
FIXME_COUNT=$(grep -rn "FIXME" NeuronDB/ NeuronAgent/ NeuronMCP/ NeuronDesktop/ --include="*.c" --include="*.go" --include="*.ts" --include="*.tsx" 2>/dev/null | wc -l || echo "0")
HACK_COUNT=$(grep -rn "HACK" NeuronDB/ NeuronAgent/ NeuronMCP/ NeuronDesktop/ --include="*.c" --include="*.go" --include="*.ts" --include="*.tsx" 2>/dev/null | wc -l || echo "0")
XXX_COUNT=$(grep -rn "XXX" NeuronDB/ NeuronAgent/ NeuronMCP/ NeuronDesktop/ --include="*.c" --include="*.go" --include="*.ts" --include="*.tsx" 2>/dev/null | wc -l || echo "0")

echo "Summary:"
echo "  TODO:  $TODO_COUNT"
echo "  FIXME: $FIXME_COUNT"
echo "  HACK:  $HACK_COUNT"
echo "  XXX:   $XXX_COUNT"
echo "  Total: $((TODO_COUNT + FIXME_COUNT + HACK_COUNT + XXX_COUNT))"
echo ""

# Breakdown by component
echo "=== Breakdown by Component ==="
echo ""
echo "NeuronDB (C Extension):"
grep -rn "TODO\|FIXME\|HACK\|XXX" NeuronDB/src --include="*.c" 2>/dev/null | wc -l || echo "0"
echo ""
echo "NeuronAgent (Go):"
grep -rn "TODO\|FIXME\|HACK\|XXX" NeuronAgent/internal NeuronAgent/cmd --include="*.go" 2>/dev/null | wc -l || echo "0"
echo ""
echo "NeuronMCP (Go + TypeScript):"
grep -rn "TODO\|FIXME\|HACK\|XXX" NeuronMCP/internal NeuronMCP/src --include="*.go" --include="*.ts" 2>/dev/null | wc -l || echo "0"
echo ""
echo "NeuronDesktop (Go + React):"
grep -rn "TODO\|FIXME\|HACK\|XXX" NeuronDesktop/api NeuronDesktop/frontend --include="*.go" --include="*.ts" --include="*.tsx" 2>/dev/null | wc -l || echo "0"
echo ""

# High-density files (top 20)
echo "=== High-Density Files (Top 20) ==="
echo ""
grep -rn "TODO\|FIXME\|HACK\|XXX" NeuronDB/ NeuronAgent/ NeuronMCP/ NeuronDesktop/ --include="*.c" --include="*.go" --include="*.ts" --include="*.tsx" 2>/dev/null | \
    cut -d: -f1 | sort | uniq -c | sort -rn | head -20 || echo "  (none found)"
echo ""

# Security-related TODOs
echo "=== Security-Related TODOs ==="
echo ""
grep -rn "TODO.*[Ss]ecurity\|TODO.*[Ss]ecure\|TODO.*[Ii]njection\|TODO.*[Aa]uth\|FIXME.*[Ss]ecurity\|HACK.*[Ss]ecurity" \
    NeuronDB/ NeuronAgent/ NeuronMCP/ NeuronDesktop/ --include="*.c" --include="*.go" --include="*.ts" --include="*.tsx" 2>/dev/null | \
    head -30 || echo "  (none found)"
echo ""

# Memory-related TODOs
echo "=== Memory-Related TODOs ==="
echo ""
grep -rn "TODO.*[Mm]emory\|TODO.*[Ll]eak\|TODO.*free\|FIXME.*[Mm]emory\|HACK.*[Mm]emory" \
    NeuronDB/ NeuronAgent/ NeuronMCP/ NeuronDesktop/ --include="*.c" --include="*.go" --include="*.ts" --include="*.tsx" 2>/dev/null | \
    head -30 || echo "  (none found)"
echo ""

# Complete inventory
echo "=== Complete Inventory ==="
echo ""
grep -rn "TODO\|FIXME\|HACK\|XXX" NeuronDB/ NeuronAgent/ NeuronMCP/ NeuronDesktop/ --include="*.c" --include="*.go" --include="*.ts" --include="*.tsx" 2>/dev/null

