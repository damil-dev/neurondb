#!/bin/bash
# Fix NeuronDB Extension Conflict Script
#
# Resolves conflicts when creating the NeuronDB extension due to existing
# functions that are not owned by the extension.
#
# Usage:
#   ./scripts/fix_extension_conflict.sh
#   DB_NAME=neurondb DB_USER=postgres ./scripts/fix_extension_conflict.sh

set -e

# Default values
DB_NAME="${DB_NAME:-neurondb}"
DB_USER="${DB_USER:-postgres}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_PASSWORD="${DB_PASSWORD:-}"

if [ -n "$DB_PASSWORD" ]; then
    export PGPASSWORD="$DB_PASSWORD"
fi

echo "Fixing NeuronDB extension conflicts..."

# Drop the conflicting function if it exists
echo "Checking for conflicting functions..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" <<EOF
-- Drop the conflicting hamming_distance function if it exists
DROP FUNCTION IF EXISTS public.hamming_distance(bit, bit) CASCADE;

-- Check for other potential conflicts
DO \$\$
DECLARE
    func_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO func_count
    FROM pg_proc p
    JOIN pg_namespace n ON p.pronamespace = n.oid
    WHERE n.nspname = 'public'
      AND p.proname = 'hamming_distance'
      AND pg_get_function_arguments(p.oid) = 'bit, bit';
    
    IF func_count > 0 THEN
        RAISE NOTICE 'Found % conflicting function(s)', func_count;
    ELSE
        RAISE NOTICE 'No conflicting functions found';
    END IF;
END \$\$;
EOF

echo ""
echo "Now you can create the extension:"
echo "  CREATE EXTENSION neurondb;"
echo ""
echo "Or run the setup script:"
echo "  ./scripts/setup_neurondb_ecosystem.sh"

