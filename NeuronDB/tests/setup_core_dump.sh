#!/bin/bash
# Setup core dump generation for debugging PostgreSQL crashes

# Set core dump file size limit to unlimited
ulimit -c unlimited

# Create core dump directory if it doesn't exist
CORE_DIR="$HOME/pge/neurondb/NeuronDB/tests/core_dumps"
mkdir -p "$CORE_DIR"

# Set kernel core dump pattern to save cores with process name and timestamp
# Format: core.<executable>.<pid>.<timestamp>
echo "Setting core dump pattern..."
echo "core.%e.%p.%t" | sudo tee /proc/sys/kernel/core_pattern

# Alternative: Save to specific directory (uncomment if preferred)
# echo "$CORE_DIR/core.%e.%p.%t" | sudo tee /proc/sys/kernel/core_pattern

# Verify core dump settings
echo ""
echo "Core dump configuration:"
echo "  Core limit: $(ulimit -c)"
echo "  Core pattern: $(cat /proc/sys/kernel/core_pattern)"
echo "  Core directory: $CORE_DIR"
echo ""
echo "To generate core dump, run the test command below."
echo "After crash, core file will be saved as: core.postgres.<pid>.<timestamp>"
echo ""



