#!/bin/bash
# Setup core dump generation for debugging PostgreSQL crashes
# Configures core dumps to be saved in /tmp/core/

set -e

echo "Setting up core dump configuration..."

# Set core dump file size limit to unlimited
ulimit -c unlimited
echo "✓ Set ulimit -c unlimited"

# Create core dump directory
CORE_DIR="/tmp/core"
mkdir -p "$CORE_DIR"
echo "✓ Created directory: $CORE_DIR"

# Set kernel core dump pattern to save cores in /tmp/core/
# Format: core.<executable>.<pid>.<timestamp>
echo "Setting core dump pattern to $CORE_DIR/core.%e.%p.%t..."
if echo "$CORE_DIR/core.%e.%p.%t" | sudo tee /proc/sys/kernel/core_pattern > /dev/null; then
    echo "✓ Configured kernel core_pattern"
else
    echo "✗ Failed to set core_pattern (may need sudo privileges)"
    exit 1
fi

# Verify core dump settings
echo ""
echo "Core dump configuration:"
echo "  Core limit: $(ulimit -c)"
echo "  Core pattern: $(cat /proc/sys/kernel/core_pattern)"
echo "  Core directory: $CORE_DIR"
echo ""
echo "Core dump setup complete!"
echo "Core dumps will be saved to: $CORE_DIR/core.*"
echo ""

