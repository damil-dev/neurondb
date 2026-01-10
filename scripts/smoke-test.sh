#!/bin/bash
#
# Backwards-compatible wrapper.
# `smoke-test.sh` was renamed to `health-check.sh`.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "smoke-test.sh was renamed to health-check.sh; running health-check.sh..."
exec "${SCRIPT_DIR}/health-check.sh" "$@"





