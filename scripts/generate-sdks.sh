#!/bin/bash
# Generate Python and TypeScript SDKs from OpenAPI specifications

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Generating SDKs from OpenAPI specifications..."

# Check for required tools
command -v openapi-generator-cli >/dev/null 2>&1 || {
    echo "Error: openapi-generator-cli not found. Install with: npm install -g @openapitools/openapi-generator-cli"
    exit 1
}

# Create output directories
PYTHON_SDK_DIR="$REPO_ROOT/sdks/python"
TYPESCRIPT_SDK_DIR="$REPO_ROOT/sdks/typescript"

mkdir -p "$PYTHON_SDK_DIR"
mkdir -p "$TYPESCRIPT_SDK_DIR"

# Generate Python SDK for NeuronAgent
echo "Generating Python SDK for NeuronAgent..."
openapi-generator-cli generate \
    -i "$REPO_ROOT/NeuronAgent/openapi/openapi.yaml" \
    -g python \
    -o "$PYTHON_SDK_DIR/neuronagent" \
    --package-name neuronagent \
    --additional-properties=packageVersion=1.0.0,packageAuthor=neurondb,packageAuthorEmail=support@neurondb.ai

# Generate TypeScript SDK for NeuronAgent
echo "Generating TypeScript SDK for NeuronAgent..."
openapi-generator-cli generate \
    -i "$REPO_ROOT/NeuronAgent/openapi/openapi.yaml" \
    -g typescript-axios \
    -o "$TYPESCRIPT_SDK_DIR/neuronagent" \
    --additional-properties=npmName=@neurondb/neuronagent,npmVersion=1.0.0

# Generate TypeScript SDK for NeuronDesktop (if OpenAPI spec exists)
if [ -f "$REPO_ROOT/NeuronDesktop/api/openapi.yaml" ]; then
    echo "Generating TypeScript SDK for NeuronDesktop..."
    openapi-generator-cli generate \
        -i "$REPO_ROOT/NeuronDesktop/api/openapi.yaml" \
        -g typescript-axios \
        -o "$TYPESCRIPT_SDK_DIR/neurondesktop" \
        --additional-properties=npmName=@neurondb/neurondesktop,npmVersion=1.0.0
fi

echo "SDK generation complete!"
echo "Python SDK: $PYTHON_SDK_DIR"
echo "TypeScript SDK: $TYPESCRIPT_SDK_DIR"

