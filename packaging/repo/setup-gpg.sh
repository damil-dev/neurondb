#!/bin/bash
set -e

# GPG key setup script for package repository signing
# Usage: ./setup-gpg.sh [KEY_NAME] [EMAIL]
# If KEY_NAME and EMAIL are not provided, will use existing key or prompt

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PACKAGING_DIR/.." && pwd)"

KEYS_DIR="$REPO_ROOT/repo/keys"
KEY_NAME="${1:-NeuronDB Package Signing Key}"
KEY_EMAIL="${2:-packages@neurondb.com}"

echo "=========================================="
echo "GPG Key Setup for Package Repository"
echo "=========================================="
echo ""

# Check for GPG
if ! command -v gpg &> /dev/null; then
    echo "Error: gpg not found. Please install gnupg package."
    exit 1
fi

# Create keys directory
mkdir -p "$KEYS_DIR"

# Check if key already exists
KEY_ID=""
EXISTING_KEYS=$(gpg --list-secret-keys --keyid-format LONG 2>/dev/null | grep -E "^sec" | head -1)

if [ -n "$EXISTING_KEYS" ]; then
    echo "Found existing GPG keys:"
    gpg --list-secret-keys --keyid-format LONG
    echo ""
    read -p "Use existing key? (y/n) [y]: " USE_EXISTING
    USE_EXISTING=${USE_EXISTING:-y}
    
    if [ "$USE_EXISTING" = "y" ] || [ "$USE_EXISTING" = "Y" ]; then
        # Extract key ID from first key
        KEY_ID=$(gpg --list-secret-keys --keyid-format LONG 2>/dev/null | grep -E "^sec" | head -1 | awk '{print $2}' | cut -d'/' -f2)
        echo "Using existing key: $KEY_ID"
    fi
fi

# Generate new key if needed
if [ -z "$KEY_ID" ]; then
    echo "Generating new GPG key..."
    echo "Key Name: $KEY_NAME"
    echo "Email: $KEY_EMAIL"
    echo ""
    
    # Create batch key generation file
    BATCH_FILE=$(mktemp)
    cat > "$BATCH_FILE" <<EOF
Key-Type: RSA
Key-Length: 4096
Subkey-Type: RSA
Subkey-Length: 4096
Name-Real: $KEY_NAME
Name-Email: $KEY_EMAIL
Expire-Date: 0
EOF
    
    # Generate key (non-interactive)
    gpg --batch --generate-key "$BATCH_FILE" 2>&1 | grep -v "gpg:"
    rm -f "$BATCH_FILE"
    
    # Get the new key ID
    KEY_ID=$(gpg --list-secret-keys --keyid-format LONG 2>/dev/null | grep -E "^sec" | head -1 | awk '{print $2}' | cut -d'/' -f2)
    
    if [ -z "$KEY_ID" ]; then
        echo "Error: Failed to generate or find GPG key"
        exit 1
    fi
    
    echo "Generated new key: $KEY_ID"
fi

# Display key information
echo ""
echo "Key Information:"
gpg --list-secret-keys --keyid-format LONG "$KEY_ID"
echo ""

# Export public key for DEB (ASCII armored)
echo "Exporting public key for DEB repository..."
gpg --armor --export "$KEY_ID" > "$KEYS_DIR/neurondb.gpg"
echo "Exported: $KEYS_DIR/neurondb.gpg"

# Export public key for RPM (ASCII armored, different name)
echo "Exporting public key for RPM repository..."
gpg --armor --export "$KEY_ID" > "$KEYS_DIR/RPM-GPG-KEY-neurondb"
echo "Exported: $KEYS_DIR/RPM-GPG-KEY-neurondb"

# Export private key for GitHub Secrets (base64 encoded)
echo ""
echo "=========================================="
echo "GitHub Secrets Setup"
echo "=========================================="
echo ""
echo "To use this key in GitHub Actions, add the following secrets:"
echo ""
echo "1. GPG_PRIVATE_KEY:"
echo "   Run the following command and copy the output:"
echo "   gpg --armor --export-secret-keys $KEY_ID | base64 -w 0"
echo ""
echo "2. GPG_KEY_ID:"
echo "   $KEY_ID"
echo ""
echo "3. (Optional) GPG_PASSPHRASE:"
echo "   If your key has a passphrase, add it as a secret"
echo ""

# Get key fingerprint
FINGERPRINT=$(gpg --fingerprint "$KEY_ID" 2>/dev/null | grep -E "^      " | head -1 | sed 's/^      //')
echo "Key Fingerprint: $FINGERPRINT"
echo ""

# Display public key info
echo "Public keys exported to:"
echo "  DEB: $KEYS_DIR/neurondb.gpg"
echo "  RPM: $KEYS_DIR/RPM-GPG-KEY-neurondb"
echo ""
echo "These keys will be available at:"
echo "  https://USERNAME.github.io/neurondb/repo/keys/neurondb.gpg"
echo "  https://USERNAME.github.io/neurondb/repo/keys/RPM-GPG-KEY-neurondb"
echo ""

