# NeuronMCP Python Environment

## Quick Start

```bash
cd NeuronMCP

# Setup (first time only)
./run_with_venv.sh setup

# Run NeuronMCP with Python environment
./run_with_venv.sh
```

## Manual Setup

```bash
cd NeuronMCP

# Create virtual environment
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify
python3 -c "import psycopg2, pandas, datasets; print('✓ Ready!')"
```

## Requirements

The `requirements.txt` includes all dependencies for:
- **dataset_loader.py**: HuggingFace dataset loading
- **PostgreSQL connectivity**: Database operations
- **Data processing**: Pandas, NumPy
- **Cloud storage**: S3, URLs support

## Running NeuronMCP

### Option 1: Use Helper Script (Recommended)
```bash
./run_with_venv.sh
```

### Option 2: Manual Activation
```bash
source .venv/bin/activate
./run_mcp_server.sh
```

### Option 3: Set PYTHON Variable
```bash
source .venv/bin/activate
export PYTHON=$(which python3)
./run_mcp_server.sh
```

## Testing Dataset Loading

```bash
source .venv/bin/activate

# Test with a small dataset
python3 internal/tools/dataset_loader.py \
  --source-type huggingface \
  --source-path "squad" \
  --split "train" \
  --limit 10 \
  --no-auto-embed
```

## Installed Packages

- psycopg2-binary (PostgreSQL)
- pandas (Data processing)
- datasets (HuggingFace)
- huggingface-hub (HF Hub client)
- boto3 (S3 support)
- requests (HTTP/URL support)
- pyarrow (Parquet support)

See `requirements-lock.txt` for exact versions.


