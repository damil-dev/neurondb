# NeuronDB Scripts Directory

This directory contains Python scripts used by NeuronDB for various ML operations.

## Scripts

### `train_postgres_llm.py`
Trains a custom transformer language model for PostgreSQL-specific tasks.

**Usage:**
```bash
python3 train_postgres_llm.py \
  --corpus-file /path/to/corpus.txt \
  --output-dir /path/to/output \
  --epochs 5 \
  --batch-size 4 \
  --learning-rate 0.001 \
  --d-model 256 \
  --nhead 4 \
  --num-layers 4 \
  --dim-feedforward 512 \
  --max-seq-length 256 \
  --vocab-size 1000
```

**Called by:**
- `neurondb.train()` function when using `transformer_llm` or `custom_llm` algorithms

**Requirements:**
- Python 3.7+
- PyTorch
- tqdm

## Environment Variables

The C extension looks for scripts in this order:
1. `NEURONDB_TOOLS_DIR` - Custom tools directory (highest priority)
2. `NEURONDB_SHARE_DIR/../scripts` - Relative to share directory
3. `/home/pge/pge/neurondb/scripts` - Default fallback

## Installation

Scripts should be installed alongside the NeuronDB extension. The installation process will copy these scripts to the appropriate location based on your PostgreSQL installation.

