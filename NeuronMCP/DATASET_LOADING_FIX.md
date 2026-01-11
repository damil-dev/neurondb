# Dataset Loading System Fix

## Issue Analysis

The HuggingFace dataset loading from CloudeDesktop is failing because:

1. **Parameter Mapping**: The tool accepts both `source_path` and `dataset_name` (for backward compatibility), but the script path finding might fail
2. **Script Path Resolution**: The `findDatasetLoaderScript()` function needs to be more robust in finding the Python script
3. **Error Handling**: Better logging needed when script is not found

## Changes Made

### 1. Enhanced Script Path Finding (`dataset_loading.go`)

- Added more search paths including executable-relative paths
- Added common installation location checks
- Better error logging when script is not found

### 2. Improved Parameter Handling

- Better support for `dataset_name` parameter (backward compatibility)
- Clearer conversion from `dataset_name` to `source_path`

### 3. Better Logging

- Log when script is found and being used
- Warn when falling back to inline Python code

## How It Works

1. **Tool Call**: CloudeDesktop calls `load_dataset` tool via MCP protocol
2. **Parameter Processing**: Tool accepts `source_path` or `dataset_name` (converts to `source_path`)
3. **Script Discovery**: Finds `dataset_loader.py` in multiple possible locations
4. **Python Execution**: Executes the Python script with proper environment variables
5. **Result Parsing**: Parses JSON output from Python script and returns to CloudeDesktop

## Testing

To test the fix:

```bash
# From CloudeDesktop, call:
{
  "name": "load_dataset",
  "arguments": {
    "source_path": "imdb",
    "source_type": "huggingface",
    "split": "train",
    "limit": 100
  }
}
```

Or with backward compatibility:
```bash
{
  "name": "load_dataset",
  "arguments": {
    "dataset_name": "imdb",
    "split": "train",
    "limit": 100
  }
}
```

## Requirements

The Python script requires:
- `psycopg2-binary` - PostgreSQL adapter
- `pandas` - Data manipulation
- `datasets` - HuggingFace datasets library

Install with:
```bash
pip install psycopg2-binary pandas datasets
```

## Next Steps

1. Ensure Python dependencies are installed where NeuronMCP runs
2. Verify the script path is found correctly
3. Test with a small dataset first (e.g., `imdb` with `limit: 10`)

