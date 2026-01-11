# RAG Testing with Sample Data

This script demonstrates how to set up sample data, generate embeddings, and test RAG (Retrieval-Augmented Generation) using NeuronMCP tools. It works through the full chain: **ClaudeDesktop -> NeuronMCP -> NeuronDB**.

## Overview

The script `test_rag_sample_data.py` performs the following steps:

1. **Creates tables** - Sets up `rag_test_documents` table with vector column
2. **Inserts sample data** - Adds 5 sample technical documents
3. **Generates embeddings** - Creates vector embeddings using NeuronDB's embedding function
4. **Tests RAG retrieval** - Uses `retrieve_context` tool to test semantic search

## Prerequisites

- NeuronDB database running (PostgreSQL 16+ with NeuronDB extension)
- NeuronMCP server configured and accessible
- Python 3.8+ with required dependencies
- MCP client library (included in `client/` directory)

## Usage

### Basic Usage

```bash
cd NeuronMCP
python test_rag_sample_data.py
```

### With Custom Config

```bash
python test_rag_sample_data.py --config /path/to/neuronmcp_server.json
```

### Skip Steps (if data already exists)

```bash
# Skip table creation and data insertion
python test_rag_sample_data.py --skip-setup

# Skip embedding generation (if embeddings already exist)
python test_rag_sample_data.py --skip-embeddings
```

## Configuration

The script uses the NeuronMCP server configuration file (default: `neuronmcp_server.json`). Make sure your configuration includes:

```json
{
  "mcpServers": {
    "neurondb": {
      "command": "/path/to/neurondb-mcp",
      "env": {
        "NEURONDB_HOST": "localhost",
        "NEURONDB_PORT": "5433",
        "NEURONDB_DATABASE": "neurondb",
        "NEURONDB_USER": "your_user",
        "NEURONDB_PASSWORD": "your_password"
      }
    }
  }
}
```

## Using Through Claude Desktop

This script demonstrates the workflow that Claude Desktop uses when interacting with NeuronMCP. The key tools used are:

1. **`postgresql_execute_query`** - For creating tables and inserting data
2. **SQL embedding functions** - For generating embeddings (via SQL)
3. **`retrieve_context`** - For RAG retrieval testing

### Manual Steps in Claude Desktop

You can also perform these steps manually through Claude Desktop:

1. **Create table:**
   ```
   Use the postgresql_execute_query tool to create the rag_test_documents table
   ```

2. **Insert sample data:**
   ```
   Use postgresql_execute_query to insert sample documents
   ```

3. **Generate embeddings:**
   ```
   Use postgresql_execute_query with UPDATE statement calling neurondb_generate_embedding()
   ```

4. **Test RAG:**
   ```
   Use the retrieve_context tool with your query
   ```

## Sample Data

The script creates 5 sample documents:

1. PostgreSQL Performance Tuning
2. Vector Databases Explained
3. Retrieval-Augmented Generation Overview
4. Python Machine Learning Best Practices
5. Database Sharding Strategies

## Output

The script provides detailed output for each step:

- ✅ Success indicators for completed steps
- ✗ Error messages if something fails
- ⚠️ Warnings for non-critical issues
- Query results showing retrieved context

## Troubleshooting

### Connection Errors

- Verify NeuronMCP server is running
- Check database connection settings in config
- Ensure PostgreSQL is accessible

### Embedding Generation Fails

- Verify NeuronDB extension is installed
- Check that embedding models are available
- Ensure sufficient database permissions

### RAG Retrieval Returns No Results

- Verify embeddings were generated successfully
- Check that table name and column names are correct
- Ensure vector index exists (should be created automatically)

## Next Steps

After running this script, you can:

- Modify sample data to use your own documents
- Experiment with different embedding models
- Try different RAG queries
- Integrate with LLM for answer generation
- Scale up to larger document collections

## Related Tools

- `retrieve_context` - Retrieve relevant context for queries
- `process_document` - Process and chunk documents
- `generate_response` - Generate responses using retrieved context
- `vector_search` - Direct vector similarity search
- `batch_embedding` - Generate embeddings for multiple texts

## See Also

- `examples/semantic-search-docs/` - Semantic search example
- `examples/rag-chatbot-pdfs/` - RAG chatbot example
- `NeuronDB/demo/RAG/` - SQL-based RAG demos
- `TOOLS_REFERENCE.md` - Complete tool reference

