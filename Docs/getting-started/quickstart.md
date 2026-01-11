# Quick Start Guide

**Get started with NeuronDB in minutes!**

This guide walks you through the fastest path from zero to your first working vector search query using NeuronDB's developer tools.

## Prerequisites

- **NeuronDB installed**: See [Installation Guide](installation.md) for setup instructions
- **PostgreSQL client**: `psql` (or any SQL client)
- **5-10 minutes**: For complete quickstart

> **New here?** Start with **[Simple Start Guide](simple-start.md)** for a beginner-friendly walkthrough.
>
> **Ecosystem Setup?** For complete NeuronDB ecosystem (Docker, all components), see the [root QUICKSTART.md](../../QUICKSTART.md).

## Step 1: Install NeuronDB

If you haven't installed NeuronDB yet, choose your method:

### Option A: Docker Compose (Recommended for Quick Start)

```bash
# From repository root
docker compose up -d neurondb

# Wait for service to be healthy
docker compose ps neurondb
```

### Option B: Native Installation

Follow the [Installation Guide](installation.md) for native PostgreSQL installation.

### Verify Installation

```bash
# With Docker Compose
docker compose exec neurondb psql -U neurondb -d neurondb -c "CREATE EXTENSION IF NOT EXISTS neurondb;"

# Or with native PostgreSQL
psql -d your_database -c "CREATE EXTENSION IF NOT EXISTS neurondb;"
```

## Step 2: Load Quickstart Data Pack

The quickstart data pack provides ~500 sample documents with pre-generated embeddings, ready for immediate use.

### Option 1: Using the CLI (Recommended)

```bash
# From repository root
./scripts/neurondb-cli.sh quickstart
```

### Option 2: Using the Loader Script

```bash
# From repository root
./examples/quickstart/load_quickstart.sh
```

### Option 3: Using psql Directly

```bash
# With Docker Compose
docker compose exec neurondb psql -U neurondb -d neurondb -f examples/quickstart/quickstart_data.sql

# Or with native PostgreSQL
psql -d your_database -f examples/quickstart/quickstart_data.sql
```

**What gets created:**
- Table: `quickstart_documents` with ~500 documents
- Index: HNSW index on embeddings
- Ready to query: Data is immediately usable

**Verify data loaded:**

```bash
# Count documents
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -c "SELECT COUNT(*) FROM quickstart_documents;"
```

## Step 3: Try SQL Recipes

The SQL recipe library provides ready-to-run queries for common operations.

### Quick Example: Basic Similarity Search

```sql
-- Find documents similar to document #1
SELECT 
    id,
    title,
    embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1) AS distance
FROM quickstart_documents
WHERE id != 1
ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)
LIMIT 10;
```

### Explore Recipe Categories

1. **[Vector Search](recipes/01_vector_search.sql)** - Basic similarity search patterns
2. **[Hybrid Search](recipes/02_hybrid_search.sql)** - Vector + full-text search
3. **[Filtered Search](recipes/03_filtered_search.sql)** - Vector search with SQL filters
4. **[Indexing](recipes/04_indexing.sql)** - Index creation patterns
5. **[Embedding Generation](recipes/05_embedding_generation.sql)** - Generate embeddings from text

**Try a recipe:**

```bash
# Run a recipe file
psql "postgresql://neurondb:neurondb@localhost:5433/neurondb" -f Docs/getting-started/recipes/01_vector_search.sql

# Or copy individual queries from recipe files
```

See the [Recipe Library README](recipes/README.md) for complete documentation.

## Step 4: Use CLI Helpers

The CLI helpers simplify common tasks like index creation and data loading.

### Index Management

```bash
# Create HNSW index with default parameters
./scripts/neurondb-cli.sh index create documents embedding

# Create HNSW index with custom parameters
./scripts/neurondb-cli.sh index create documents embedding --type hnsw --m 16 --ef_construction 200

# List all indexes
./scripts/neurondb-cli.sh index list

# Drop an index
./scripts/neurondb-cli.sh index drop idx_documents_hnsw
```

### Quickstart Data

```bash
# Load quickstart data pack
./scripts/neurondb-cli.sh quickstart
```

### Connection Options

The CLI auto-detects Docker Compose setup, or you can specify connection:

```bash
# Use custom connection
./scripts/neurondb-cli.sh index list -h localhost -p 5432 -d mydb -U postgres
```

See `./scripts/neurondb-cli.sh --help` for complete usage.

## Next Steps

Now that you have NeuronDB running with sample data, try these:

### 1. Explore SQL Recipes

- Try different similarity search patterns
- Experiment with hybrid search
- Use filtered search for category-based queries

See: [SQL Recipe Library](recipes/README.md)

### 2. Load Your Own Data

- Create your own table
- Generate embeddings from your text
- Create indexes for fast search

See: [Embedding Generation Recipes](recipes/05_embedding_generation.sql) and [Indexing Recipes](recipes/04_indexing.sql)

### 3. Try Python Examples

Explore the Python examples for more advanced use cases:

- **[Basics Examples](../../examples/basics/)** - Simple Python examples
- **[Semantic Search](../../examples/semantic-search-docs/)** - Document search system
- **[RAG Chatbot](../../examples/rag-chatbot-pdfs/)** - RAG with PDFs

### 4. Read the Documentation

- **[Vector Search Guide](../../NeuronDB/docs/vector-search/)** - Complete vector search documentation
- **[Hybrid Search Guide](../../NeuronDB/docs/hybrid-search/)** - Combining search methods
- **[API Reference](../../NeuronDB/docs/sql-api.md)** - SQL function reference

## Common Use Cases

### Use Case: Build a Recommendation System

1. Load quickstart data (Step 2)
2. Try similarity search recipes (Step 3)
3. Create indexes with CLI helpers (Step 4)
4. Adapt for your data

### Use Case: Semantic Search

1. Load quickstart data
2. Try hybrid search recipes
3. Generate embeddings for your documents
4. Create indexes and query

### Use Case: Ingest New Documents

1. Create your table
2. Generate embeddings (see embedding recipes)
3. Create indexes (use CLI helpers)
4. Query with vector search

## Troubleshooting

### "Extension neurondb does not exist"

**Solution**: Install NeuronDB extension first:
```sql
CREATE EXTENSION IF NOT EXISTS neurondb;
```

### "Table quickstart_documents does not exist"

**Solution**: Load the quickstart data pack (Step 2):
```bash
./scripts/neurondb-cli.sh quickstart
```

### "Connection refused"

**Docker Compose:**
```bash
# Check service status
docker compose ps neurondb

# Start if needed
docker compose up -d neurondb
```

**Native PostgreSQL:**
```bash
# Check if PostgreSQL is running
psql -h localhost -p 5432 -U postgres -c "SELECT 1;"
```

### CLI script not found

**Solution**: Make sure you're in the repository root:
```bash
cd /path/to/neurondb2
./scripts/neurondb-cli.sh --help
```

### Recipes don't work

**Solution**: 
1. Make sure quickstart data is loaded
2. Check you're using the correct table name (default: `quickstart_documents`)
3. Adjust table/column names in recipes to match your schema

## Quick Reference

### Quickstart Commands

```bash
# Load quickstart data
./scripts/neurondb-cli.sh quickstart

# Or use loader script
./examples/quickstart/load_quickstart.sh

# Create index
./scripts/neurondb-cli.sh index create table_name column_name

# List indexes
./scripts/neurondb-cli.sh index list
```

### Common SQL Queries

```sql
-- View sample documents
SELECT id, title, LEFT(content, 50) AS preview 
FROM quickstart_documents LIMIT 5;

-- Basic similarity search
SELECT id, title,
       embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1) AS distance
FROM quickstart_documents
WHERE id != 1
ORDER BY embedding <=> (SELECT embedding FROM quickstart_documents WHERE id = 1)
LIMIT 10;

-- Get statistics
SELECT COUNT(*), pg_size_pretty(pg_total_relation_size('quickstart_documents'))
FROM quickstart_documents;
```

## Related Resources

- **[Quickstart Data Pack](../../examples/quickstart/)** - Sample dataset documentation
- **[SQL Recipe Library](recipes/)** - Ready-to-run queries
- **[Simple Start Guide](simple-start.md)** - Beginner-friendly walkthrough
- **[Installation Guide](installation.md)** - Detailed installation options
- **[Ecosystem Quickstart](../../QUICKSTART.md)** - Complete ecosystem setup

## Support

- **Documentation**: [neurondb.ai/docs](https://neurondb.ai/docs)
- **Issues**: [GitHub Issues](https://github.com/neurondb/neurondb/issues)
- **Quickstart Data**: [examples/quickstart/](../../examples/quickstart/)
- **Recipes**: [Docs/getting-started/recipes/](recipes/)

---

**Happy Learning! 🚀**
