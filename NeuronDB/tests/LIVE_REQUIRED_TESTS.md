# Live Required Tests (Regenerated)

Some tests may require external connectivity or access to runtime services/models (for example, “live embeddings” or external model endpoints).

## Guidance

- Mark tests clearly when they are not hermetic.
- Provide a CPU-only fallback test when possible.
- Make “live required” suites opt-in for CI.

# Live-Required Tests for Embeddings, RAG, and Vector Search

This document explains the live-required tests that need an API key to run successfully.

## Overview

Several test files in `tests/sql/basic/` require a configured Hugging Face API key to execute real embedding generation and LLM completions. These tests will **fail fast** with a clear error message if the API key is not configured, ensuring that CI/CD pipelines and developers know exactly what's needed.

## Test Files

The following test files are live-required:

- **078_embedding_live_required.sql** - Embedding quality tests (dimensions, norms, semantic ordering)
- **079_embedding_cache_live_required.sql** - Embedding cache functionality and access counting
- **080_vector_index_semantics.sql** - Vector search with HNSW/IVFFlat indexes and ranking
- **081_hybrid_search_filters_and_weights.sql** - Hybrid search with weights, filters, and query types
- **082_rag_end_to_end_live_required.sql** - Complete RAG pipeline (retrieval → answer generation)

## Configuration

### Required Settings

These tests require the following PostgreSQL configuration parameters to be set:

```sql
-- Set your Hugging Face API key
SET neurondb.llm_api_key = 'your-huggingface-api-key-here';

-- Ensure provider is set to Hugging Face (default, but explicit is better)
SET neurondb.llm_provider = 'huggingface';

-- Disable fail-open mode (tests will fail if API is unavailable)
SET neurondb.llm_fail_open = off;
```

### Getting a Hugging Face API Key

1. Sign up or log in to [Hugging Face](https://huggingface.co/)
2. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Create a new token with "Read" permissions
4. Copy the token and set it as `neurondb.llm_api_key`

### Setting Configuration

#### Option 1: Session-Level (Recommended for Testing)

```sql
-- Connect to your database and set for current session
SET neurondb.llm_api_key = 'hf_...';
SET neurondb.llm_provider = 'huggingface';
SET neurondb.llm_fail_open = off;
```

#### Option 2: Database-Level (Persistent)

```sql
-- Set at database level (persists across sessions)
ALTER DATABASE neurondb SET neurondb.llm_api_key = 'hf_...';
ALTER DATABASE neurondb SET neurondb.llm_provider = 'huggingface';
ALTER DATABASE neurondb SET neurondb.llm_fail_open = off;
```

#### Option 3: System-Level (All Databases)

```sql
-- Requires superuser, persists across databases
ALTER SYSTEM SET neurondb.llm_api_key = 'hf_...';
ALTER SYSTEM SET neurondb.llm_provider = 'huggingface';
ALTER SYSTEM SET neurondb.llm_fail_open = off;

-- Then reload configuration
SELECT pg_reload_conf();
```

#### Option 4: Environment Variable (PostgreSQL Startup)

Add to your PostgreSQL environment:

```bash
export NDB_LLM_API_KEY='hf_...'
export NDB_LLM_PROVIDER='huggingface'
export NDB_LLM_FAIL_OPEN='off'
```

## Running the Tests

### Using run_test.py

```bash
# Run all live-required tests
python3 tests/run_test.py --category basic --module embedding
python3 tests/run_test.py --category basic --module rag
python3 tests/run_test.py --category basic --module vector

# Run specific test file
python3 tests/run_test.py --category basic --test 078_embedding_live_required

# Run all basic tests (will fail on live-required if key not set)
python3 tests/run_test.py --category basic
```

### Using psql Directly

```bash
# Set API key first
psql -d neurondb -c "SET neurondb.llm_api_key = 'hf_...';"

# Then run the test
psql -d neurondb -f tests/sql/basic/078_embedding_live_required.sql
```

## What These Tests Validate

### 078_embedding_live_required.sql
- ✅ Embedding dimensions (expects 384 for `all-MiniLM-L6-v2`)
- ✅ Non-zero embedding norms for non-empty text
- ✅ Semantic ordering (relevant docs closer than irrelevant ones)
- ✅ Batch embedding consistency
- ✅ Different texts produce different embeddings

### 079_embedding_cache_live_required.sql
- ✅ Cache population after embedding generation
- ✅ Cache access count increment on repeated calls
- ✅ Cache key uniqueness for different texts
- ✅ Cache entry metadata (created_at, last_accessed)

### 080_vector_index_semantics.sql
- ✅ Basic vector search without index (brute force)
- ✅ Top-K retrieval with stable ordering
- ✅ HNSW index creation and search
- ✅ IVFFlat index creation and search
- ✅ Distance threshold filtering

### 081_hybrid_search_filters_and_weights.sql
- ✅ Pure vector search (vector_weight = 1.0)
- ✅ Pure FTS search (vector_weight = 0.0)
- ✅ Metadata filters excluding non-matching documents
- ✅ Query type variations (plain, phrase, to_tsquery)
- ✅ Balanced hybrid search (vector_weight = 0.5)
- ✅ Combined filters and weights

### 082_rag_end_to_end_live_required.sql
- ✅ Context retrieval using `neurondb_retrieve_context_c`
- ✅ Answer generation using `neurondb_generate_answer`
- ✅ Complete RAG pipeline (retrieval → answer)
- ✅ Multiple queries across different topics

## Error Handling

If the API key is not configured, tests will fail immediately with a clear message:

```
ERROR: This test requires neurondb.llm_api_key to be set. 
Please configure your Hugging Face API key: 
SET neurondb.llm_api_key = 'your-key-here';
```

This ensures that:
- CI/CD pipelines can detect missing configuration early
- Developers know exactly what's needed to run the tests
- Tests don't silently pass with fallback behavior

## Differences from Other Tests

Other embedding/RAG tests (like `059_rag_rag.sql`, `060_rag_hybrid_search.sql`) use `SET neurondb.llm_fail_open = on` to allow graceful fallback with zero vectors when the API key is not configured. These live-required tests explicitly **require** the API key to ensure real behavior is tested.

## CI/CD Considerations

For CI/CD pipelines:

1. **Store API key as secret** (GitHub Secrets, GitLab CI Variables, etc.)
2. **Set it before running tests**:
   ```bash
   psql -d neurondb -c "SET neurondb.llm_api_key = '$HF_API_KEY';"
   python3 tests/run_test.py --category basic --module embedding
   ```
3. **Consider rate limiting** - Hugging Face API has rate limits, so batch test runs may need delays
4. **Monitor API usage** - These tests make real API calls, which may incur costs if using paid tiers

## Troubleshooting

### "This test requires neurondb.llm_api_key to be set"
- Ensure you've set the API key using one of the methods above
- Check that the key is set in the current session: `SHOW neurondb.llm_api_key;`

### "embedding generation failed"
- Verify your API key is valid
- Check Hugging Face API status
- Ensure you have internet connectivity
- Check rate limits haven't been exceeded

### "HNSW/IVFFlat index creation skipped"
- Some index types may not be available in all builds
- This is a warning, not an error - tests will continue with fallback behavior

## Related Documentation

- [Embedding Generation](../docs/ml-embeddings/embedding-generation.md)
- [Hybrid Search](../docs/hybrid-search/overview.md)
- [RAG Pipeline](../docs/rag/overview.md)
- [Configuration](../docs/configuration.md)

