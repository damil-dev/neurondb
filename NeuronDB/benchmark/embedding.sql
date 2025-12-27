-- ============================================================================
-- NeuronDB pgbench Benchmark: Embedding Operations
-- ============================================================================
-- This file benchmarks embedding generation functions including:
-- - Text embedding generation
-- - Batch embedding generation
-- - Embedding caching
-- - Model configuration
-- ============================================================================

-- Setup: Random text samples for embedding
\set text_id random(1, 100)

-- Benchmark 1: Single Text Embedding
SELECT embed_text(
    CASE :text_id % 5
        WHEN 0 THEN 'The quick brown fox jumps over the lazy dog'
        WHEN 1 THEN 'Machine learning and artificial intelligence are transforming technology'
        WHEN 2 THEN 'PostgreSQL is a powerful open source relational database system'
        WHEN 3 THEN 'Vector embeddings enable semantic search and similarity matching'
        ELSE 'Neural networks process information through interconnected layers'
    END,
    'all-MiniLM-L6-v2'
) AS single_embedding;

-- Benchmark 2: Single Text Embedding (alias function)
SELECT neurondb_embed(
    CASE :text_id % 5
        WHEN 0 THEN 'The quick brown fox jumps over the lazy dog'
        WHEN 1 THEN 'Machine learning and artificial intelligence are transforming technology'
        WHEN 2 THEN 'PostgreSQL is a powerful open source relational database system'
        WHEN 3 THEN 'Vector embeddings enable semantic search and similarity matching'
        ELSE 'Neural networks process information through interconnected layers'
    END,
    'all-MiniLM-L6-v2'
) AS neurondb_embedding;

-- Benchmark 3: Cached Embedding (if caching is enabled)
SELECT embed_cached(
    CASE :text_id % 5
        WHEN 0 THEN 'The quick brown fox jumps over the lazy dog'
        WHEN 1 THEN 'Machine learning and artificial intelligence are transforming technology'
        WHEN 2 THEN 'PostgreSQL is a powerful open source relational database system'
        WHEN 3 THEN 'Vector embeddings enable semantic search and similarity matching'
        ELSE 'Neural networks process information through interconnected layers'
    END,
    'all-MiniLM-L6-v2'
) AS cached_embedding;

-- Benchmark 4: Batch Embedding (small batch)
SELECT embed_text_batch(
    ARRAY[
        CASE :text_id % 5 
            WHEN 0 THEN 'The quick brown fox jumps over the lazy dog'
            WHEN 1 THEN 'Machine learning and artificial intelligence are transforming technology'
            WHEN 2 THEN 'PostgreSQL is a powerful open source relational database system'
            WHEN 3 THEN 'Vector embeddings enable semantic search and similarity matching'
            ELSE 'Neural networks process information through interconnected layers'
        END,
        CASE (:text_id + 1) % 5 
            WHEN 0 THEN 'The quick brown fox jumps over the lazy dog'
            WHEN 1 THEN 'Machine learning and artificial intelligence are transforming technology'
            WHEN 2 THEN 'PostgreSQL is a powerful open source relational database system'
            WHEN 3 THEN 'Vector embeddings enable semantic search and similarity matching'
            ELSE 'Neural networks process information through interconnected layers'
        END,
        CASE (:text_id + 2) % 5 
            WHEN 0 THEN 'The quick brown fox jumps over the lazy dog'
            WHEN 1 THEN 'Machine learning and artificial intelligence are transforming technology'
            WHEN 2 THEN 'PostgreSQL is a powerful open source relational database system'
            WHEN 3 THEN 'Vector embeddings enable semantic search and similarity matching'
            ELSE 'Neural networks process information through interconnected layers'
        END
    ]::text[],
    'all-MiniLM-L6-v2'
) AS batch_embedding;

-- Benchmark 5: Batch Embedding (alias function)
SELECT neurondb_embed_batch(
    ARRAY[
        CASE :text_id % 5 
            WHEN 0 THEN 'The quick brown fox jumps over the lazy dog'
            WHEN 1 THEN 'Machine learning and artificial intelligence are transforming technology'
            WHEN 2 THEN 'PostgreSQL is a powerful open source relational database system'
            WHEN 3 THEN 'Vector embeddings enable semantic search and similarity matching'
            ELSE 'Neural networks process information through interconnected layers'
        END,
        CASE (:text_id + 1) % 5 
            WHEN 0 THEN 'The quick brown fox jumps over the lazy dog'
            WHEN 1 THEN 'Machine learning and artificial intelligence are transforming technology'
            WHEN 2 THEN 'PostgreSQL is a powerful open source relational database system'
            WHEN 3 THEN 'Vector embeddings enable semantic search and similarity matching'
            ELSE 'Neural networks process information through interconnected layers'
        END
    ]::text[],
    'all-MiniLM-L6-v2'
) AS neurondb_batch_embedding;

-- Benchmark 6: Embedding Dimension Check
SELECT vector_dims(embed_text(
    CASE :text_id % 5
        WHEN 0 THEN 'The quick brown fox jumps over the lazy dog'
        WHEN 1 THEN 'Machine learning and artificial intelligence are transforming technology'
        WHEN 2 THEN 'PostgreSQL is a powerful open source relational database system'
        WHEN 3 THEN 'Vector embeddings enable semantic search and similarity matching'
        ELSE 'Neural networks process information through interconnected layers'
    END,
    'all-MiniLM-L6-v2'
)) AS embedding_dims;

-- Benchmark 7: Embedding Consistency (same text, same embedding)
SELECT vector_l2_distance(
    embed_text('Consistency test text', 'all-MiniLM-L6-v2'),
    embed_text('Consistency test text', 'all-MiniLM-L6-v2')
) AS consistency_distance;

