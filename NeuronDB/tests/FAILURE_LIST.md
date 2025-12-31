# Failure List (Regenerated)

Track known failures that are not yet fixed.

## Template entry

- Failure:
- Where (test file):
- Environment:
- Status:
- Notes:

=== TEST FAILURE LIST ===

CPU MODE (10 failures):
- 016_vector_embeddings_text.sql
- 018_vector_embeddings_config.sql
- 024_vector_ivfflat_vector.sql
- 028_vector_ivfflat_halfvec.sql
- 029_vector_ivfflat_bit.sql
- 059_rag_rag.sql
- 061_pgvector_ivfflat_vector.sql
- 063_gpu_search.sql
- 073_vector_ivfflat_vector.sql
- 075_vector_hnsw_sparsevec.sql

GPU MODE (13 failures, 3 crashes):
- 016_vector_embeddings_text.sql
- 018_vector_embeddings_config.sql
- 024_vector_ivfflat_vector.sql
- 028_vector_ivfflat_halfvec.sql
- 029_vector_ivfflat_bit.sql
- 043_ml_knn.sql
- 055_ml_automl.sql
- 059_rag_rag.sql
- 061_pgvector_ivfflat_vector.sql
- 063_gpu_search.sql
- 073_vector_ivfflat_vector.sql
- 075_vector_hnsw_sparsevec.sql
- Additional GPU-specific crashes

AUTO MODE (14 failures, 4 crashes):
- 016_vector_embeddings_text.sql
- 018_vector_embeddings_config.sql
- 024_vector_ivfflat_vector.sql
- 028_vector_ivfflat_halfvec.sql
- 029_vector_ivfflat_bit.sql
- 043_ml_knn.sql
- 055_ml_automl.sql
- 059_rag_rag.sql
- 061_pgvector_ivfflat_vector.sql
- 063_gpu_search.sql
- 073_vector_ivfflat_vector.sql
- 075_vector_hnsw_sparsevec.sql
- Additional AUTO-specific crashes

COMMON FAILURES (all 3 modes):
- 016_vector_embeddings_text.sql
- 018_vector_embeddings_config.sql
- 024_vector_ivfflat_vector.sql
- 028_vector_ivfflat_halfvec.sql
- 029_vector_ivfflat_bit.sql
- 059_rag_rag.sql
- 061_pgvector_ivfflat_vector.sql
- 063_gpu_search.sql
- 073_vector_ivfflat_vector.sql
- 075_vector_hnsw_sparsevec.sql

GPU/AUTO ONLY:
- 043_ml_knn.sql
- 055_ml_automl.sql
