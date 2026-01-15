/*-------------------------------------------------------------------------
 *
 * pq_scan.c
 *    PQ index scan implementation with two-stage retrieval
 *
 * Implements:
 * - Stage 1: Coarse search using PQ-encoded vectors (GPU-accelerated)
 * - Stage 2: Fine rerank with full-precision vectors
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/scan/pq_scan.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_types.h"
#include "fmgr.h"
#include "access/relscan.h"
#include "utils/rel.h"
#include "storage/bufmgr.h"
#include "utils/builtins.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_gpu.h"
#include "gpu_pq_retrieval.h"

/*
 * PQ scan state
 */
typedef struct PqScanState
{
	/* Query parameters */
	const float *query;
	int			dim;
	int			k;
	int			rerank_k;

	/* PQ parameters */
	int			m;				/* Number of subspaces */
	int			ks;				/* Codebook size */
	int			subspace_dim;

	/* Stage 1: Coarse search results */
	uint32_t   *coarse_indices;
	float	   *coarse_distances;
	int			coarse_count;

	/* Stage 2: Fine rerank results */
	uint32_t   *fine_indices;
	float	   *fine_distances;
	int			fine_count;

	/* Current result position */
	int			current_pos;
}			PqScanState;

/*
 * Initialize PQ scan
 */
static PqScanState *
pq_scan_init(const float *query, int dim, int k, int rerank_k, int m, int ks)
{
	PqScanState *state = NULL;

	nalloc(state, PqScanState, 1);
	NDB_CHECK_ALLOC(state, "state");

	state->query = query;
	state->dim = dim;
	state->k = k;
	state->rerank_k = rerank_k;
	state->m = m;
	state->ks = ks;
	state->subspace_dim = dim / m;

	nalloc(state->coarse_indices, uint32_t, rerank_k);
	nalloc(state->coarse_distances, float, rerank_k);
	nalloc(state->fine_indices, uint32_t, k);
	nalloc(state->fine_distances, float, k);

	state->coarse_count = 0;
	state->fine_count = 0;
	state->current_pos = 0;

	return state;
}

/*
 * Free PQ scan state
 */
static void
pq_scan_free(PqScanState * state)
{
	if (state)
	{
		pfree(state->coarse_indices);
		pfree(state->coarse_distances);
		pfree(state->fine_indices);
		pfree(state->fine_distances);
		pfree(state);
	}
}

/*
 * Stage 1: Coarse search using PQ codes
 * Uses GPU-accelerated asymmetric distance computation
 */
static int
pq_scan_coarse_search(PqScanState * state,
					  const uint8_t *pq_codes,
					  const float *codebooks,
					  int num_vectors)
{
	/* Use GPU for fast PQ distance computation */
	if (neurondb_gpu_is_available())
	{
		return neurondb_gpu_pq_asymmetric_search(state->query,
												 pq_codes,
												 codebooks,
												 num_vectors,
												 state->dim,
												 state->m,
												 state->ks,
												 state->rerank_k,
												 state->coarse_indices,
												 state->coarse_distances);
	}

	/*
	 * TODO: Implement CPU fallback for PQ coarse search.
	 * When GPU is unavailable or fails, this function should perform the
	 * Product Quantization coarse search on CPU. This requires implementing
	 * the PQ codebook lookup and distance computation using CPU SIMD operations
	 * for optimal performance.
	 */
	return -1;
}

/*
 * Stage 2: Fine rerank with full-precision vectors
 */
static int
pq_scan_fine_rerank(PqScanState * state, const float *full_vectors)
{
	int			i;
	float	   *distances = NULL;

	nalloc(distances, float, state->coarse_count);

	/* Compute distances to full-precision vectors */
	for (i = 0; i < state->coarse_count; i++)
	{
		uint32_t	idx = state->coarse_indices[i];
		const float *vec = full_vectors + idx * state->dim;
		float		dist = 0.0f;
		int			d;

		for (d = 0; d < state->dim; d++)
		{
			float		diff = state->query[d] - vec[d];

			dist += diff * diff;
		}
		distances[i] = sqrtf(dist);
	}

	/*
	 * TODO: Use proper heap/selection algorithm for top-k selection.
	 * The current O(n*k) selection sort should be replaced with a more
	 * efficient algorithm such as a min-heap (O(n log k)) or quickselect
	 * (O(n) average case) to improve performance for large coarse_count values.
	 */
	for (i = 0; i < state->k && i < state->coarse_count; i++)
	{
		int			best_idx = i;
		float		best_dist = distances[i];
		int			j;

		for (j = i + 1; j < state->coarse_count; j++)
		{
			if (distances[j] < best_dist)
			{
				best_dist = distances[j];
				best_idx = j;
			}
		}

		/* Swap */
		if (best_idx != i)
		{
			float		temp_dist = distances[i];
			uint32_t	temp_idx = state->coarse_indices[i];

			distances[i] = distances[best_idx];
			state->coarse_indices[i] = state->coarse_indices[best_idx];
			distances[best_idx] = temp_dist;
			state->coarse_indices[best_idx] = temp_idx;
		}

		state->fine_indices[i] = state->coarse_indices[i];
		state->fine_distances[i] = distances[i];
	}

	state->fine_count = i;
	pfree(distances);

	return 0;
}

