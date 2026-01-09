/*-------------------------------------------------------------------------
 *
 * vector_sql_functions.c
 *    SQL functions for advanced vector search operations
 *
 * Implements batch search, PQ search, filtered search, and recall evaluation.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/sql/vector_sql_functions.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_types.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "access/htup_details.h"
#include "catalog/pg_type.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/*
 * vector_batch_search(queries vector[], k int)
 *    Batch search for multiple query vectors
 */
PG_FUNCTION_INFO_V1(vector_batch_search);
Datum
vector_batch_search(PG_FUNCTION_ARGS)
{
	ArrayType *queries_array = PG_GETARG_ARRAYTYPE_P(0);
	int32		k = PG_GETARG_INT32(1);
	FuncCallContext *funcctx = NULL;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		TupleDesc	tupdesc;

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		/* Build tuple descriptor */
		if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("function returning record called in context that cannot accept type record")));

		funcctx->tuple_desc = BlessTupleDesc(tupdesc);
		
		/* Extract queries from array */
		Datum	   *queries_elems;
		bool	   *queries_nulls;
		int			queries_count;
		float	   *query_vectors = NULL;
		int			dim = 0;
		int			i;
		Vector	   *first_query = NULL;

		deconstruct_array(queries_array, TEXTOID, -1, false, 'i',
						  &queries_elems, &queries_nulls, &queries_count);

		if (queries_count == 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("queries array cannot be empty")));

		/* Get dimension from first query */
		first_query = DatumGetVectorP(queries_elems[0]);
		NDB_CHECK_VECTOR_VALID(first_query);
		dim = VECTOR_SIZE(first_query);

		/* Allocate query vectors array */
		nalloc(query_vectors, float, queries_count * dim);
		NDB_CHECK_ALLOC(query_vectors, "query_vectors");

		/* Extract all query vectors */
		for (i = 0; i < queries_count; i++)
		{
			if (queries_nulls[i])
				continue;

			Vector	   *query = DatumGetVectorP(queries_elems[i]);
			float	   *query_data = NULL;

			NDB_CHECK_VECTOR_VALID(query);
			if (VECTOR_SIZE(query) != dim)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("all query vectors must have the same dimension")));

			query_data = VECTOR_DATA(query);
			memcpy(query_vectors + i * dim, query_data, dim * sizeof(float));
		}

		/* Store batch scan state */
		funcctx->user_fctx = query_vectors;
		funcctx->max_calls = queries_count * k; /* Total results */

		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();

	/* TODO: Execute batch search and return results */
	/* For now, return done */
	SRF_RETURN_DONE(funcctx);
}

/*
 * vector_pq_search(query vector, k int, rerank_k int)
 *    PQ-accelerated search with two-stage retrieval
 */
PG_FUNCTION_INFO_V1(vector_pq_search);
Datum
vector_pq_search(PG_FUNCTION_ARGS)
{
	Vector	   *query = PG_GETARG_VECTOR_P(0);
	int32		k = PG_GETARG_INT32(1);
	int32		rerank_k = PG_GETARG_INT32(2);

	FuncCallContext *funcctx = NULL;

	NDB_CHECK_VECTOR_VALID(query);

	if (k <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("k must be positive")));

	if (rerank_k <= 0 || rerank_k < k)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("rerank_k must be positive and >= k")));

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		TupleDesc	tupdesc;

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("function returning record called in context that cannot accept type record")));

		funcctx->tuple_desc = BlessTupleDesc(tupdesc);
		funcctx->user_fctx = NULL; /* TODO: Store PQ search state */
		funcctx->max_calls = k;

		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();

	/* TODO: Execute PQ search and return results */
	/* For now, return done */
	SRF_RETURN_DONE(funcctx);
}

/*
 * vector_filtered_search(query vector, filter_predicate text, k int)
 *    Filtered search with auto-tuning
 */
PG_FUNCTION_INFO_V1(vector_filtered_search);
Datum
vector_filtered_search(PG_FUNCTION_ARGS)
{
	Vector	   *query = PG_GETARG_VECTOR_P(0);
	text	   *filter_predicate = PG_GETARG_TEXT_P(1);
	int32		k = PG_GETARG_INT32(2);

	FuncCallContext *funcctx = NULL;

	NDB_CHECK_VECTOR_VALID(query);

	if (k <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("k must be positive")));

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		TupleDesc	tupdesc;
		char	   *filter_str = NULL;

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("function returning record called in context that cannot accept type record")));

		funcctx->tuple_desc = BlessTupleDesc(tupdesc);

		/* Parse filter predicate */
		filter_str = text_to_cstring(filter_predicate);
		/* TODO: Parse and compile filter predicate */
		funcctx->user_fctx = filter_str;
		funcctx->max_calls = k;

		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();

	/* TODO: Execute filtered search and return results */
	/* For now, return done */
	SRF_RETURN_DONE(funcctx);
}

/*
 * vector_recall_eval(query vector, approximate_results bigint[], exact_results bigint[])
 *    Evaluate recall of approximate search vs exact search
 */
PG_FUNCTION_INFO_V1(vector_recall_eval);
Datum
vector_recall_eval(PG_FUNCTION_ARGS)
{
	Vector	   *query = PG_GETARG_VECTOR_P(0);
	ArrayType *approx_array = PG_GETARG_ARRAYTYPE_P(1);
	ArrayType *exact_array = PG_GETARG_ARRAYTYPE_P(2);
	Datum	   *approx_elems;
	Datum	   *exact_elems;
	bool	   *approx_nulls;
	bool	   *exact_nulls;
	int			approx_count;
	int			exact_count;
	int			matches = 0;
	float8		recall;

	/* Extract array elements */
	deconstruct_array(approx_array, INT8OID, 8, true, 'd',
					  &approx_elems, &approx_nulls, &approx_count);
	deconstruct_array(exact_array, INT8OID, 8, true, 'd',
					  &exact_elems, &exact_nulls, &exact_count);

	/* Count matches */
	for (int i = 0; i < approx_count; i++)
	{
		if (approx_nulls[i])
			continue;

		int64		approx_val = DatumGetInt64(approx_elems[i]);

		for (int j = 0; j < exact_count; j++)
		{
			if (exact_nulls[j])
				continue;

			int64		exact_val = DatumGetInt64(exact_elems[j]);

			if (approx_val == exact_val)
			{
				matches++;
				break;
			}
		}
	}

	/* Compute recall */
	if (exact_count > 0)
		recall = (float8) matches / (float8) exact_count;
	else
		recall = 0.0;

	PG_RETURN_FLOAT8(recall);
}

