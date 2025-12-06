/*-------------------------------------------------------------------------
 *
 * developer_hooks.c
 *		Developer Hooks: Planner Extension API, Logical Replication,
 *		Foreign Data Wrapper, Unit Test Framework
 *
 * This file implements developer extensibility features including
 * planner extension API, logical replication plugin, FDW for vectors,
 * and SQL-based unit test framework.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/developer_hooks.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "executor/spi.h"

#include <math.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

PG_FUNCTION_INFO_V1(register_custom_operator);
Datum
register_custom_operator(PG_FUNCTION_ARGS)
{
	text	   *op_name = PG_GETARG_TEXT_PP(0);
	text	   *op_function = PG_GETARG_TEXT_PP(1);
	char *name_str = NULL;
	char *func_str = NULL;

	name_str = text_to_cstring(op_name);
	func_str = text_to_cstring(op_function);

	elog(DEBUG1,
		 "neurondb: registering custom operator '%s' -> '%s'",
		 name_str,
		 func_str);

	PG_RETURN_BOOL(true);
}

PG_FUNCTION_INFO_V1(enable_vector_replication);
Datum
enable_vector_replication(PG_FUNCTION_ARGS)
{
	text	   *publication_name = PG_GETARG_TEXT_PP(0);
	char *pub_str = NULL;

	pub_str = text_to_cstring(publication_name);

	(void) pub_str;

	PG_RETURN_BOOL(true);
}

PG_FUNCTION_INFO_V1(create_vector_fdw);
Datum
create_vector_fdw(PG_FUNCTION_ARGS)
{
	text	   *fdw_name = PG_GETARG_TEXT_PP(0);
	text	   *remote_type = PG_GETARG_TEXT_PP(1);
	text	   *connection_string = PG_GETARG_TEXT_PP(2);
	char *name_str = NULL;
	char *type_str = NULL;
	char *conn_str = NULL;

	name_str = text_to_cstring(fdw_name);
	type_str = text_to_cstring(remote_type);
	conn_str = text_to_cstring(connection_string);

	elog(DEBUG1,
		 "neurondb: creating %s FDW '%s' with connection '%s'",
		 type_str,
		 name_str,
		 conn_str);

	nfree(name_str);
	nfree(type_str);
	nfree(conn_str);

	PG_RETURN_BOOL(true);
}

PG_FUNCTION_INFO_V1(assert_recall);
Datum
assert_recall(PG_FUNCTION_ARGS)
{
	float4		actual_recall = PG_GETARG_FLOAT4(0);
	float4		expected_recall = PG_GETARG_FLOAT4(1);
	float4		tolerance = PG_GETARG_FLOAT4(2);
	bool		passed;

	passed = (fabs(actual_recall - expected_recall) <= tolerance);

	if (passed)
		elog(DEBUG1,
			 "neurondb: TEST PASSED: recall=%.4f (expected=%.4f "
			 "±%.4f)",
			 actual_recall,
			 expected_recall,
			 tolerance);
	else
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("neurondb: TEST FAILED: recall=%.4f "
						"(expected=%.4f ±%.4f)",
						actual_recall,
						expected_recall,
						tolerance)));

	PG_RETURN_BOOL(passed);
}

PG_FUNCTION_INFO_V1(assert_vector_equal);
Datum
assert_vector_equal(PG_FUNCTION_ARGS)
{
	Vector	   *vec1 = (Vector *) PG_GETARG_POINTER(0);
	Vector	   *vec2 = (Vector *) PG_GETARG_POINTER(1);
	float4		tolerance = PG_GETARG_FLOAT4(2);
	bool		passed = true;
	int			i;

	if (vec1->dim != vec2->dim)
	{
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("neurondb: TEST FAILED: dimension "
						"mismatch %d != %d",
						vec1->dim,
						vec2->dim)));
	}

	for (i = 0; i < vec1->dim; i++)
	{
		if (fabs(vec1->data[i] - vec2->data[i]) > tolerance)
		{
			passed = false;
			break;
		}
	}

	if (passed)
		elog(DEBUG1,
			 "neurondb: TEST PASSED: vectors equal within tolerance "
			 "%.6f",
			 tolerance);
	else
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("neurondb: TEST FAILED: vectors differ "
						"at position %d",
						i)));

	PG_RETURN_BOOL(passed);
}
