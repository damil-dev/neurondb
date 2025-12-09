/*-------------------------------------------------------------------------
 *
 * analytics.c
 *    Vector analytics and machine learning analysis.
 *
 * This module implements comprehensive vector analytics including clustering,
 * dimensionality reduction, outlier detection, and quality metrics.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/analytics.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"
#include "parser/parse_type.h"
#include "nodes/makefuncs.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_simd.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"

/*
 * feedback_loop_integrate
 *    Feedback loop integration: records feedback in a dedicated table,
 *    updating aggregations. Table: neurondb_feedback (query TEXT, result TEXT,
 *    rating REAL, ts TIMESTAMPTZ DEFAULT now()). If the table does not exist, creates it.
 */
PG_FUNCTION_INFO_V1(feedback_loop_integrate);

Datum
feedback_loop_integrate(PG_FUNCTION_ARGS)
{
	text	   *query;
	text	   *result;
	float4		user_rating;
	char *query_str = NULL;
	char *result_str = NULL;
	const char *tbl_def;
	int			ret;
	NdbSpiSession *spi_session = NULL;
	MemoryContext oldcontext;

	/* Validate argument count */
	if (PG_NARGS() < 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: feedback_loop_integrate requires at least 3 arguments")));

	query = PG_GETARG_TEXT_PP(0);
	result = PG_GETARG_TEXT_PP(1);
	user_rating = PG_GETARG_FLOAT4(2);

	query_str = text_to_cstring(query);
	result_str = text_to_cstring(result);

	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	tbl_def = "CREATE TABLE IF NOT EXISTS neurondb_feedback ("
		"id SERIAL PRIMARY KEY, "
		"query TEXT NOT NULL, "
		"result TEXT NOT NULL, "
		"rating REAL NOT NULL, "
		"ts TIMESTAMPTZ NOT NULL DEFAULT now()"
		")";
	ret = ndb_spi_execute(spi_session, tbl_def, false, 0);
	if (ret != SPI_OK_UTILITY)
	{
		NDB_SPI_SESSION_END(spi_session);
		nfree(query_str);
		nfree(result_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to create neurondb_feedback table")));
	}

	/* Use parameterized query to prevent SQL injection */
	{
		const char *insert_query = "INSERT INTO neurondb_feedback (query, result, rating) VALUES ($1, $2, $3)";
		Oid			argtypes[3] = {TEXTOID, TEXTOID, FLOAT4OID};
		Datum		values[3];
		const char nulls[3] = {' ', ' ', ' '};

		/* Validate rating is in reasonable range (0-5) */
		if (user_rating < 0.0f || user_rating > 5.0f)
		{
			NDB_SPI_SESSION_END(spi_session);
			nfree(query_str);
			nfree(result_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: rating must be between 0 and 5, got %g", user_rating)));
		}

		values[0] = PointerGetDatum(query);
		values[1] = PointerGetDatum(result);
		values[2] = Float4GetDatum(user_rating);

		ret = ndb_spi_execute_with_args(spi_session, insert_query, 3, argtypes, values, nulls, false, 0);
		if (ret != SPI_OK_INSERT)
		{
			NDB_SPI_SESSION_END(spi_session);
			nfree(query_str);
			nfree(result_str);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: failed to insert feedback row")));
		}
	}

	NDB_SPI_SESSION_END(spi_session);
	nfree(query_str);
	nfree(result_str);

	PG_RETURN_BOOL(true);
}

/* DBSCAN moved to ml_dbscan.c */

/*
 * =============================================================================
 * PCA - Principal Component Analysis
 * =============================================================================
 * Dimensionality reduction via singular value decomposition (SVD)
 * - n_components: Target dimension (must be <= original dimension)
 * - Returns projected vectors in lower dimensional space
 */

static void
pca_power_iteration(float **data,
					int nvec,
					int dim,
					float *eigvec,
					int max_iter)
{
	float *y = NULL;
	int			iter,
				i,
				j;
	double		norm;

	nalloc(y, float, dim);

	for (i = 0; i < dim; i++)
		eigvec[i] = (float) (rand() % 1000) / 1000.0f;

	norm = 0.0;
	for (i = 0; i < dim; i++)
		norm += eigvec[i] * eigvec[i];
	norm = sqrt(norm);
	for (i = 0; i < dim; i++)
		eigvec[i] /= norm;

	/* Power iteration - SIMD optimized */
	for (iter = 0; iter < max_iter; iter++)
	{
		/* y = X^T * X * eigvec */
		memset(y, 0, sizeof(float) * dim);

		for (j = 0; j < nvec; j++)
		{
			/* Use SIMD-optimized dot product */
			double		dot = neurondb_dot_product(data[j], eigvec, dim);

			for (i = 0; i < dim; i++)
				y[i] += data[j][i] * dot;
		}

		/* Normalize y */
		norm = 0.0;
		for (i = 0; i < dim; i++)
			norm += y[i] * y[i];
		norm = sqrt(norm);

		if (norm < 1e-10)
			break;

		for (i = 0; i < dim; i++)
			eigvec[i] = y[i] / norm;
	}

	nfree(y);
}

/* Deflate matrix by removing component of eigenvector */
static void
pca_deflate(float **data, int nvec, int dim, const float *eigvec)
{
	int			i,
				j;

	for (j = 0; j < nvec; j++)
	{
		double		dot = 0.0;

		for (i = 0; i < dim; i++)
			dot += data[j][i] * eigvec[i];

		for (i = 0; i < dim; i++)
			data[j][i] -= dot * eigvec[i];
	}
}

PG_FUNCTION_INFO_V1(reduce_pca);

Datum
reduce_pca(PG_FUNCTION_ARGS)
{
	ArrayType  *result_array = NULL;
	char	   *col_str = NULL;
	char	   *tbl_str = NULL;
	float	   *mean = NULL;
	float	   **components = NULL;
	float	   **data = NULL;
	float	   **centered_data = NULL;	/* Copy of centered data for projection */
	float	   **projected = NULL;
	int			c = 0;
	int			dim = 0;
	int			i = 0;
	int			j = 0;
	int			n_components = 0;
	int			nvec = 0;
	text	   *column_name = NULL;
	text	   *table_name = NULL;
	char		typalign = 0;
	bool		typbyval = false;
	int16		typlen = 0;

	/* Validate argument count */
	if (PG_NARGS() < 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: reduce_pca requires at least 3 arguments")));

	table_name = PG_GETARG_TEXT_PP(0);
	column_name = PG_GETARG_TEXT_PP(1);
	n_components = PG_GETARG_INT32(2);

	if (n_components < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_components must be at least 1")));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(column_name);

	elog(DEBUG1,
		 "neurondb: PCA dimensionality reduction on %s.%s "
		 "(n_components=%d)",
		 tbl_str,
		 col_str,
		 n_components);

	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);
	if (data == NULL || nvec == 0)
	{
		nfree(tbl_str);
		nfree(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));
	}

	if (dim <= 0)
	{
		nfree(tbl_str);
		nfree(col_str);
		if (data != NULL)
		{
			for (j = 0; j < nvec; j++)
			{
				if (data[j] != NULL)
					nfree(data[j]);
			}
			nfree(data);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimension: %d", dim)));
	}

	if (n_components > dim)
	{
		nfree(tbl_str);
		nfree(col_str);
		for (j = 0; j < nvec; j++)
		{
			if (data[j] != NULL)
				nfree(data[j]);
		}
		nfree(data);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_components (%d) cannot exceed "
						"dimension (%d)",
						n_components,
						dim)));
	}

	/* Compute mean */
	nalloc(mean, float, dim);
	memset(mean, 0, sizeof(float) * dim);
	for (j = 0; j < nvec; j++)
		for (i = 0; i < dim; i++)
			mean[i] += data[j][i];
	for (i = 0; i < dim; i++)
		mean[i] /= nvec;

	/* Center the data */
	for (j = 0; j < nvec; j++)
		for (i = 0; i < dim; i++)
			data[j][i] -= mean[i];

	/* Keep a copy of centered data for projection */
	nalloc(centered_data, float *, nvec);
	for (j = 0; j < nvec; j++)
	{
		nalloc(centered_data[j], float, dim);
		for (i = 0; i < dim; i++)
			centered_data[j][i] = data[j][i];
	}

	/* Compute principal components using power iteration and deflation */
	nalloc(components, float *, n_components);
	for (c = 0; c < n_components; c++)
	{
		float	   *component_row = NULL;

		nalloc(component_row, float, dim);
		components[c] = component_row;
		pca_power_iteration(data, nvec, dim, components[c], 100);
		pca_deflate(data, nvec, dim, components[c]);
	}

	/* Project centered data onto principal components */
	nalloc(projected, float *, nvec);
	for (j = 0; j < nvec; j++)
	{
		float *projected_row = NULL;
		nalloc(projected_row, float, n_components);
		projected[j] = projected_row;
		for (c = 0; c < n_components; c++)
		{
			double		dot = 0.0;

			/* Use original centered data, not deflated residuals */
			for (i = 0; i < dim; i++)
				dot += centered_data[j][i] * components[c][i];
			projected[j][c] = dot;
		}
	}

	/* Build 2D array real[][]: dims = [nvec][n_components] */
	{
		int			dims[2];
		int			lbs[2];
		Datum	   *flat_datums = NULL;
		int			idx = 0;

		dims[0] = nvec;
		dims[1] = n_components;
		lbs[0] = 1;
		lbs[1] = 1;

		nalloc(flat_datums, Datum, nvec * n_components);

		idx = 0;
		for (j = 0; j < nvec; j++)
		{
			for (c = 0; c < n_components; c++)
			{
				/* Validate projected value is finite */
				if (!isfinite(projected[j][c]))
				{
					nfree(flat_datums);
					for (i = 0; i < nvec; i++)
					{
						if (data[i] != NULL)
							nfree(data[i]);
						if (centered_data[i] != NULL)
							nfree(centered_data[i]);
						if (projected[i] != NULL)
							nfree(projected[i]);
					}
					for (c = 0; c < n_components; c++)
					{
						if (components[c] != NULL)
							nfree(components[c]);
					}
					nfree(data);
					nfree(centered_data);
					nfree(projected);
					nfree(components);
					nfree(mean);
					nfree(tbl_str);
					nfree(col_str);
					ereport(ERROR,
							(errcode(ERRCODE_DATA_EXCEPTION),
							 errmsg("reduce_pca: non-finite value in "
									"projected[%d][%d]", j, c)));
				}

				flat_datums[idx++] = Float4GetDatum(projected[j][c]);
			}
		}

		get_typlenbyvalalign(FLOAT4OID, &typlen, &typbyval, &typalign);

		result_array = construct_md_array(flat_datums,
										  NULL,
										  2,
										  dims,
										  lbs,
										  FLOAT4OID,
										  typlen,
										  typbyval,
										  typalign);

		if (result_array == NULL)
		{
			nfree(flat_datums);
			for (j = 0; j < nvec; j++)
			{
				if (data[j] != NULL)
					nfree(data[j]);
				if (centered_data[j] != NULL)
					nfree(centered_data[j]);
				if (projected[j] != NULL)
					nfree(projected[j]);
			}
			for (c = 0; c < n_components; c++)
			{
				if (components[c] != NULL)
					nfree(components[c]);
			}
			nfree(data);
			nfree(centered_data);
			nfree(projected);
			nfree(components);
			nfree(mean);
			nfree(tbl_str);
			nfree(col_str);
			ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
					 errmsg("reduce_pca: failed to construct result array")));
		}

		nfree(flat_datums);
	}
	for (j = 0; j < nvec; j++)
	{
		if (data[j] != NULL)
			nfree(data[j]);
		if (centered_data[j] != NULL)
			nfree(centered_data[j]);
		if (projected[j] != NULL)
			nfree(projected[j]);
	}
	for (c = 0; c < n_components; c++)
	{
		if (components[c] != NULL)
			nfree(components[c]);
	}
	nfree(data);
	nfree(centered_data);
	nfree(projected);
	nfree(components);
	nfree(mean);
	nfree(tbl_str);
	nfree(col_str);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * =============================================================================
 * Isolation Forest - Outlier Detection
 * =============================================================================
 * Anomaly detection using ensemble of isolation trees
 * - n_trees: Number of trees in the forest (default 100)
 * - contamination: Expected proportion of outliers (0.0-0.5)
 * - Returns anomaly scores (higher = more anomalous)
 */

typedef struct IsoTreeNode
{
	int			split_dim;		/* Dimension to split on (-1 = leaf) */
	float		split_val;		/* Value to split at */
	struct IsoTreeNode *left;
	struct IsoTreeNode *right;
	int			size;			/* Number of points in this node */
}			IsoTreeNode;

static IsoTreeNode *
build_iso_tree(float **data,
			   int *indices,
			   int n,
			   int dim,
			   int depth,
			   int max_depth)
{
	IsoTreeNode *node = NULL;
	int			i,
				split_dim;
	float		split_val,
				min_val,
				max_val;
	int			left_count,
				right_count;
	int *left_indices = NULL;
	int *right_indices = NULL;

	nalloc(node, IsoTreeNode, 1);
	node->size = n;

	if (n <= 1 || depth >= max_depth)
	{
		node->split_dim = -1;	/* Leaf node */
		return node;
	}

	split_dim = rand() % dim;
	node->split_dim = split_dim;

	min_val = max_val = data[indices[0]][split_dim];
	for (i = 1; i < n; i++)
	{
		float		val = data[indices[i]][split_dim];

		if (val < min_val)
			min_val = val;
		if (val > max_val)
			max_val = val;
	}

	if (max_val - min_val < 1e-6)
	{
		node->split_dim = -1;	/* Can't split */
		return node;
	}
	split_val = min_val + (float) (((double) rand() / (double) RAND_MAX)) * (max_val - min_val);
	node->split_val = split_val;

	nalloc(left_indices, int, n);
	nalloc(right_indices, int, n);
	left_count = right_count = 0;

	for (i = 0; i < n; i++)
	{
		if (data[indices[i]][split_dim] < split_val)
			left_indices[left_count++] = indices[i];
		else
			right_indices[right_count++] = indices[i];
	}

	if (left_count > 0)
		node->left = build_iso_tree(data,
									left_indices,
									left_count,
									dim,
									depth + 1,
									max_depth);
	if (right_count > 0)
		node->right = build_iso_tree(data,
									 right_indices,
									 right_count,
									 dim,
									 depth + 1,
									 max_depth);

	nfree(left_indices);
	nfree(right_indices);

	return node;
}

static double
iso_tree_path_length(IsoTreeNode * node, const float *point, int depth)
{
	double		h;

	if (node->split_dim == -1)
	{
		if (node->size <= 1)
			return depth;
		h = log(node->size) + 0.5772156649; /* Euler's constant */
		return depth + h;
	}

	if (point[node->split_dim] < node->split_val && node->left)
		return iso_tree_path_length(node->left, point, depth + 1);
	else if (node->right)
		return iso_tree_path_length(node->right, point, depth + 1);
	else
		return depth;
}

static void
free_iso_tree(IsoTreeNode * node)
{
	if (node == NULL)
		return;
	free_iso_tree(node->left);
	free_iso_tree(node->right);
	nfree(node);
}

PG_FUNCTION_INFO_V1(detect_outliers);

Datum
detect_outliers(PG_FUNCTION_ARGS)
{
	text *table_name = NULL;
	text *column_name = NULL;
	int			n_trees;
	float		contamination;
	char *tbl_str = NULL;
	char *col_str = NULL;
	float	  **data;
	int			nvec,
				dim;

	IsoTreeNode * * forest = NULL;
	double *scores = NULL;
	int			i,
				t;

	int *indices = NULL;
	int			max_depth;
	double		avg_path_length_full;
	ArrayType *result_array = NULL;

	Datum *result_datums = NULL;
	char		typalign = 0;
	bool		typbyval = false;
	int16		typlen = 0;

	/* Validate argument count */
	if (PG_NARGS() < 2 || PG_NARGS() > 4)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: detect_outliers requires 2 to 4 arguments")));

	table_name = PG_GETARG_TEXT_PP(0);
	column_name = PG_GETARG_TEXT_PP(1);
	n_trees = PG_GETARG_INT32(2);
	contamination = PG_GETARG_FLOAT4(3);

	if (n_trees < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_trees must be at least 1")));

	if (contamination < 0.0 || contamination > 0.5)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("contamination must be between 0.0 and "
						"0.5")));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(column_name);

	elog(DEBUG1,
		 "neurondb: Isolation Forest on %s.%s (n_trees=%d, contamination=%.3f)",
		 tbl_str,
		 col_str,
		 n_trees,
		 contamination);

	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);
	if (nvec == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));

	max_depth = (int) ceil(log2(nvec));
	nalloc(forest, IsoTreeNode *, n_trees);
	nalloc(indices, int, nvec);

	/* Seed random number generator for reproducible results */
	/* Note: In production, this should be seeded once in module init */
	srand((unsigned int) time(NULL));

	for (t = 0; t < n_trees; t++)
	{
		int			sample_size = (nvec < 256) ? nvec : 256;

		for (i = 0; i < sample_size; i++)
			indices[i] = rand() % nvec;

		forest[t] = build_iso_tree(
								   data, indices, sample_size, dim, 0, max_depth);
	}

	avg_path_length_full = (nvec > 1) ? 2.0 * (log(nvec - 1) + 0.5772156649)
		- 2.0 * (nvec - 1.0) / nvec
		: 0.0;
	nalloc(scores, double, nvec);

	for (i = 0; i < nvec; i++)
	{
		double		avg_path = 0.0;

		for (t = 0; t < n_trees; t++)
			avg_path += iso_tree_path_length(forest[t], data[i], 0);
		avg_path /= n_trees;

		if (avg_path_length_full > 0)
			scores[i] = pow(2.0, -avg_path / avg_path_length_full);
		else
			scores[i] = 0.0;
	}

	nalloc(result_datums, Datum, nvec);
	for (i = 0; i < nvec; i++)
		result_datums[i] = Float4GetDatum((float) scores[i]);

	get_typlenbyvalalign(FLOAT4OID, &typlen, &typbyval, &typalign);
	result_array = construct_array(
								   result_datums, nvec, FLOAT4OID, typlen, typbyval, typalign);

	for (t = 0; t < n_trees; t++)
		free_iso_tree(forest[t]);
	for (i = 0; i < nvec; i++)
		nfree(data[i]);
	nfree(data);
	nfree(forest);
	nfree(scores);
	nfree(indices);
	nfree(result_datums);
	nfree(tbl_str);
	nfree(col_str);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * =============================================================================
 * KNN Graph Construction
 * =============================================================================
 * Build k-nearest neighbor graph for vectors
 * - k: Number of neighbors per point
 * - Returns edge list as array of (source, target, distance) tuples
 */

typedef struct KNNEdge
{
	int			target;
	float		distance;
}			KNNEdge;

static int
knn_edge_compare(const void *a, const void *b)
{
	const		KNNEdge *ea = (const KNNEdge *) a;
	const		KNNEdge *eb = (const KNNEdge *) b;

	if (ea->distance < eb->distance)
		return -1;
	if (ea->distance > eb->distance)
		return 1;
	return 0;
}

PG_FUNCTION_INFO_V1(build_knn_graph);

Datum
build_knn_graph(PG_FUNCTION_ARGS)
{
	text *table_name = NULL;
	text *column_name = NULL;
	int			k;
	char *tbl_str = NULL;
	char *col_str = NULL;
	float	  **data;
	int			nvec,
				dim;
	int			i,
				j,
				n;

	KNNEdge *edges = NULL;
	ArrayType *result_array = NULL;

	Datum *result_datums = NULL;
	int			result_count = 0;
	char		typalign = 0;
	bool		typbyval = false;
	int16		typlen = 0;

	/* Validate argument count */
	if (PG_NARGS() < 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: build_knn_graph requires at least 3 arguments")));

	table_name = PG_GETARG_TEXT_PP(0);
	column_name = PG_GETARG_TEXT_PP(1);
	k = PG_GETARG_INT32(2);

	if (k < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("k must be at least 1")));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(column_name);

	elog(DEBUG1,
		 "neurondb: Building KNN graph on %s.%s (k=%d)",
		 tbl_str,
		 col_str,
		 k);

	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);
	if (data == NULL || nvec == 0)
	{
		nfree(tbl_str);
		nfree(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));
	}

	if (dim <= 0)
	{
		nfree(tbl_str);
		nfree(col_str);
		/* Free data array and rows if data is not NULL */
		if (data != NULL)
		{
			for (i = 0; i < nvec; i++)
			{
				if (data[i] != NULL)
					nfree(data[i]);
			}
			nfree(data);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimension: %d", dim)));
	}

	if (k >= nvec)
		k = nvec - 1;

	/* Allocate edges array - we need nvec-1 edges per node (excluding self) */
	nalloc(edges, KNNEdge, nvec - 1);
	result_count = 0;
	/* Store as array of real[3] arrays: [src, dst, dist] */
	nalloc(result_datums, Datum, nvec * k);

	for (i = 0; i < nvec; i++)
	{
		int			edge_count = 0;
		double		dist_sq;
		double		diff;

		/* Build edges list excluding self */
		for (j = 0; j < nvec; j++)
		{
			if (i == j)
				continue;

			dist_sq = 0.0;
			for (n = 0; n < dim; n++)
			{
				diff = (double) data[i][n] - (double) data[j][n];
				dist_sq += diff * diff;
			}
			edges[edge_count].target = j;
			edges[edge_count].distance = sqrt(dist_sq);
			edge_count++;
		}

		/* Sort by distance */
		qsort(edges, edge_count, sizeof(KNNEdge), knn_edge_compare);

		/* Take top k neighbors */
		for (j = 0; j < k && j < edge_count; j++)
		{
			/* Build array [src, dst, dist] for this edge */
			Datum	   *edge_datums = NULL;
			ArrayType  *edge_array = NULL;
			int			edge_idx = result_count++;

			nalloc(edge_datums, Datum, 3);
			edge_datums[0] = Int32GetDatum(i);
			edge_datums[1] = Int32GetDatum(edges[j].target);
			edge_datums[2] = Float4GetDatum(edges[j].distance);

			get_typlenbyvalalign(FLOAT4OID, &typlen, &typbyval, &typalign);
			edge_array = construct_array(edge_datums, 3, FLOAT4OID, typlen, typbyval, typalign);
			nfree(edge_datums);

			if (edge_array == NULL)
			{
				/* Cleanup */
				for (i = 0; i < nvec; i++)
					nfree(data[i]);
				nfree(data);
				nfree(edges);
				for (i = 0; i < edge_idx; i++)
				{
					if (result_datums[i] != 0)
					{
						ArrayType  *arr = DatumGetArrayTypeP(result_datums[i]);
						nfree(arr);
					}
				}
				nfree(result_datums);
				nfree(tbl_str);
				nfree(col_str);
				ereport(ERROR,
						(errcode(ERRCODE_OUT_OF_MEMORY),
						 errmsg("build_knn_graph: failed to construct edge array")));
			}

			result_datums[edge_idx] = PointerGetDatum(edge_array);
		}
	}

	/* Build 2D array real[][3]: dims = [result_count][3] */
	/* Convert array-of-arrays to true 2D array */
	{
		int			dims[2];
		int			lbs[2];
		Datum	   *flat_datums = NULL;
		int			idx = 0;

		dims[0] = result_count;
		dims[1] = 3;
		lbs[0] = 1;
		lbs[1] = 1;

		nalloc(flat_datums, Datum, result_count * 3);

		idx = 0;
		for (i = 0; i < result_count; i++)
		{
			ArrayType  *edge_array = NULL;
			Datum	   *edge_elems = NULL;
			bool	   *nulls = NULL;
			int			nelems;
			int			e;

			if (result_datums[i] != 0)
			{
				edge_array = DatumGetArrayTypeP(result_datums[i]);
				deconstruct_array(edge_array,
								  FLOAT4OID,
								  sizeof(float4),
								  true,
								  'i',
								  &edge_elems,
								  &nulls,
								  &nelems);

				for (e = 0; e < nelems && e < 3; e++)
				{
					if (!nulls[e])
						flat_datums[idx++] = edge_elems[e];
					else
						flat_datums[idx++] = Float4GetDatum(0.0);
				}
				/* Free the inner array */
				nfree(edge_array);
			}
			else
			{
				/* Fill with zeros if array is NULL */
				flat_datums[idx++] = Float4GetDatum(0.0);
				flat_datums[idx++] = Float4GetDatum(0.0);
				flat_datums[idx++] = Float4GetDatum(0.0);
			}
		}

		get_typlenbyvalalign(FLOAT4OID, &typlen, &typbyval, &typalign);

		result_array = construct_md_array(flat_datums,
										  NULL,
										  2,
										  dims,
										  lbs,
										  FLOAT4OID,
										  typlen,
										  typbyval,
										  typalign);

		nfree(flat_datums);
	}

	/* Free result_datums (inner arrays already freed above) */

	for (i = 0; i < nvec; i++)
		nfree(data[i]);
	nfree(data);
	nfree(edges);
	nfree(result_datums);
	nfree(tbl_str);
	nfree(col_str);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * =============================================================================
 * Embedding Quality Metrics
 * =============================================================================
 * Compute quality metrics for embeddings (silhouette score, etc.)
 * - Returns quality score between -1 and 1 (higher = better)
 */

PG_FUNCTION_INFO_V1(compute_embedding_quality);

Datum
compute_embedding_quality(PG_FUNCTION_ARGS)
{
	text *table_name = NULL;
	text *column_name = NULL;
	text *cluster_column = NULL;
	char *tbl_str = NULL;
	char *col_str = NULL;
	char *cluster_col_str = NULL;
	float	  **data;

	int *clusters = NULL;
	int			nvec,
				dim;
	int			i,
				j;

	double *a_scores = NULL;	/* Average distance to same cluster */
	double *b_scores = NULL;	/* Average distance to nearest other
										 * cluster */
	double		silhouette;
	StringInfoData sql;
	int			ret;

	NdbSpiSession *spi_session = NULL;
	MemoryContext oldcontext;

	/* Validate argument count */
	if (PG_NARGS() < 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: compute_embedding_quality requires at least 3 arguments")));

	table_name = PG_GETARG_TEXT_PP(0);
	column_name = PG_GETARG_TEXT_PP(1);
	cluster_column = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(column_name);
	cluster_col_str = text_to_cstring(cluster_column);

	elog(DEBUG1,
		 "neurondb: Computing embedding quality for %s.%s (clusters=%s)",
		 tbl_str,
		 col_str,
		 cluster_col_str);

	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);
	if (data == NULL || nvec == 0)
	{
		nfree(tbl_str);
		nfree(col_str);
		nfree(cluster_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));
	}

	if (dim <= 0)
	{
		nfree(tbl_str);
		nfree(col_str);
		nfree(cluster_col_str);
		/* Free data array and rows if data is not NULL */
		if (data != NULL)
		{
			for (i = 0; i < nvec; i++)
			{
				if (data[i] != NULL)
					nfree(data[i]);
			}
			nfree(data);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimension: %d", dim)));
	}

	oldcontext = CurrentMemoryContext;
	nalloc(clusters, int, nvec);

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ndb_spi_stringinfo_init(spi_session, &sql);
	/* Note: No ORDER BY clause - views don't have ctid, and ordering isn't required */
	appendStringInfo(&sql, "SELECT %s FROM %s", cluster_col_str, tbl_str);
	ret = ndb_spi_execute(spi_session, sql.data, true, 0);

	if (ret != SPI_OK_SELECT || (int) SPI_processed != nvec)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		nfree(clusters);
		nfree(tbl_str);
		nfree(col_str);
		nfree(cluster_col_str);
		/* Free data array and rows */
		for (i = 0; i < nvec; i++)
		{
			if (data[i] != NULL)
				nfree(data[i]);
		}
		nfree(data);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Failed to fetch cluster assignments")));
	}

	for (i = 0; i < nvec; i++)
	{
		int32		val;

		if (ndb_spi_get_int32(spi_session, i, 1, &val))
		{
			clusters[i] = val;
		}
		else
		{
			clusters[i] = -1;
		}
	}

	ndb_spi_stringinfo_free(spi_session, &sql);
	NDB_SPI_SESSION_END(spi_session);

	nalloc(a_scores, double, nvec);
	nalloc(b_scores, double, nvec);

	for (i = 0; i < nvec; i++)
	{
		int			my_cluster = clusters[i];
		int			same_count = 0;
		double		same_dist = 0.0;
		double		min_other_dist = DBL_MAX;
		double		dist;
		int			d;
		double		diff;

		if (my_cluster == -1)	/* Noise point */
			continue;

		for (j = 0; j < nvec; j++)
		{
			if (i == j)
				continue;

			dist = 0.0;
			for (d = 0; d < dim; d++)
			{
				diff = (double) data[i][d] - (double) data[j][d];
				dist += diff * diff;
			}
			dist = sqrt(dist);

			if (clusters[j] == my_cluster)
			{
				same_dist += dist;
				same_count++;
			}
			else if (clusters[j] != -1)
			{
				if (dist < min_other_dist)
					min_other_dist = dist;
			}
		}

		if (same_count > 0)
			a_scores[i] = same_dist / same_count;
		b_scores[i] = min_other_dist;
	}

	{
		int			valid_count = 0;
		double		s;

		silhouette = 0.0;
		for (i = 0; i < nvec; i++)
		{
			if (clusters[i] == -1)
				continue;

			if (a_scores[i] < b_scores[i])
				s = 1.0 - a_scores[i] / b_scores[i];
			else if (a_scores[i] > b_scores[i])
				s = b_scores[i] / a_scores[i] - 1.0;
			else
				s = 0.0;

			silhouette += s;
			valid_count++;
		}

		if (valid_count > 0)
			silhouette /= valid_count;
	}

	for (i = 0; i < nvec; i++)
		nfree(data[i]);
	nfree(data);
	nfree(clusters);
	nfree(a_scores);
	nfree(b_scores);
	nfree(tbl_str);
	nfree(col_str);
	nfree(cluster_col_str);

	PG_RETURN_FLOAT8(silhouette);
}
