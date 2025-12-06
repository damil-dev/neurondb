/*-------------------------------------------------------------------------
 *
 * neurondb_macros.h
 *	  Strict pointer lifetime helpers for NeurondDB
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_MACROS_H
#define NEURONDB_MACROS_H

#include "postgres.h"
#include "utils/memutils.h"
#include "neurondb_constants.h"

/*
 * nalloc
 * Allocate with palloc0, require previous value NULL.
 *
 * Usage:
 *	 nalloc(model, LinRegModel, 1);
 *	 nalloc(coeffs, double, n_features);
 */
#define nalloc(ptr, type, count)				\
	Assert((ptr) == NULL);						\
	do {									\
		Assert((ptr) == NULL);				\
		(ptr) = (type *) palloc0(sizeof(type) * (count));	\
	} while (0)

/*
 * NBP_ALLOC
 * Allocate with palloc (non-zero-initialized), require previous value NULL.
 *
 * Usage:
 *	 NBP_ALLOC(buf, char, size);
 *	 NBP_ALLOC(data, float, n_elements);
 */
#define NBP_ALLOC(ptr, type, count)				\
	Assert((ptr) == NULL);						\
	do {									\
		Assert((ptr) == NULL);				\
		(ptr) = (type *) palloc(sizeof(type) * (count));	\
	} while (0)

/*
 * nfree
 * Free pointer if not NULL and then set to NULL.
 *
 * Usage:
 *	 nfree(model);
 *	 nfree(coeffs);
 */
#define nfree(ptr)					\
	do {								\
		if ((ptr) != NULL)				\
		{								\
			pfree(ptr);					\
			(ptr) = NULL;				\
		}								\
	} while (0)

#endif							/* NEURONDB_MACROS_H */
