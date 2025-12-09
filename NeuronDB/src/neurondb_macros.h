#ifndef NEURONDB_MACROS_H
#define NEURONDB_MACROS_H

#include "postgres.h"

/* nalloc: Allocate and zero-initialize memory, pointer must be NULL */
#define nalloc(ptr, type, count) do { \
	Assert((ptr) == NULL); \
	(ptr) = (type *) palloc(sizeof(type) * (count)); \
	MemSet((ptr), 0, sizeof(type) * (count)); \
} while(0)

/* nfree: Free pointer if not NULL and set to NULL */
#define nfree(ptr) do { \
	if ((ptr) != NULL) { \
		pfree(ptr); \
		(ptr) = NULL; \
	} \
} while(0)

#endif
