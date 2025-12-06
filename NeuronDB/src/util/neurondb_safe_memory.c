/*-------------------------------------------------------------------------
 *
 * neurondb_safe_memory.c
 *    Memory context validation utilities for NeuronDB crash prevention
 *
 * Provides memory context validation and management functions.
 * Note: Safe pointer freeing functionality has been moved to neurondb_macros.h
 * (use nfree() instead of ndb_safe_pfree()).
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/util/neurondb_safe_memory.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/memutils.h"
#include "utils/elog.h"

#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"

bool
ndb_memory_context_validate(MemoryContext context)
{
	if (context == NULL)
		return false;

	/*
	 * In PostgreSQL, MemoryContextIsValid is a macro that checks if context
	 * has valid magic number. We use a similar check.
	 */
	return MemoryContextIsValid(context);
}

bool
ndb_ensure_memory_context(MemoryContext context)
{
	if (!ndb_memory_context_validate(context))
	{
		elog(WARNING,
			 "neurondb: attempt to switch to invalid memory context");
		return false;
	}

	if (CurrentMemoryContext != context)
		MemoryContextSwitchTo(context);

	return true;
}

void
ndb_safe_context_cleanup(MemoryContext context, MemoryContext oldcontext)
{
	if (context == NULL)
		return;

	if (!ndb_memory_context_validate(context))
	{
		elog(WARNING,
			 "neurondb: attempt to delete invalid memory context");
		return;
	}

	if (CurrentMemoryContext == context)
	{
		if (oldcontext == NULL || !ndb_memory_context_validate(oldcontext))
		{
			elog(ERROR,
				 "neurondb: cannot delete current memory context without valid old context");
			return;
		}
		MemoryContextSwitchTo(oldcontext);
	}

	MemoryContextDelete(context);
}

#ifdef NDB_DEBUG_MEMORY

typedef struct NdbPointerEntry
{
	void *ptr;
	MemoryContext context;
	const char *alloc_func;
}			NdbPointerEntry;

static NdbPointerEntry *ptr_tracker = NULL;
static int	ptr_tracker_size = 0;
	static int	ptr_tracker_count = 0;

void
ndb_track_allocation(void *ptr, const char *alloc_func)
{
	if (ptr == NULL)
		return;

	elog(DEBUG2,
		 "neurondb: tracking allocation %p from %s in context %p",
		 ptr,
		 alloc_func,
		 CurrentMemoryContext);
}

void
ndb_untrack_allocation(void *ptr)
{
	if (ptr == NULL)
		return;

	elog(DEBUG2, "neurondb: untracking allocation %p", ptr);
}

#endif							/* NDB_DEBUG_MEMORY */
