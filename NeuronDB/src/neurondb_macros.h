#ifndef NEURONDB_MACROS_H
#define NEURONDB_MACROS_H

#include "postgres.h"

#define nalloc(ptr, type, count) do { ptr = (type *) palloc(sizeof(type) * (count)); } while(0)
#define nfree(ptr) do { if (ptr) pfree(ptr); ptr = NULL; } while(0)

#endif
