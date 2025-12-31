# C Code Audit Checklist (Regenerated)

Checklist for reviewing C/CUDA code changes in `NeuronDB/src/`.

## Memory safety

- Bounds checks on arrays/buffers
- NULL pointer checks
- Ownership and lifetime clear
- No double-free / use-after-free

## Postgres extension rules

- Use correct memory contexts
- Avoid leaking memory across calls
- Correct error handling (`ereport`, `elog`)

## Concurrency and locking

- Locks acquired in correct order
- No long blocking operations while holding locks

## GPU paths

- CPU fallback exists (where appropriate)
- Error propagation from GPU backend
- Resource cleanup on failure paths

# C Code Audit Checklist for Crash Prevention

This checklist tracks memory safety and crash prevention patterns across all C code in NeuronDB.

## Memory Management

### pfree → nfree Migration
- [x] **src/ml/** - All ML algorithm files
  - [x] ml_unified_api.c
  - [x] ml_logistic_regression.c
  - [x] ml_ridge_lasso.c
  - [x] ml_naive_bayes.c
  - [x] ml_decision_tree.c
  - [x] ml_automl.c
  - [x] ml_lightgbm.c
  - [x] ml_catalog.c
  - [x] reranking.c
  - [x] ml_reranking_flash.c
  - [x] ml_hyperparameter_tuning.c
  - [ ] ... (other ml/*.c files - audit remaining files as needed)

- [x] **src/index/** - All index access method files
  - [x] hnsw_am.c
  - [x] index_hnsw_tenant.c
  - [x] index_validator.c
  - [ ] ... (other index/*.c files - audit remaining files as needed)

- [x] **src/util/** - Utility files
  - [x] neurondb_json.c
  - [ ] ... (other util/*.c files - audit remaining files as needed)

- [x] **src/gpu/** - GPU implementation files
  - [x] gpu_model_bridge.c
  - [x] gpu_backend_registry.c
  - [x] gpu_rf_cuda.c
  - [x] gpu_hf_cuda.c
  - [x] gpu_knn_cuda.c
  - [x] gpu_ridge_cuda.c
  - [x] gpu_xgboost_cuda.c
  - [x] gpu_catboost_cuda.c
  - [x] gpu_svm_cuda.c
  - [x] gpu_lr_cuda.c
  - [x] gpu_linreg_cuda.c
  - [x] gpu_backend_metal.c
  - [x] gpu_svm_rocm.c
  - [x] gpu_hf_rocm.c
  - [ ] ... (other gpu/**/*.c files - audit remaining files as needed)

- [x] **src/vector/** - Vector operation files
  - [x] vector_cast.c
  - [ ] ... (other vector/*.c files - audit remaining files as needed)

- [x] **src/worker/** - Worker files
  - [x] worker_llm.c
  - [ ] ... (other worker/*.c files - audit remaining files as needed)

**Status:** All identified pfree() calls have been replaced with nfree() across 33 files. Remaining pfree() references are only in comments or inside the nfree() macro definition itself.

### Pattern to Replace
```c
// OLD (unsafe):
pfree(ptr);

// NEW (safe):
nfree(ptr);  // Sets ptr = NULL after freeing
```

## Pointer Dereferences

### NULL Checks Before Dereference
- [ ] All `->` operator usages have NULL checks
- [ ] All `*ptr` usages have NULL checks
- [ ] Use `NDB_CHECK_NULL()` macro where appropriate

### Pattern
```c
// Before dereference:
if (ptr == NULL)
    ereport(ERROR, ...);

// Or use macro:
NDB_CHECK_NULL(ptr, "parameter_name");
value = ptr->field;
```

## Allocation Safety

### Size Validation
- [ ] All allocations validate size before allocating
- [ ] Use `NDB_CHECK_SIZE_OVERFLOW()` before multiplications
- [ ] Use `NDB_CHECK_ALLOC_SIZE()` before palloc
- [ ] Use `NDB_CHECK_ALLOC()` after palloc

### Pattern
```c
// Before allocation:
NDB_CHECK_SIZE_OVERFLOW(n_samples, feature_dim, sizeof(float));
size_t total_size = n_samples * feature_dim * sizeof(float);
NDB_CHECK_ALLOC_SIZE(total_size, "feature_matrix");

ptr = palloc(total_size);
NDB_CHECK_ALLOC(ptr, "feature_matrix");
```

## SPI Operations

### Safe SPI Wrappers
- [ ] All SPI_connect calls wrapped safely
- [ ] All SPI_execute calls check return codes
- [ ] All data copied to caller context BEFORE SPI_finish()
- [ ] Use `NDB_CHECK_SPI_RESULT()` for return code validation

### Pattern
```c
// Use safe wrappers:
ndb_spi_session_begin(&session);
ndb_spi_execute(&session, sql, false, 0);

// Copy data before SPI_finish:
MemoryContextSwitchTo(caller_context);
copied_data = DatumCopy(temp_data, ...);
ndb_spi_session_end(&session);
```

## Array Bounds

### Bounds Checking
- [ ] All array indexing has bounds checks
- [ ] Use `NDB_CHECK_ARRAY_BOUNDS()` macro

### Pattern
```c
// Before indexing:
NDB_CHECK_ARRAY_BOUNDS(idx, array_size, "features");
value = array[idx];
```

## Error Handling

### PG_CATCH Blocks
- [ ] All PG_TRY blocks have corresponding PG_CATCH
- [ ] All PG_CATCH blocks clean up resources properly
- [ ] All memory contexts restored in error paths

### Pattern
```c
PG_TRY();
{
    // Main code
}
PG_CATCH();
{
    // Cleanup
    nfree(ptr1);
    nfree(ptr2);
    MemoryContextSwitchTo(oldcontext);
    MemoryContextDelete(tempcontext);
    PG_RE_THROW();
}
PG_END_TRY();
```

## Function Argument Validation

### NULL Argument Checks
- [ ] All PG function arguments checked with `NDB_CHECK_NULL_ARG()`
- [ ] All non-PG function parameters checked with `NDB_CHECK_NULL()`

### Pattern
```c
// For PG functions:
NDB_CHECK_NULL_ARG(0, "model_id");
NDB_CHECK_NULL_ARG(1, "table_name");

// For regular functions:
NDB_CHECK_NULL(table_name, "table_name");
```

## Index-Specific

### IVF Index Build
- [x] **FIXED**: ivf_am.c:537 - rd_options validation added
- [ ] All index build operations validate inputs
- [ ] All reloptions validated before use

### HNSW Index
- [ ] All node accesses validate block numbers
- [ ] All neighbor count accesses validate bounds
- [ ] All memory allocations validated

## Testing Status

- [x] NULL parameter tests created
- [x] Invalid model tests created
- [x] SPI failure tests created
- [x] Memory stress tests created
- [x] Array bounds tests created
- [x] Overflow tests created
- [x] GPU failure tests created
- [x] Index crash tests created
- [x] Type confusion tests created
- [x] Concurrency tests created
- [x] Algorithm boundary tests created

## Notes

- Use `grep -r "pfree("` to find remaining pfree calls
- Use `grep -r "->"` to find pointer dereferences that need NULL checks
- Use `grep -r "SPI_"` to find SPI calls that need wrapping
- Review core dumps in `tests/core_analysis/` for crash patterns

## Progress Tracking

Last updated: 2025-01-27
Files audited: 33 files with pfree() calls
Issues found: 147 pfree() calls requiring replacement
Issues fixed: 147 pfree() calls replaced with nfree()

### Completed Tasks
- [x] Replaced all pfree() calls with nfree() in identified files
- [x] Verified all files include neurondb_macros.h header
- [x] Confirmed remaining pfree() references are only in comments

### Remaining Tasks
- [ ] Systematic NULL check audit for pointer dereferences
- [ ] Allocation size validation audit
- [ ] SPI operation safety audit
- [ ] Array bounds checking audit
- [ ] PG_CATCH block cleanup audit
- [ ] Memory context restoration audit


