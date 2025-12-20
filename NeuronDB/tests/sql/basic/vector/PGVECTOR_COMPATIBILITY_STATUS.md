# Compatibility Test Status

## Overview
This document tracks the status of compatibility tests and implementation.

## Test Files Created
All test files have been created in `tests/sql/basic/`:
- 058_vector_type.sql - Vector type operations
- 059_cast.sql - Type casting
- 060_hnsw_vector.sql - HNSW index for vector
- 061_ivfflat_vector.sql - IVFFlat (IVF) index for vector
- 062_hnsw_halfvec.sql - HNSW index for halfvec
- 063_hnsw_sparsevec.sql - HNSW index for sparsevec
- 064_hnsw_bit.sql - HNSW index for bit
- 065_ivfflat_halfvec.sql - IVFFlat index for halfvec
- 066_ivfflat_bit.sql - IVFFlat index for bit
- 067_halfvec.sql - Halfvec type operations
- 068_sparsevec.sql - Sparsevec type operations
- 069_bit.sql - Bit type operations
- 070_copy.sql - COPY command tests
- 071_btree.sql - B-tree index tests

## Compatibility Functions Added
The following compatible function aliases have been added to `neurondb--1.0.sql`:

### Vector Functions
- `l2_distance(vector, vector)` -> `vector_l2_distance`
- `inner_product(vector, vector)` -> `vector_inner_product`
- `cosine_distance(vector, vector)` -> `vector_cosine_distance`
- `l1_distance(vector, vector)` -> `vector_l1_distance`
- `l2_normalize(vector)` -> `vector_normalize`
- `vector_cmp(vector, vector)` -> comparison function for btree

### Halfvec Functions
- `l2_distance(halfvec, halfvec)` -> `halfvec_l2_distance`
- `inner_product(halfvec, halfvec)` -> `halfvec_inner_product`
- `cosine_distance(halfvec, halfvec)` -> `halfvec_cosine_distance`
- `l2_norm(halfvec)` -> converts to vector for calculation
- `l2_normalize(halfvec)` -> converts to vector for calculation
- `halfvec_cmp(halfvec, halfvec)` -> comparison function

### Sparsevec Functions
- `l2_norm(sparsevec)` -> `sparsevec_l2_norm`
- `sparsevec_cmp(sparsevec, sparsevec)` -> comparison function

### Bit Functions
- `hamming_distance(bit, bit)` -> `bit_hamming_distance`
- `jaccard_distance(bit, bit)` -> `bit_jaccard_distance`

## Recent Fixes (December 2024)

### 1. Arithmetic Operators ✅
- **Halfvec operators implemented**: Added `halfvec_add`, `halfvec_sub`, `halfvec_mul`, `halfvec_div`, `halfvec_neg` functions
- **SQL operators added**: `+`, `-`, `*`, `/`, and unary `-` for halfvec type
- **Sparsevec operators implemented**: Added `sparsevec_add`, `sparsevec_sub`, `sparsevec_mul` functions
- **SQL operators added**: `+`, `-`, `*` for sparsevec type

### 2. Index Crash Fixes ✅
- **HNSW bit type crash fixed**: Added null check for `bit_data` in `hnswExtractVectorData` function
- This prevents server crashes when processing bit vectors in HNSW indexes

### 3. Empty Sparsevec Support ✅
- **Empty format support**: Modified `sparsevec_in` to support `{}/dim` format
- Allows creation of empty sparse vectors with explicit dimension specification

### 4. B-tree Operator Classes ✅
- **Vector btree operator class**: Added `vector_btree_ops` for btree indexes on vector type
- Note: Halfvec and sparsevec btree classes require `<`, `<=`, `>`, `>=` operators (not yet implemented)

## Known Issues and Limitations

### 1. Index Issues (Partially Fixed)
- HNSW index crashes for bit type have been fixed with null check
- Some index operations may still need testing and verification
- IVFFlat index uses `ivf` access method name instead of `ivfflat` (intentional difference)

### 3. Function Name Differences
- NeuronDB uses `vector_l2_distance` instead of `l2_distance` (aliases added)
- NeuronDB uses `vector_normalize` instead of `l2_normalize` (aliases added)
- Some functions may have different parameter signatures

### 4. Type Casting
- Some casting operations may behave differently
- Need to verify all cast combinations work correctly

### 5. Comparison Functions
- `vector_cmp`, `halfvec_cmp`, `sparsevec_cmp` functions added but may need refinement
- B-tree indexes may require additional operator class setup

## Next Steps

1. **Fix Index Issues**: Investigate and fix HNSW/IVF index crashes
2. **Add Missing Operators**: Implement arithmetic operators for halfvec and sparsevec
3. **Verify Casting**: Test all type casting combinations
4. **Complete Operator Classes**: Ensure all operator classes are properly configured
5. **Test Edge Cases**: Verify edge cases (NaN, Infinity, overflow, underflow) are handled correctly

## Test Execution
Run tests using:
```bash
cd NeuronDB/tests
python3 run_test.py --category basic --test <test_name>
```

Or run all compatibility tests:
```bash
python3 run_test.py --category basic
```

