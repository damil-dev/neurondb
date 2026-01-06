# NeuronDB Configuration Complete Reference

**Complete reference for all GUC (Grand Unified Configuration) variables in NeuronDB.**

> **Version:** 1.0  
> **PostgreSQL Compatibility:** 16, 17, 18  
> **Last Updated:** 2025-01-01

## Table of Contents

- [Core/Index Settings](#coreindex-settings)
- [GPU Settings](#gpu-settings)
- [LLM Settings](#llm-settings)
- [Worker Settings](#worker-settings)
- [ONNX Runtime Settings](#onnx-runtime-settings)
- [Quota Settings](#quota-settings)
- [AutoML Settings](#automl-settings)
- [Configuration Examples](#configuration-examples)

---

## Core/Index Settings

### `neurondb.hnsw_ef_search`

**Type:** Integer  
**Default:** `64`  
**Range:** `1` to `10000`  
**Context:** `PGC_USERSET`  
**Description:** Sets the `ef_search` parameter for HNSW index scans.

**Impact:**
- Higher values improve recall but increase search time
- Lower values improve speed but may reduce recall
- Recommended range: 32-256 for most use cases

**Example:**
```sql
SET neurondb.hnsw_ef_search = 128;
```

**Performance:**
- `ef_search = 32`: Fast, lower recall
- `ef_search = 64`: Balanced (default)
- `ef_search = 128`: Higher recall, slower
- `ef_search = 256+`: Very high recall, much slower

---

### `neurondb.hnsw_k`

**Type:** Integer  
**Default:** `10`  
**Range:** `1` to `1000`  
**Context:** `PGC_USERSET`  
**Description:** Sets the `k` parameter for HNSW index scans (number of nearest neighbors to return).

**Impact:**
- Controls how many results are returned
- Higher values return more results but are slower
- Should match your application's `LIMIT` clause

**Example:**
```sql
SET neurondb.hnsw_k = 20;
```

---

### `neurondb.ivf_probes`

**Type:** Integer  
**Default:** `10`  
**Range:** `1` to `1000`  
**Context:** `PGC_USERSET`  
**Description:** Sets the number of probes for IVF index scans.

**Impact:**
- Higher values improve recall but increase search time
- Lower values improve speed but may reduce recall
- Should be proportional to number of lists in IVF index

**Example:**
```sql
SET neurondb.ivf_probes = 20;
```

**Performance:**
- `ivf_probes = 5`: Fast, lower recall
- `ivf_probes = 10`: Balanced (default)
- `ivf_probes = 20`: Higher recall, slower
- `ivf_probes = 50+`: Very high recall, much slower

---

### `neurondb.ef_construction`

**Type:** Integer  
**Default:** `200`  
**Range:** `4` to `2000`  
**Context:** `PGC_USERSET`  
**Description:** Sets the `ef_construction` parameter for HNSW index builds.

**Impact:**
- Higher values improve index quality but increase build time
- Lower values build faster but may reduce index quality
- Only affects index creation, not queries

**Example:**
```sql
SET neurondb.ef_construction = 400;
```

**Performance:**
- `ef_construction = 100`: Fast build, lower quality
- `ef_construction = 200`: Balanced (default)
- `ef_construction = 400`: Higher quality, slower build
- `ef_construction = 800+`: Very high quality, much slower build

---

## GPU Settings

### `neurondb.compute_mode`

**Type:** Integer  
**Default:** `0` (CPU)  
**Range:** `0` to `2`  
**Context:** `PGC_USERSET`  
**Description:** Controls whether ML operations run on CPU, GPU, or auto-select.

**Values:**
- `0` (cpu): CPU only, don't initialize GPU
- `1` (gpu): GPU required, error if unavailable
- `2` (auto): Try GPU first, fallback to CPU

**Example:**
```sql
SET neurondb.compute_mode = 2;  -- Auto-select
```

**Impact:**
- `0`: No GPU initialization, CPU-only operations
- `1`: GPU required, fails if GPU unavailable
- `2`: Best of both worlds, automatic fallback

---

### `neurondb.gpu_backend_type`

**Type:** Integer  
**Default:** Platform-specific  
**Range:** `0` to `2`  
**Context:** `PGC_USERSET`  
**Description:** Selects GPU backend implementation.

**Values:**
- `0` (cuda): NVIDIA CUDA
- `1` (rocm): AMD ROCm
- `2` (metal): Apple Metal

**Default by Platform:**
- Linux (NVIDIA): `0` (CUDA)
- Linux (AMD): `1` (ROCm)
- macOS: `2` (Metal)

**Note:** Only valid when `neurondb.compute_mode` is `1` (gpu) or `2` (auto). Ignored when `compute_mode` is `0` (cpu).

**Example:**
```sql
SET neurondb.gpu_backend_type = 0;  -- CUDA
```

---

### `neurondb.gpu_device`

**Type:** Integer  
**Default:** `0`  
**Range:** `0` to `16`  
**Context:** `PGC_USERSET`  
**Description:** GPU device ID to use (0-based).

**Example:**
```sql
SET neurondb.gpu_device = 1;  -- Use second GPU
```

---

### `neurondb.gpu_batch_size`

**Type:** Integer  
**Default:** `8192`  
**Range:** `64` to `65536`  
**Context:** `PGC_USERSET`  
**Description:** Batch size for GPU operations.

**Impact:**
- Larger batches improve GPU utilization
- Smaller batches reduce memory usage
- Optimal value depends on GPU memory and workload

**Example:**
```sql
SET neurondb.gpu_batch_size = 16384;
```

---

### `neurondb.gpu_streams`

**Type:** Integer  
**Default:** `2`  
**Range:** `1` to `8`  
**Context:** `PGC_USERSET`  
**Description:** Number of CUDA/HIP streams for parallel operations.

**Impact:**
- More streams enable better parallelism
- Too many streams can cause overhead
- Recommended: 2-4 for most GPUs

**Example:**
```sql
SET neurondb.gpu_streams = 4;
```

---

### `neurondb.gpu_memory_pool_mb`

**Type:** Real (double precision)  
**Default:** `512.0`  
**Range:** `64.0` to `32768.0`  
**Context:** `PGC_USERSET`  
**Description:** GPU memory pool size in MB.

**Impact:**
- Larger pools reduce allocation overhead
- Smaller pools leave more memory for other operations
- Should be less than available GPU memory

**Example:**
```sql
SET neurondb.gpu_memory_pool_mb = 1024.0;
```

---

### `neurondb.gpu_kernels`

**Type:** String  
**Default:** `"l2,cosine,ip,rf_split,rf_predict"`  
**Context:** `PGC_USERSET`  
**Description:** List of GPU-accelerated kernels (comma-separated).

**Available Kernels:**
- `l2`: L2 distance
- `cosine`: Cosine distance
- `ip`: Inner product
- `rf_split`: Random Forest split
- `rf_predict`: Random Forest prediction

**Example:**
```sql
SET neurondb.gpu_kernels = 'l2,cosine,ip';
```

---

### `neurondb.gpu_timeout_ms`

**Type:** Integer  
**Default:** `30000` (30 seconds)  
**Range:** `1000` to `300000`  
**Context:** `PGC_USERSET`  
**Description:** GPU kernel execution timeout in milliseconds.

**Impact:**
- Prevents hung GPU operations
- Too short may cancel legitimate long operations
- Too long may delay error detection

**Example:**
```sql
SET neurondb.gpu_timeout_ms = 60000;  -- 60 seconds
```

---

## LLM Settings

### `neurondb.llm_provider`

**Type:** String  
**Default:** `"huggingface"`  
**Context:** `PGC_USERSET`  
**Description:** LLM provider.

**Options:**
- `"huggingface"`: Hugging Face API (default)
- `"huggingface-local"`: Local Hugging Face models
- `"hf-local"`: Alias for huggingface-local
- `"hf-http"`: Hugging Face HTTP API
- `"openai"`: OpenAI API

**Example:**
```sql
SET neurondb.llm_provider = 'huggingface-local';
```

---

### `neurondb.llm_model`

**Type:** String  
**Default:** `"sentence-transformers/all-MiniLM-L6-v2"`  
**Context:** `PGC_USERSET`  
**Description:** Default LLM model ID.

**Example:**
```sql
SET neurondb.llm_model = 'sentence-transformers/all-mpnet-base-v2';
```

---

### `neurondb.llm_endpoint`

**Type:** String  
**Default:** `"https://router.huggingface.co"`  
**Context:** `PGC_USERSET`  
**Description:** LLM endpoint base URL.

**Example:**
```sql
SET neurondb.llm_endpoint = 'https://api-inference.huggingface.co';
```

---

### `neurondb.llm_api_key`

**Type:** String  
**Default:** `""` (empty)  
**Context:** `PGC_SUSET` (superuser only)  
**Description:** LLM API key (set via ALTER SYSTEM or environment variable).

**Security:** Superuser-only setting. Should be set via:
- `ALTER SYSTEM SET neurondb.llm_api_key = 'key';`
- Environment variable: `NEURONDB_LLM_API_KEY`

**Example:**
```sql
ALTER SYSTEM SET neurondb.llm_api_key = 'hf_xxxxxxxxxxxxx';
SELECT pg_reload_conf();
```

---

### `neurondb.llm_timeout_ms`

**Type:** Integer  
**Default:** `30000` (30 seconds)  
**Range:** `1000` to `600000`  
**Context:** `PGC_USERSET`  
**Description:** HTTP timeout in milliseconds.

**Impact:**
- Too short may timeout on slow networks
- Too long may delay error detection
- Recommended: 30-60 seconds

**Example:**
```sql
SET neurondb.llm_timeout_ms = 60000;  -- 60 seconds
```

---

### `neurondb.llm_cache_ttl`

**Type:** Integer  
**Default:** `600` (10 minutes)  
**Range:** `0` to `86400`  
**Context:** `PGC_USERSET`  
**Description:** Cache TTL in seconds.

**Impact:**
- `0`: No caching
- `600`: 10 minutes (default)
- `3600`: 1 hour
- `86400`: 24 hours

**Example:**
```sql
SET neurondb.llm_cache_ttl = 3600;  -- 1 hour
```

---

### `neurondb.llm_rate_limiter_qps`

**Type:** Integer  
**Default:** `5`  
**Range:** `1` to `10000`  
**Context:** `PGC_USERSET`  
**Description:** Rate limiter queries per second.

**Impact:**
- Prevents API rate limit errors
- Lower values more conservative
- Higher values more aggressive

**Example:**
```sql
SET neurondb.llm_rate_limiter_qps = 10;
```

---

### `neurondb.llm_fail_open`

**Type:** Boolean  
**Default:** `true`  
**Context:** `PGC_USERSET`  
**Description:** Fail open on provider errors.

**Impact:**
- `true`: Return error instead of failing query
- `false`: Query fails on LLM errors

**Example:**
```sql
SET neurondb.llm_fail_open = false;
```

---

## Worker Settings

### NeurANQ (Queue Executor)

#### `neurondb.neuranq_naptime`

**Type:** Integer  
**Default:** `1000` (1 second)  
**Range:** `100` to `60000`  
**Context:** `PGC_SIGHUP`  
**Description:** Duration between job processing cycles in milliseconds.

**Example:**
```sql
SET neurondb.neuranq_naptime = 2000;  -- 2 seconds
```

#### `neurondb.neuranq_queue_depth`

**Type:** Integer  
**Default:** `10000`  
**Range:** `100` to `1000000`  
**Context:** `PGC_SIGHUP`  
**Description:** Maximum job queue size.

**Example:**
```sql
SET neurondb.neuranq_queue_depth = 50000;
```

#### `neurondb.neuranq_batch_size`

**Type:** Integer  
**Default:** `100`  
**Range:** `1` to `10000`  
**Context:** `PGC_SIGHUP`  
**Description:** Jobs to process per cycle.

**Example:**
```sql
SET neurondb.neuranq_batch_size = 200;
```

#### `neurondb.neuranq_timeout`

**Type:** Integer  
**Default:** `30000` (30 seconds)  
**Range:** `1000` to `300000`  
**Context:** `PGC_SIGHUP`  
**Description:** Job execution timeout in milliseconds.

**Example:**
```sql
SET neurondb.neuranq_timeout = 60000;  -- 60 seconds
```

#### `neurondb.neuranq_max_retries`

**Type:** Integer  
**Default:** `3`  
**Range:** `0` to `10`  
**Context:** `PGC_SIGHUP`  
**Description:** Maximum retry attempts per job.

**Example:**
```sql
SET neurondb.neuranq_max_retries = 5;
```

#### `neurondb.neuranq_enabled`

**Type:** Boolean  
**Default:** `true`  
**Context:** `PGC_SIGHUP`  
**Description:** Enable queue worker.

**Example:**
```sql
SET neurondb.neuranq_enabled = false;  -- Disable worker
```

---

### NeurANMon (Auto-Tuner)

#### `neurondb.neuranmon_naptime`

**Type:** Integer  
**Default:** `60000` (60 seconds)  
**Range:** `10000` to `600000`  
**Context:** `PGC_SIGHUP`  
**Description:** Duration between tuning cycles in milliseconds.

**Example:**
```sql
SET neurondb.neuranmon_naptime = 120000;  -- 2 minutes
```

#### `neurondb.neuranmon_sample_size`

**Type:** Integer  
**Default:** `1000`  
**Range:** `100` to `100000`  
**Context:** `PGC_SIGHUP`  
**Description:** Number of queries to sample.

**Example:**
```sql
SET neurondb.neuranmon_sample_size = 5000;
```

#### `neurondb.neuranmon_target_latency`

**Type:** Real (double precision)  
**Default:** `100.0` (100 ms)  
**Range:** `1.0` to `10000.0`  
**Context:** `PGC_SIGHUP`  
**Description:** Target query latency in milliseconds.

**Example:**
```sql
SET neurondb.neuranmon_target_latency = 50.0;  -- 50 ms
```

#### `neurondb.neuranmon_target_recall`

**Type:** Real (double precision)  
**Default:** `0.95`  
**Range:** `0.5` to `1.0`  
**Context:** `PGC_SIGHUP`  
**Description:** Target recall@k threshold.

**Example:**
```sql
SET neurondb.neuranmon_target_recall = 0.98;  -- 98% recall
```

#### `neurondb.neuranmon_enabled`

**Type:** Boolean  
**Default:** `true`  
**Context:** `PGC_SIGHUP`  
**Description:** Enable tuner worker.

**Example:**
```sql
SET neurondb.neuranmon_enabled = false;  -- Disable tuner
```

---

### NeurANDefrag (Index Maintenance)

#### `neurondb.neurandefrag_naptime`

**Type:** Integer  
**Default:** `300000` (5 minutes)  
**Range:** `60000` to `3600000`  
**Context:** `PGC_SIGHUP`  
**Description:** Duration between maintenance cycles in milliseconds.

**Example:**
```sql
SET neurondb.neurandefrag_naptime = 600000;  -- 10 minutes
```

#### `neurondb.neurandefrag_compact_threshold`

**Type:** Integer  
**Default:** `10000`  
**Range:** `1000` to `1000000`  
**Context:** `PGC_SIGHUP`  
**Description:** Edge count threshold for compaction trigger.

**Example:**
```sql
SET neurondb.neurandefrag_compact_threshold = 50000;
```

#### `neurondb.neurandefrag_fragmentation_threshold`

**Type:** Real (double precision)  
**Default:** `0.3`  
**Range:** `0.1` to `0.9`  
**Context:** `PGC_SIGHUP`  
**Description:** Fragmentation ratio necessary to trigger a full rebuild.

**Example:**
```sql
SET neurondb.neurandefrag_fragmentation_threshold = 0.5;  -- 50% fragmentation
```

#### `neurondb.neurandefrag_maintenance_window`

**Type:** String  
**Default:** `"02:00-04:00"`  
**Context:** `PGC_SIGHUP`  
**Description:** Maintenance window in HH:MM-HH:MM format.

**Example:**
```sql
SET neurondb.neurandefrag_maintenance_window = '03:00-05:00';
```

#### `neurondb.neurandefrag_enabled`

**Type:** Boolean  
**Default:** `true`  
**Context:** `PGC_SIGHUP`  
**Description:** Enable/disable the NeurANDefrag background worker.

**Example:**
```sql
SET neurondb.neurandefrag_enabled = false;  -- Disable maintenance
```

---

## ONNX Runtime Settings

### `neurondb.onnx_model_path`

**Type:** String  
**Default:** `"/var/lib/neurondb/models"`  
**Context:** `PGC_SUSET` (superuser only)  
**Description:** Directory with ONNX model files.

**Note:** Files exported from HuggingFace transformers in ONNX format must be placed under this directory.

**Example:**
```sql
ALTER SYSTEM SET neurondb.onnx_model_path = '/opt/neurondb/models';
SELECT pg_reload_conf();
```

---

### `neurondb.onnx_use_gpu`

**Type:** Boolean  
**Default:** `true`  
**Context:** `PGC_SUSET` (superuser only)  
**Description:** Attempt to use GPU acceleration for ONNX inference.

**Impact:**
- If enabled, CUDA (NVIDIA) or CoreML (macOS) execution will be tried before falling back to CPU
- Requires ONNX Runtime with GPU support

**Example:**
```sql
ALTER SYSTEM SET neurondb.onnx_use_gpu = true;
SELECT pg_reload_conf();
```

---

### `neurondb.onnx_threads`

**Type:** Integer  
**Default:** `4`  
**Range:** `1` to `64`  
**Context:** `PGC_SUSET` (superuser only)  
**Description:** Number of ONNX Runtime intra-operator threads.

**Impact:**
- Controls the intra-op-thread pool for ONNX inference
- More threads improve parallelism but increase memory usage
- Recommended: 4-8 for most systems

**Example:**
```sql
ALTER SYSTEM SET neurondb.onnx_threads = 8;
SELECT pg_reload_conf();
```

---

### `neurondb.onnx_cache_size`

**Type:** Integer  
**Default:** `10`  
**Range:** `1` to `100`  
**Context:** `PGC_SUSET` (superuser only)  
**Description:** ONNX model LRU cache size (number of sessions).

**Impact:**
- When this limit is reached, the least recently used session will be evicted
- Larger cache reduces reload overhead but uses more memory
- Recommended: 10-20 for most systems

**Example:**
```sql
ALTER SYSTEM SET neurondb.onnx_cache_size = 20;
SELECT pg_reload_conf();
```

---

## Quota Settings

### `neurondb.default_max_vectors`

**Type:** Integer (int64)  
**Default:** `1000000` (1 million)  
**Range:** `1000` to `INT_MAX`  
**Context:** `PGC_SIGHUP`  
**Description:** Default maximum vectors per tenant (in thousands).

**Example:**
```sql
SET neurondb.default_max_vectors = 5000000;  -- 5 million
```

---

### `neurondb.default_max_storage_mb`

**Type:** Integer (int64)  
**Default:** `10240` (10 GB)  
**Range:** `100` to `INT_MAX`  
**Context:** `PGC_SIGHUP`  
**Description:** Default maximum storage (MB) per tenant.

**Example:**
```sql
SET neurondb.default_max_storage_mb = 51200;  -- 50 GB
```

---

### `neurondb.default_max_qps`

**Type:** Integer  
**Default:** `1000`  
**Range:** `1` to `INT_MAX`  
**Context:** `PGC_SIGHUP`  
**Description:** Default maximum queries per second per tenant.

**Example:**
```sql
SET neurondb.default_max_qps = 5000;
```

---

### `neurondb.enforce_quotas`

**Type:** Boolean  
**Default:** `true`  
**Context:** `PGC_SUSET` (superuser only)  
**Description:** Enable hard quota enforcement.

**Impact:**
- `true`: Quotas are enforced, operations fail when exceeded
- `false`: Quotas are monitored but not enforced

**Example:**
```sql
ALTER SYSTEM SET neurondb.enforce_quotas = true;
SELECT pg_reload_conf();
```

---

## AutoML Settings

### `neurondb.automl.use_gpu`

**Type:** Boolean  
**Default:** `false`  
**Context:** `PGC_USERSET`  
**Description:** Enable GPU acceleration for AutoML training.

**Impact:**
- When enabled, AutoML will prefer GPU training for supported algorithms
- Requires GPU backend to be configured

**Example:**
```sql
SET neurondb.automl.use_gpu = true;
```

---

### `neurondb.vector_capsule_enabled`

**Type:** Boolean  
**Default:** `false`  
**Context:** `PGC_USERSET`  
**Description:** Enable VectorCapsule features (multi-representation vectors with metadata).

**Impact:**
- When enabled, allows creation of VectorCapsule types with:
  - Adaptive representation selection
  - Integrity checking
  - Provenance tracking

**Example:**
```sql
SET neurondb.vector_capsule_enabled = true;
```

---

## Configuration Examples

### High-Performance Setup

```sql
-- Index settings for high recall
SET neurondb.hnsw_ef_search = 128;
SET neurondb.ef_construction = 400;
SET neurondb.ivf_probes = 20;

-- GPU acceleration
SET neurondb.compute_mode = 2;  -- Auto
SET neurondb.gpu_batch_size = 16384;
SET neurondb.gpu_streams = 4;

-- LLM settings
SET neurondb.llm_provider = 'huggingface-local';
SET neurondb.llm_cache_ttl = 3600;  -- 1 hour
SET neurondb.llm_rate_limiter_qps = 10;
```

### Memory-Constrained Setup

```sql
-- Index settings for lower memory
SET neurondb.hnsw_ef_search = 32;
SET neurondb.ef_construction = 100;
SET neurondb.ivf_probes = 5;

-- GPU settings
SET neurondb.gpu_memory_pool_mb = 256.0;
SET neurondb.gpu_batch_size = 4096;

-- LLM settings
SET neurondb.llm_cache_ttl = 300;  -- 5 minutes
```

### Development Setup

```sql
-- Fast index builds
SET neurondb.ef_construction = 100;

-- CPU-only (no GPU required)
SET neurondb.compute_mode = 0;

-- Minimal caching
SET neurondb.llm_cache_ttl = 60;  -- 1 minute

-- Disable workers for testing
SET neurondb.neuranq_enabled = false;
SET neurondb.neuranmon_enabled = false;
SET neurondb.neurandefrag_enabled = false;
```

### Production Setup

```sql
-- High-quality indexes
SET neurondb.ef_construction = 400;
SET neurondb.hnsw_ef_search = 128;

-- GPU acceleration
SET neurondb.compute_mode = 2;  -- Auto
SET neurondb.gpu_device = 0;
SET neurondb.gpu_batch_size = 8192;

-- LLM with caching
SET neurondb.llm_provider = 'huggingface-local';
SET neurondb.llm_cache_ttl = 3600;  -- 1 hour
SET neurondb.llm_rate_limiter_qps = 10;

-- Workers enabled
SET neurondb.neuranq_enabled = true;
SET neurondb.neuranmon_enabled = true;
SET neurondb.neurandefrag_enabled = true;

-- Quotas
SET neurondb.default_max_vectors = 10000000;  -- 10M
SET neurondb.default_max_storage_mb = 102400;  -- 100 GB
SET neurondb.enforce_quotas = true;
```

---

## Configuration Contexts

### `PGC_USERSET`

Can be set by any user in any session.

**Examples:**
- `neurondb.hnsw_ef_search`
- `neurondb.compute_mode`
- `neurondb.llm_provider`

### `PGC_SUSET`

Can only be set by superuser, requires `ALTER SYSTEM` or server restart.

**Examples:**
- `neurondb.llm_api_key`
- `neurondb.onnx_model_path`
- `neurondb.enforce_quotas`

### `PGC_SIGHUP`

Requires server reload (`SELECT pg_reload_conf();`) or restart.

**Examples:**
- `neurondb.neuranq_naptime`
- `neurondb.neuranmon_naptime`
- `neurondb.neurandefrag_naptime`

---

## Viewing Configuration

### Show All NeuronDB Settings

```sql
SELECT name, setting, unit, context, vartype, min_val, max_val, boot_val, reset_val
FROM pg_settings
WHERE name LIKE 'neurondb.%'
ORDER BY name;
```

### Show Current Values

```sql
SHOW neurondb.hnsw_ef_search;
SHOW neurondb.compute_mode;
SHOW neurondb.llm_provider;
```

### Show All Settings in Category

```sql
-- Core/index settings
SELECT name, setting FROM pg_settings
WHERE name IN (
    'neurondb.hnsw_ef_search',
    'neurondb.hnsw_k',
    'neurondb.ivf_probes',
    'neurondb.ef_construction'
);

-- GPU settings
SELECT name, setting FROM pg_settings
WHERE name LIKE 'neurondb.gpu%';

-- LLM settings
SELECT name, setting FROM pg_settings
WHERE name LIKE 'neurondb.llm%';
```

---

## Configuration Best Practices

### Index Settings

1. **Start with defaults** and tune based on workload
2. **ef_construction**: Higher for production, lower for development
3. **ef_search**: Balance between recall and latency
4. **ivf_probes**: Proportional to number of lists

### GPU Settings

1. **compute_mode**: Use `2` (auto) for flexibility
2. **gpu_batch_size**: Larger for better utilization, smaller for memory constraints
3. **gpu_streams**: 2-4 for most GPUs
4. **gpu_memory_pool_mb**: Leave headroom for other operations

### LLM Settings

1. **llm_provider**: Use `huggingface-local` for better performance
2. **llm_cache_ttl**: Longer for stable workloads, shorter for development
3. **llm_rate_limiter_qps**: Match your API limits
4. **llm_fail_open**: `true` for graceful degradation

### Worker Settings

1. **Enable workers** in production for automatic optimization
2. **Tune naptime** based on workload frequency
3. **Set maintenance windows** during low-traffic periods
4. **Monitor worker logs** for optimization insights

---

## Related Documentation

- [SQL API Reference](sql-api-complete.md)
- [Data Types Reference](data-types.md)
- [Index Methods](../internals/index-methods.md)
- [GPU Acceleration](../advanced/gpu-acceleration-complete.md)
- [Background Workers](../internals/background-workers.md)

---

**Last Updated:** 2025-01-01  
**Documentation Version:** 1.0.0



