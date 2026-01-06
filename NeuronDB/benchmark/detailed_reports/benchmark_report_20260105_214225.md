# NeuronDB Detailed Benchmark Report

**Generated:** 2026-01-05 16:42:25 UTC
**Branch:** REL1_STABLE
**Version:** v1.0.0-alpha

## System Information

**Test Date:** 2026-01-05 16:42:25 UTC


### Hardware

| Component | Specification |
|-----------|---------------|
| CPU | 13th Gen Intel(R) Core(TM) i5-13400F (16 cores) |
| RAM | 31.1 GB |
| GPU | NVIDIA GeForce RTX 5060, 8151 MiB |

### Software

| Component | Version |
|-----------|---------|
| PostgreSQL | 18.1 |


## Benchmark Execution Summary

This report contains detailed performance metrics from comprehensive
benchmark runs across vector search, hybrid search, and RAG pipelines.

## Vector Search Benchmarks

### Test Configuration

- **Dataset:** sift-128-euclidean
- **Dimensions:** 128
- **Training Vectors:** 0
- **Test Queries:** 0


### HNSW Index Performance

| Dataset | Dimensions | Index Type | Recall@10 | Recall@100 | QPS | Avg Latency (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
|---------|------------|------------|-----------|------------|-----|------------------|----------|----------|----------|
| sift-128-euclidean | 128 | hnsw | 1.000 | 1.000 | 1.91 | 523.45 | 518.12 | 549.70 | 569.82 |
| sift-128-euclidean | 128 | hnsw | 1.000 | 1.000 | 1.93 | 518.76 | 518.11 | 527.41 | 539.12 |
| sift-128-euclidean | 128 | hnsw | 1.000 | 1.000 | 1.91 | 523.51 | 520.66 | 545.32 | 561.41 |
| sift-128-euclidean | 128 | hnsw | 1.000 | 1.000 | 1.94 | 515.50 | 514.14 | 528.74 | 541.57 |
| sift-128-euclidean | 128 | hnsw | 1.000 | 1.000 | 1.95 | 512.98 | 512.39 | 521.40 | 521.79 |


**Average Performance:** QPS: 1.93, Avg Latency: 518.84 ms



## Hybrid Search Benchmarks

No hybrid benchmark results found.

## RAG Pipeline Benchmarks

RAG pipeline benchmarks evaluate end-to-end retrieval-augmented generation.

Found 1 RAG benchmark result(s).

## Reproducing These Results

To reproduce these benchmarks:

```bash
cd NeuronDB/benchmark
./run_bm.sh
```

## Notes

- Results may vary based on hardware, dataset size, and system load
- GPU results require CUDA/ROCm/Metal support
- See individual benchmark directories for detailed configuration
