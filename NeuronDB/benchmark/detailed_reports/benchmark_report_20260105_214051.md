# NeuronDB Detailed Benchmark Report

**Generated:** 2026-01-05 16:40:51 UTC
**Branch:** REL1_STABLE
**Version:** v1.0.0-alpha

## System Information

**Test Date:** 2026-01-05 16:40:51 UTC


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

### HNSW Index Performance

| No results available | - | - | - | - | - | - | - | - |


### IVF Index Performance

| No results available | - | - | - | - | - | - | - | - |

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
