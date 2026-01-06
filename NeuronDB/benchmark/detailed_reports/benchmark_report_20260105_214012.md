# NeuronDB Detailed Benchmark Report

**Generated:** 2026-01-05 16:40:12 UTC
**Branch:** REL1_STABLE
**Version:** v1.0.0-alpha

## System Information

**Test Date:** 2026-01-05 16:40:12 UTC

### Hardware

| Component | Specification |
|-----------|---------------|
| CPU | 13th Gen Intel(R) Core(TM) i5-13400F (16 cores) |
| RAM | 31.1 GB |
| GPU | NVIDIA GeForce RTX 5060, 8151 MiB |

### Software

| Component | Version |
|-----------|---------|
| PostgreSQL |  |
| NeuronDB |  |

## Benchmark Execution Summary

This report contains detailed performance metrics from comprehensive
benchmark runs across vector search, hybrid search, and RAG pipelines.

Running vector benchmarks...
Traceback (most recent call last):
  File "/home/pge/pge/neurondb/NeuronDB/benchmark/vector/run_bm.py", line 892, in main
    if not orchestrator.load_to_database():
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/pge/pge/neurondb/NeuronDB/benchmark/vector/run_bm.py", line 635, in load_to_database
    db_loader.connect()
  File "/home/pge/pge/neurondb/NeuronDB/benchmark/vector/run_bm.py", line 305, in connect
    self.conn = psycopg2.connect(**{k: v for k, v in self.db_config.items() if v is not None})
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/pge/pge/neurondb/NeuronDB/benchmark/vector/venv/lib/python3.12/site-packages/psycopg2/__init__.py", line 122, in connect
    conn = _connect(dsn, connection_factory=connection_factory, **kwasync)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
psycopg2.OperationalError: connection to server at "localhost" (127.0.0.1), port 5432 failed: FATAL:  role "neurondb" does not exist


[1m[96m
╔══════════════════════════════════════════════════════════════════════════════╗
║             NeuronDB Vector Search Benchmark Suite v1.0.0                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
[0m


[1m[95m────────────────────────────────────────────────────────────────────────────────[0m
[1m[95m  Data Preparation[0m
[1m[95m────────────────────────────────────────────────────────────────────────────────[0m

[21:40:12] [92m✓[0m Dataset 'sift-128-euclidean' already cached
[21:40:12] [92m✓[0m Data preparation completed

[1m[96m
╔══════════════════════════════════════════════════════════════════════════════╗
║             NeuronDB Vector Search Benchmark Suite v1.0.0                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
[0m


[1m[95m────────────────────────────────────────────────────────────────────────────────[0m
[1m[95m  Database Loading[0m
[1m[95m────────────────────────────────────────────────────────────────────────────────[0m

[21:40:12] [91m✗[0m Database connection failed: connection to server at "localhost" (127.0.0.1), port 5432 failed: FATAL:  role "neurondb" does not exist


[91mFatal error: connection to server at "localhost" (127.0.0.1), port 5432 failed: FATAL:  role "neurondb" does not exist
[0m
## Vector Search Benchmarks

### HNSW Index Performance

| Dataset | Dimensions | Recall@10 | Recall@100 | QPS | Avg Latency (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
|---------|------------|-----------|------------|-----|------------------|----------|----------|----------|
