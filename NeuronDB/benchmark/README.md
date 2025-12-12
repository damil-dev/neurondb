# NeuronDB Benchmark Suite

A modular and extensible benchmark tool for NeuronDB, designed to compare performance against pgvector and other systems.

## Features

- **Modular Architecture**: Clean separation of concerns with easy extensibility
- **Vector Benchmarks**: Comprehensive vector search benchmarking
- **Fair Comparison**: Same queries, same data, same hardware
- **Multiple Output Formats**: Console, JSON, CSV
- **Extensive Metrics**: Latency (p50, p95, p99), throughput, recall, index sizes

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Vector Benchmark

```bash
python neurondb_bm.py --vector \
  --neurondb-dsn "host=localhost dbname=neurondb user=postgres"
```

### Compare with pgvector

```bash
python neurondb_bm.py --vector \
  --neurondb-dsn "host=localhost dbname=neurondb user=postgres" \
  --pgvector-dsn "host=localhost dbname=pgvector user=postgres"
```

### Custom Configuration

```bash
python neurondb_bm.py --vector \
  --dimensions 128,384,768 \
  --sizes 1000,10000,100000 \
  --metrics l2,cosine \
  --k-values 1,10,100 \
  --iterations 200 \
  --output console,json,csv \
  --output-file results
```

## Command-Line Options

### Benchmark Selection
- `--vector`: Run vector search benchmarks
- `--embeddings`: Run embedding benchmarks (future)
- `--ml`: Run ML benchmarks (future)

### Database Connections
- `--neurondb-dsn`: NeuronDB connection string (or set `NEURONDB_DSN` env var)
- `--pgvector-dsn`: pgvector connection string for comparison (or set `PGVECTOR_DSN` env var)

### Benchmark Parameters
- `--dimensions`: Comma-separated vector dimensions (default: 128,384,768,1536)
- `--sizes`: Comma-separated dataset sizes (default: 1000,10000,100000)
- `--metrics`: Distance metrics: l2,cosine,inner_product (default: all)
- `--k-values`: K values for KNN search (default: 1,10,100)
- `--index` / `--no-index`: Test with/without indexes (default: with index)
- `--iterations`: Number of query iterations per test (default: 100)
- `--warmup`: Number of warmup iterations (default: 10)

### Output
- `--output`: Output formats: console,json,csv,all (default: console)
- `--output-file`: Output file path for JSON/CSV

## Architecture

```
neurondb_bm.py (main entry point)
├── modules/
│   ├── base.py (Base benchmark class)
│   └── vector.py (Vector benchmarks)
├── utils/
│   ├── database.py (DB connection management)
│   ├── data_generator.py (Synthetic data generation)
│   ├── metrics.py (Performance metrics)
│   └── output.py (Results formatting)
└── config.py (Configuration management)
```

## Output Metrics

Each benchmark run collects:
- **Latency**: p50, p95, p99, mean, min, max (in milliseconds)
- **Throughput**: Queries per second (QPS)
- **Accuracy**: Recall@K
- **Index Metrics**: Build time, index size
- **Table Metrics**: Table size, insertion time

## Examples

### Quick Test
```bash
python neurondb_bm.py --vector \
  --dimensions 128 \
  --sizes 1000 \
  --iterations 10
```

### Full Benchmark Suite
```bash
python neurondb_bm.py --vector \
  --dimensions 128,384,768,1536 \
  --sizes 1000,10000,100000,1000000 \
  --metrics l2,cosine,inner_product \
  --k-values 1,10,100 \
  --iterations 1000 \
  --output all \
  --output-file full_benchmark
```

## Environment Variables

- `NEURONDB_DSN`: Default NeuronDB connection string
- `PGVECTOR_DSN`: Default pgvector connection string

## Requirements

- Python 3.8+
- PostgreSQL 12+ with NeuronDB extension
- psycopg2-binary
- numpy
- tabulate

## License

See main project LICENSE file.

