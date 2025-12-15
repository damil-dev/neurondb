# NeuronDB Benchmark Suite

A modular and extensible benchmark tool for NeuronDB, designed to compare performance against pgvector and other systems.

## Features

- **Modular Architecture**: Clean separation of concerns with easy extensibility
- **Vector Benchmarks**: Comprehensive vector search benchmarking
- **Fair Comparison**: Same queries, same data, same hardware
- **Multiple Output Formats**: Console, JSON, CSV
- **Extensive Metrics**: Latency (p50, p95, p99), throughput, recall, index sizes
- **Result Validation**: Automatic accuracy checking and ground truth comparison
- **Summary Statistics**: Geometric mean calculations and speedup ratios
- **Easy Runner Script**: Convenient shell script with pre-flight checks

## Quick Start

### Using the Runner Script (Recommended)

The easiest way to run benchmarks is using the provided shell script:

```bash
# Set your database connection strings
export NEURONDB_DSN="host=localhost dbname=neurondb user=postgres"
export PGVECTOR_DSN="host=localhost dbname=pgvector user=postgres"  # Optional

# Run benchmark with defaults
./run_benchmark.sh

# Or customize parameters
export DIMENSIONS="128,384,768"
export SIZES="1000,10000,100000"
export ITERATIONS=200
./run_benchmark.sh
```

The script will:
- Check Python dependencies
- Verify database connections
- Ensure extensions are installed
- Run the benchmark
- Save results to `./results/`

### Manual Installation

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
  --index-m 16 \
  --index-ef-construction 200 \
  --output console,json,csv \
  --output-file results
```

### Sequential Scan Comparison

To compare indexed vs sequential scan performance:

```bash
python neurondb_bm.py --vector \
  --neurondb-dsn "host=localhost dbname=neurondb user=postgres" \
  --no-index \
  --output all \
  --output-file sequential_scan_results
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
- `--index-m`: HNSW M parameter (default: 16)
- `--index-ef-construction`: HNSW ef_construction parameter (default: 200)
- `--compare-sequential-scan`: Also benchmark sequential scan for comparison

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
- **Accuracy**: Recall@K (validated against ground truth)
- **Index Metrics**: Build time, index size
- **Table Metrics**: Table size, insertion time
- **Summary Statistics**: Geometric mean across all scenarios
- **Speedup Ratios**: Performance comparison between systems

### Result Validation

The benchmark automatically validates results:
- Computes ground truth using exact distance calculations
- Compares database results with ground truth
- Reports recall@K accuracy
- Warns if distance calculations don't match expected values

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

## Setup Requirements

### Database Setup

1. **PostgreSQL 12+** installed and running
2. **NeuronDB extension** installed:
   ```sql
   CREATE DATABASE neurondb;
   \c neurondb
   CREATE EXTENSION neurondb;
   ```

3. **pgvector extension** (optional, for comparison):
   ```sql
   CREATE DATABASE pgvector;
   \c pgvector
   CREATE EXTENSION vector;
   ```

### Python Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- Python 3.8+
- psycopg2-binary
- numpy
- tabulate

## Troubleshooting

### Extension Not Found

If you see errors about missing extensions:

```bash
# Check if extension is installed
psql -d neurondb -c "SELECT * FROM pg_extension WHERE extname = 'neurondb';"

# Install if missing
psql -d neurondb -c "CREATE EXTENSION neurondb;"
```

### Connection Errors

Verify your connection string format:
```bash
# Test connection
psql "host=localhost dbname=neurondb user=postgres" -c "SELECT 1;"
```

### Index Creation Fails

If index creation fails with large datasets:
- The benchmark automatically retries with smaller parameters
- You can manually specify smaller index parameters: `--index-m 8 --index-ef-construction 100`

### Low Recall Scores

If recall@K is low:
- This may indicate index quality issues
- Try increasing `--index-ef-construction` for better accuracy
- Check that vectors are properly normalized (benchmark does this automatically)

## Output Format

### Console Output

The console shows:
- Per-scenario results with key metrics
- Side-by-side comparison tables (when comparing systems)
- Summary statistics with geometric means and speedup ratios

### JSON Output

JSON files contain:
- Complete results for each scenario
- Summary statistics
- All raw metrics for further analysis

### CSV Output

CSV files are suitable for:
- Spreadsheet analysis
- Plotting with tools like pandas, matplotlib
- Statistical analysis

## Examples

### Quick Test (Small Dataset)

```bash
python neurondb_bm.py --vector \
  --dimensions 128 \
  --sizes 1000 \
  --iterations 10 \
  --neurondb-dsn "host=localhost dbname=neurondb user=postgres"
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
  --output-file full_benchmark \
  --neurondb-dsn "host=localhost dbname=neurondb user=postgres" \
  --pgvector-dsn "host=localhost dbname=pgvector user=postgres"
```

### Custom Index Parameters

```bash
python neurondb_bm.py --vector \
  --index-m 32 \
  --index-ef-construction 400 \
  --neurondb-dsn "host=localhost dbname=neurondb user=postgres"
```

## License

See main project LICENSE file.

