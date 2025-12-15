#!/bin/bash
# NeuronDB vs pgvector Benchmark Runner
#
# Convenient script to run vector benchmarks comparing NeuronDB and pgvector.
# Supports environment variable configuration and pre-flight checks.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default configuration
NEURONDB_DSN="${NEURONDB_DSN:-host=localhost dbname=neurondb user=postgres}"
PGVECTOR_DSN="${PGVECTOR_DSN:-host=localhost dbname=pgvector user=postgres}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
ITERATIONS="${ITERATIONS:-100}"
DIMENSIONS="${DIMENSIONS:-128,384,768}"
SIZES="${SIZES:-1000,10000,100000}"

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if PostgreSQL extension is available
check_extension() {
    local dsn="$1"
    local ext_name="$2"
    
    print_info "Checking for $ext_name extension..."
    
    if psql "$dsn" -c "SELECT 1 FROM pg_extension WHERE extname = '$ext_name';" -t 2>/dev/null | grep -q 1; then
        print_info "$ext_name extension is installed"
        return 0
    else
        print_warn "$ext_name extension not found. Attempting to create..."
        if psql "$dsn" -c "CREATE EXTENSION IF NOT EXISTS $ext_name;" 2>/dev/null; then
            print_info "$ext_name extension created successfully"
            return 0
        else
            print_error "Failed to create $ext_name extension"
            return 1
        fi
    fi
}

# Function to test database connection
test_connection() {
    local dsn="$1"
    local name="$2"
    
    print_info "Testing connection to $name..."
    if psql "$dsn" -c "SELECT 1;" >/dev/null 2>&1; then
        print_info "Connection to $name successful"
        return 0
    else
        print_error "Failed to connect to $name"
        return 1
    fi
}

# Pre-flight checks
print_info "Running pre-flight checks..."

# Check Python dependencies
print_info "Checking Python dependencies..."
if ! python3 -c "import psycopg2, numpy, tabulate" 2>/dev/null; then
    print_error "Missing Python dependencies. Please install:"
    echo "  pip install -r requirements.txt"
    exit 1
fi
print_info "Python dependencies OK"

# Test NeuronDB connection
if ! test_connection "$NEURONDB_DSN" "NeuronDB"; then
    print_error "Cannot connect to NeuronDB database"
    exit 1
fi

# Check NeuronDB extension
if ! check_extension "$NEURONDB_DSN" "neurondb"; then
    print_error "NeuronDB extension not available"
    exit 1
fi

# Test pgvector connection (optional)
PGVECTOR_AVAILABLE=false
if test_connection "$PGVECTOR_DSN" "pgvector"; then
    if check_extension "$PGVECTOR_DSN" "vector"; then
        PGVECTOR_AVAILABLE=true
        print_info "pgvector is available for comparison"
    else
        print_warn "pgvector extension not available, will run NeuronDB-only benchmark"
    fi
else
    print_warn "pgvector database not available, will run NeuronDB-only benchmark"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
print_info "Output directory: $OUTPUT_DIR"

# Build benchmark command
BENCHMARK_CMD="python3 neurondb_bm.py --vector"
BENCHMARK_CMD="$BENCHMARK_CMD --neurondb-dsn \"$NEURONDB_DSN\""
BENCHMARK_CMD="$BENCHMARK_CMD --dimensions $DIMENSIONS"
BENCHMARK_CMD="$BENCHMARK_CMD --sizes $SIZES"
BENCHMARK_CMD="$BENCHMARK_CMD --iterations $ITERATIONS"
BENCHMARK_CMD="$BENCHMARK_CMD --output all"
BENCHMARK_CMD="$BENCHMARK_CMD --output-file $OUTPUT_DIR/benchmark_results"

if [ "$PGVECTOR_AVAILABLE" = true ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --pgvector-dsn \"$PGVECTOR_DSN\""
fi

# Run benchmark
print_info "Starting benchmark..."
print_info "Command: $BENCHMARK_CMD"
echo ""

eval $BENCHMARK_CMD

if [ $? -eq 0 ]; then
    print_info "Benchmark completed successfully!"
    print_info "Results saved to: $OUTPUT_DIR/"
    ls -lh "$OUTPUT_DIR"/*.json "$OUTPUT_DIR"/*.csv 2>/dev/null || true
else
    print_error "Benchmark failed"
    exit 1
fi


