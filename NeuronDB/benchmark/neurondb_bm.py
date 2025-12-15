#!/usr/bin/env python3
"""
NeuronDB Benchmark Suite

A modular and extensible benchmark tool for NeuronDB, supporting:
- Vector search benchmarks (comparing NeuronDB vs pgvector)
- Embedding benchmarks (future)
- ML benchmarks (future)

Usage:
    python neurondb_bm.py --vector [options]
    python neurondb_bm.py --embeddings [options]  # Future
    python neurondb_bm.py --ml [options]  # Future
"""

import sys
import os
from pathlib import Path

# Add benchmark directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import BenchmarkConfig, parse_args
from utils.database import DatabaseManager
from utils.output import OutputFormatter
from modules.vector import VectorBenchmark


def run_vector_benchmark(config: BenchmarkConfig) -> None:
    """
    Run vector search benchmarks.
    
    Args:
        config: Benchmark configuration
    """
    print("\n" + "=" * 80)
    print("NeuronDB Vector Benchmark Suite")
    print("=" * 80)
    
    # Create NeuronDB database manager
    neurondb_db = DatabaseManager(connection_string=config.neurondb_dsn)
    
    # Run NeuronDB benchmarks
    neurondb_benchmark = VectorBenchmark(
        db_manager=neurondb_db,
        config=config,
        system_name='neurondb'
    )
    
    neurondb_results = neurondb_benchmark.run()
    
    # Run pgvector benchmarks if DSN provided
    pgvector_results = None
    if config.pgvector_dsn:
        print("\n" + "=" * 80)
        print("Running pgvector benchmarks for comparison...")
        print("=" * 80)
        
        pgvector_db = DatabaseManager(connection_string=config.pgvector_dsn)
        pgvector_benchmark = VectorBenchmark(
            db_manager=pgvector_db,
            config=config,
            system_name='pgvector'
        )
        
        pgvector_results = pgvector_benchmark.run()
        
        pgvector_db.close()
    
    neurondb_db.close()
    
    # Output results
    formatter = OutputFormatter()
    
    # Console output
    if 'console' in config.output_formats or 'all' in config.output_formats:
        print(formatter.format_summary(neurondb_results, "NeuronDB"))
        
        if pgvector_results:
            print(formatter.format_summary(pgvector_results, "pgvector"))
            
            # Comparison table for each scenario
            for i, (ndb_result, pgv_result) in enumerate(zip(neurondb_results, pgvector_results)):
                if i < len(pgvector_results):
                    metrics = [
                        'latency_p50_ms',
                        'latency_p95_ms',
                        'latency_p99_ms',
                        'latency_mean_ms',
                        'throughput_qps',
                        'recall',
                        'index_build_time_seconds',
                        'index_size_bytes',
                    ]
                    print(formatter.format_comparison_table(
                        ndb_result,
                        pgv_result,
                        metrics
                    ))
            
            # Summary statistics
            summary_stats = formatter.compute_summary_statistics(neurondb_results, pgvector_results)
            if summary_stats:
                print("\n" + "=" * 80)
                print(" Summary Statistics (Geometric Mean)")
                print("=" * 80)
                for key, value in sorted(summary_stats.items()):
                    if 'speedup' in key:
                        print(f"  {key}: {value:.2f}x")
                    else:
                        print(f"  {key}: {value:.2f}")
    
    # JSON output
    if 'json' in config.output_formats or 'all' in config.output_formats:
        output_file = config.output_file or 'benchmark_results.json'
        if output_file.endswith('.json'):
            json_file = output_file
        else:
            json_file = f"{output_file}.json"
        
        all_results = {
            'neurondb': neurondb_results,
        }
        if pgvector_results:
            all_results['pgvector'] = pgvector_results
            # Add summary statistics
            summary_stats = formatter.compute_summary_statistics(neurondb_results, pgvector_results)
            if summary_stats:
                all_results['summary_statistics'] = summary_stats
        
        formatter.export_json(all_results, json_file)
        print(f"\nResults exported to: {json_file}")
    
    # CSV output
    if 'csv' in config.output_formats or 'all' in config.output_formats:
        output_file = config.output_file or 'benchmark_results.csv'
        if output_file.endswith('.csv'):
            csv_file = output_file
        else:
            csv_file = f"{output_file}.csv"
        
        # Combine results with system label
        all_results = []
        for result in neurondb_results:
            all_results.append(result)
        if pgvector_results:
            for result in pgvector_results:
                all_results.append(result)
        
        formatter.export_csv(all_results, csv_file)
        print(f"Results exported to: {csv_file}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Check that at least one benchmark type is specified
    if not (args.vector or args.embeddings or args.ml):
        print("Error: Must specify at least one benchmark type (--vector, --embeddings, --ml)")
        sys.exit(1)
    
    # Create configuration
    try:
        config = BenchmarkConfig.from_args(args)
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    # Run selected benchmarks
    try:
        if args.vector:
            run_vector_benchmark(config)
        elif args.embeddings:
            print("Embedding benchmarks not yet implemented.")
            sys.exit(1)
        elif args.ml:
            print("ML benchmarks not yet implemented.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nError running benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

