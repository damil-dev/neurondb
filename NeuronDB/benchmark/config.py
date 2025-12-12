"""
Configuration management for benchmarks.
"""

import argparse
import os
from typing import List, Optional


class BenchmarkConfig:
    """
    Configuration for benchmark execution.
    """
    
    def __init__(self):
        """Initialize with defaults."""
        # Database connections
        self.neurondb_dsn: Optional[str] = None
        self.pgvector_dsn: Optional[str] = None
        
        # Benchmark parameters
        self.dimensions: List[int] = [128, 384, 768, 1536]
        self.dataset_sizes: List[int] = [1000, 10000, 100000]
        self.distance_metrics: List[str] = ['l2', 'cosine', 'inner_product']
        self.k_values: List[int] = [1, 10, 100]
        self.use_index: bool = True
        self.iterations: int = 100
        self.warmup_iterations: int = 10
        
        # Output
        self.output_formats: List[str] = ['console']
        self.output_file: Optional[str] = None
        
        # Data generation
        self.seed: Optional[int] = 42
        self.distribution: str = 'normal'
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'BenchmarkConfig':
        """
        Create config from command-line arguments.
        
        Args:
            args: Parsed command-line arguments
        
        Returns:
            BenchmarkConfig instance
        """
        config = cls()
        
        # Database connections
        config.neurondb_dsn = args.neurondb_dsn or os.getenv('NEURONDB_DSN')
        config.pgvector_dsn = args.pgvector_dsn or os.getenv('PGVECTOR_DSN')
        
        # Parse comma-separated lists
        if args.dimensions:
            config.dimensions = [int(d) for d in args.dimensions.split(',')]
        if args.sizes:
            config.dataset_sizes = [int(s) for s in args.sizes.split(',')]
        if args.metrics:
            config.distance_metrics = args.metrics.split(',')
        if args.k_values:
            config.k_values = [int(k) for k in args.k_values.split(',')]
        
        # Boolean flags
        config.use_index = args.index if hasattr(args, 'index') else True
        
        # Numeric parameters
        if hasattr(args, 'iterations'):
            config.iterations = args.iterations
        if hasattr(args, 'warmup'):
            config.warmup_iterations = args.warmup
        
        # Output
        if args.output:
            config.output_formats = args.output.split(',')
        if args.output_file:
            config.output_file = args.output_file
        
        # Data generation
        if hasattr(args, 'seed') and args.seed:
            config.seed = args.seed
        
        return config
    
    def validate(self) -> None:
        """
        Validate configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.neurondb_dsn:
            raise ValueError("NeuronDB connection string is required")
        
        if len(self.dimensions) == 0:
            raise ValueError("At least one dimension must be specified")
        
        if len(self.dataset_sizes) == 0:
            raise ValueError("At least one dataset size must be specified")
        
        if len(self.distance_metrics) == 0:
            raise ValueError("At least one distance metric must be specified")
        
        if self.iterations < 1:
            raise ValueError("Iterations must be at least 1")
        
        valid_metrics = {'l2', 'cosine', 'inner_product'}
        for metric in self.distance_metrics:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid distance metric: {metric}. Must be one of {valid_metrics}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='NeuronDB Benchmark Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Vector benchmark with defaults
  python neurondb_bm.py --vector
  
  # Custom vector benchmark
  python neurondb_bm.py --vector \\
    --dimensions 128,384,768 \\
    --sizes 1000,10000 \\
    --metrics l2,cosine \\
    --iterations 200 \\
    --output console,json
  
  # Compare with pgvector
  python neurondb_bm.py --vector \\
    --neurondb-dsn "host=localhost dbname=neurondb" \\
    --pgvector-dsn "host=localhost dbname=pgvector"
        """
    )
    
    # Benchmark type selection
    parser.add_argument(
        '--vector',
        action='store_true',
        help='Run vector benchmarks'
    )
    parser.add_argument(
        '--embeddings',
        action='store_true',
        help='Run embedding benchmarks (not yet implemented)'
    )
    parser.add_argument(
        '--ml',
        action='store_true',
        help='Run ML benchmarks (not yet implemented)'
    )
    
    # Database connections
    parser.add_argument(
        '--neurondb-dsn',
        type=str,
        help='NeuronDB connection string (or set NEURONDB_DSN env var)'
    )
    parser.add_argument(
        '--pgvector-dsn',
        type=str,
        help='pgvector connection string for comparison (or set PGVECTOR_DSN env var)'
    )
    
    # Benchmark parameters
    parser.add_argument(
        '--dimensions',
        type=str,
        help='Comma-separated vector dimensions (default: 128,384,768,1536)'
    )
    parser.add_argument(
        '--sizes',
        type=str,
        help='Comma-separated dataset sizes (default: 1000,10000,100000)'
    )
    parser.add_argument(
        '--metrics',
        type=str,
        help='Comma-separated distance metrics: l2,cosine,inner_product (default: all)'
    )
    parser.add_argument(
        '--k-values',
        type=str,
        help='Comma-separated K values for KNN search (default: 1,10,100)'
    )
    parser.add_argument(
        '--index',
        action='store_true',
        default=True,
        help='Test with indexes (default: True)'
    )
    parser.add_argument(
        '--no-index',
        dest='index',
        action='store_false',
        help='Test without indexes'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of query iterations per test (default: 100)'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=10,
        help='Number of warmup iterations (default: 10)'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        default='console',
        help='Output formats: console,json,csv,all (default: console)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='Output file path (for JSON/CSV)'
    )
    
    # Data generation
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility (default: 42)'
    )
    
    return parser.parse_args()

