"""
Vector search benchmarks.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from psycopg2 import sql

import sys
from pathlib import Path

# Handle both package and script imports
try:
    from .base import Benchmark
    from ..utils.database import DatabaseManager
    from ..utils.data_generator import DataGenerator
    from ..utils.metrics import MetricsCollector
    from ..config import BenchmarkConfig
except ImportError:
    # Running as script, use absolute imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from modules.base import Benchmark
    from utils.database import DatabaseManager
    from utils.data_generator import DataGenerator
    from utils.metrics import MetricsCollector
    from config import BenchmarkConfig


class VectorBenchmark(Benchmark):
    """
    Benchmark vector search operations.
    
    Compares NeuronDB vs pgvector with identical queries and data.
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        config: BenchmarkConfig,
        system_name: str = 'neurondb'
    ):
        """
        Initialize vector benchmark.
        
        Args:
            db_manager: Database manager instance
            config: Benchmark configuration
            system_name: System name ('neurondb' or 'pgvector')
        """
        super().__init__(db_manager, config)
        self.system_name = system_name
        self.table_name = f"benchmark_vectors_{system_name}"
        self.index_name = f"{self.table_name}_idx"
        self.vector_column = "embedding"
        
        # Operator mapping for distance metrics
        self.operators = {
            'l2': '<->',
            'cosine': '<=>',
            'inner_product': '<#>',
        }
        
        # Order direction for each metric
        self.order_directions = {
            'l2': 'ASC',
            'cosine': 'ASC',
            'inner_product': 'DESC',  # Higher is better for inner product
        }
        
        # Operator class mapping for index creation
        self.opclass_mapping = {
            'neurondb': {
                'l2': 'vector_l2_ops',
                'cosine': 'vector_cosine_ops',
                'inner_product': 'vector_ip_ops'
            },
            'pgvector': {
                'l2': 'vector_l2_ops',
                'cosine': 'vector_cosine_ops',
                'inner_product': 'vector_ip_ops'
            }
        }
    
    def setup(self) -> None:
        """Set up benchmark tables and indexes."""
        # Ensure extension is installed
        if self.system_name == 'neurondb':
            self.db.ensure_extension('neurondb')
        elif self.system_name == 'pgvector':
            self.db.ensure_extension('vector')
        
        # Drop existing table if it exists
        self.db.drop_table_if_exists(self.table_name)
        
        # Create table
        conn = self.db._get_connection()
        try:
            with conn.cursor() as cur:
                # Use vector type for both systems (pgvector compatible)
                cur.execute(
                    sql.SQL("""
                        CREATE TABLE {} (
                            id SERIAL PRIMARY KEY,
                            {} vector,
                            metadata JSONB
                        )
                    """).format(
                        sql.Identifier(self.table_name),
                        sql.Identifier(self.vector_column)
                    )
                )
            conn.commit()
        finally:
            self.db._return_connection(conn)
    
    def insert_vectors(
        self,
        vectors: np.ndarray,
        batch_size: int = 1000
    ) -> float:
        """
        Insert vectors into the table.
        
        Args:
            vectors: Array of vectors to insert
            batch_size: Batch size for insertion
        
        Returns:
            Total insertion time in seconds
        """
        vector_strings = DataGenerator.vectors_to_sql_format(vectors)
        start_time = time.perf_counter()
        
        conn = self.db._get_connection()
        try:
            with conn.cursor() as cur:
                # Insert in batches using executemany for efficiency
                # Use string formatting for table/column names (safe since we control them)
                query = f"""
                    INSERT INTO {self.table_name} ({self.vector_column})
                    VALUES (%s::vector)
                """
                
                for i in range(0, len(vector_strings), batch_size):
                    batch = vector_strings[i:i + batch_size]
                    # Convert to list of tuples for executemany
                    params = [(vec,) for vec in batch]
                    cur.executemany(query, params)
            conn.commit()
        finally:
            self.db._return_connection(conn)
        
        end_time = time.perf_counter()
        return end_time - start_time
    
    def create_index(
        self,
        dimensions: int,
        metric: str = 'l2',
        index_type: str = 'hnsw',
        m: int = 16,
        ef_construction: int = 200
    ) -> Tuple[float, int]:
        """
        Create vector index.
        
        Args:
            dimensions: Vector dimensions
            metric: Distance metric ('l2', 'cosine', 'inner_product')
            index_type: Index type ('hnsw')
            m: HNSW parameter M
            ef_construction: HNSW parameter ef_construction
        
        Returns:
            Tuple of (build_time_seconds, index_size_bytes)
        """
        # Drop existing index
        self.db.drop_index_if_exists(self.index_name)
        
        start_time = time.perf_counter()
        
        conn = self.db._get_connection()
        original_isolation = conn.isolation_level
        try:
            # CREATE INDEX must run in autocommit mode
            from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cur:
                if index_type == 'hnsw':
                    # Determine operator class based on system and metric
                    if metric not in self.opclass_mapping[self.system_name]:
                        raise ValueError(f"Unknown metric: {metric}")
                    
                    opclass = self.opclass_mapping[self.system_name][metric]
                    
                    # Build column specification with operator class
                    # Format: column_name operator_class
                    column_spec = sql.SQL("{} {}").format(
                        sql.Identifier(self.vector_column),
                        sql.SQL(opclass)
                    )
                    
                    index_query = sql.SQL("""
                        CREATE INDEX {} ON {} USING hnsw ({})
                        WITH (m = {}, ef_construction = {})
                    """).format(
                        sql.Identifier(self.index_name),
                        sql.Identifier(self.table_name),
                        column_spec,
                        sql.Literal(m),
                        sql.Literal(ef_construction)
                    )
                    
                    cur.execute(index_query)
        finally:
            # Reset isolation level before returning connection to pool
            try:
                conn.set_isolation_level(original_isolation)
            except:
                pass
            self.db._return_connection(conn)
        
        end_time = time.perf_counter()
        build_time = end_time - start_time
        
        # Get index size
        index_size = self.db.get_index_size(self.index_name)
        
        return build_time, index_size
    
    def run_knn_query(
        self,
        query_vector: np.ndarray,
        k: int,
        metric: str
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Run a KNN query.
        
        Args:
            query_vector: Query vector
            k: Number of neighbors
            metric: Distance metric ('l2', 'cosine', 'inner_product')
        
        Returns:
            Tuple of (results, execution_time_seconds)
        """
        vector_str = DataGenerator.vectors_to_sql_format([query_vector])[0]
        operator = self.operators[metric]
        order_dir = self.order_directions[metric]
        
        query = f"""
            SELECT id, {self.vector_column} {operator} %s::vector AS distance
            FROM {self.table_name}
            ORDER BY distance {order_dir}
            LIMIT %s
        """
        
        results, exec_time = self.db.execute_query(
            query,
            params=(vector_str, k),
            timing=True
        )
        
        return results or [], exec_time
    
    def run_benchmark_scenario(
        self,
        dimensions: int,
        dataset_size: int,
        metric: str,
        k: int
    ) -> Dict[str, Any]:
        """
        Run a single benchmark scenario.
        
        Args:
            dimensions: Vector dimensions
            dataset_size: Number of vectors in dataset
            metric: Distance metric
            k: Number of neighbors
        
        Returns:
            Dictionary with benchmark results
        """
        print(f"  Running: dim={dimensions}, size={dataset_size}, metric={metric}, k={k}")
        
        # Generate data
        print(f"    Generating {dataset_size} vectors...")
        vectors = DataGenerator.generate_vectors(
            count=dataset_size,
            dimensions=dimensions,
            distribution=self.config.distribution,
            seed=self.config.seed,
            normalize=True
        )
        
        # Insert vectors
        print(f"    Inserting vectors...")
        insert_time = self.insert_vectors(vectors)
        
        # Get table size
        table_size = self.db.get_table_size(self.table_name)
        
        # Create index if requested
        index_build_time = 0.0
        index_size = 0
        if self.config.use_index:
            print(f"    Creating index...")
            try:
                # Small delay to ensure data is flushed
                import time
                time.sleep(0.1)
                
                # Try with smaller parameters if first attempt fails
                try:
                    index_build_time, index_size = self.create_index(
                        dimensions=dimensions,
                        metric=metric,
                        index_type='hnsw',
                        m=16,
                        ef_construction=200
                    )
                except Exception:
                    # Retry with smaller parameters
                    print(f"    Retrying index creation with smaller parameters...")
                    time.sleep(0.2)
                    index_build_time, index_size = self.create_index(
                        dimensions=dimensions,
                        metric=metric,
                        index_type='hnsw',
                        m=8,
                        ef_construction=100
                    )
                
                if index_size > 0:
                    print(f"    Index created successfully (size: {index_size / 1024 / 1024:.2f} MB)")
            except Exception as e:
                print(f"    WARNING: Index creation failed: {e}")
                print(f"    Continuing benchmark without index...")
                index_build_time = 0.0
                index_size = 0
        
        # Generate query vectors
        query_vectors = DataGenerator.generate_query_vectors(
            count=self.config.iterations + self.config.warmup_iterations,
            dimensions=dimensions,
            method='random',
            seed=self.config.seed,
            normalize=True
        )
        
        # Warmup
        print(f"    Warmup ({self.config.warmup_iterations} iterations)...")
        for i in range(self.config.warmup_iterations):
            self.run_knn_query(query_vectors[i], k, metric)
        
        # Actual benchmark
        print(f"    Benchmarking ({self.config.iterations} iterations)...")
        metrics = MetricsCollector()
        
        for i in range(
            self.config.warmup_iterations,
            self.config.warmup_iterations + self.config.iterations
        ):
            _, exec_time = self.run_knn_query(query_vectors[i], k, metric)
            metrics.add_timing(exec_time)
        
        # Compute ground truth for first query (for accuracy)
        ground_truth_indices, _ = DataGenerator.compute_ground_truth(
            query_vectors[self.config.warmup_iterations],
            vectors,
            k,
            metric
        )
        
        # Get results from first query for recall calculation
        first_results, _ = self.run_knn_query(
            query_vectors[self.config.warmup_iterations],
            k,
            metric
        )
        predicted_indices = np.array([r['id'] for r in first_results])
        recall = DataGenerator.compute_recall(predicted_indices, ground_truth_indices)
        
        # Collect metrics
        result = self.collect_metrics(metrics, {
            'system': self.system_name,
            'dimensions': dimensions,
            'dataset_size': dataset_size,
            'metric': metric,
            'k': k,
            'insert_time_seconds': insert_time,
            'index_build_time_seconds': index_build_time,
            'table_size_bytes': table_size,
            'index_size_bytes': index_size,
            'recall': recall,
        })
        
        return result
    
    def run(self) -> List[Dict[str, Any]]:
        """
        Run all vector benchmarks.
        
        Returns:
            List of benchmark result dictionaries
        """
        print(f"\n{'=' * 80}")
        print(f"Running Vector Benchmarks: {self.system_name.upper()}")
        print(f"{'=' * 80}\n")
        
        results = []
        
        for dimensions in self.config.dimensions:
            for dataset_size in self.config.dataset_sizes:
                for metric in self.config.distance_metrics:
                    for k in self.config.k_values:
                        try:
                            # Setup for this scenario
                            self.setup()
                            
                            # Run benchmark
                            result = self.run_benchmark_scenario(
                                dimensions, dataset_size, metric, k
                            )
                            results.append(result)
                            
                            # Teardown
                            self.teardown()
                            
                        except Exception as e:
                            print(f"    ERROR: {e}")
                            continue
        
        self.results = results
        return results
    
    def teardown(self) -> None:
        """Clean up benchmark tables and indexes."""
        self.db.drop_table_if_exists(self.table_name)
        self.db.drop_index_if_exists(self.index_name)

