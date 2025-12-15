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
        metric: str,
        use_index: bool = True
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Run a KNN query.
        
        Args:
            query_vector: Query vector
            k: Number of neighbors
            metric: Distance metric ('l2', 'cosine', 'inner_product')
            use_index: Whether to use index (if False, forces sequential scan)
        
        Returns:
            Tuple of (results, execution_time_seconds)
        """
        vector_str = DataGenerator.vectors_to_sql_format([query_vector])[0]
        operator = self.operators[metric]
        order_dir = self.order_directions[metric]
        
        # Build query
        query = f"""
            SELECT id, {self.vector_column} {operator} %s::vector AS distance
            FROM {self.table_name}
            ORDER BY distance {order_dir}
            LIMIT %s
        """
        
        # Note: use_index parameter is mainly for documentation
        # Sequential scan comparison would require dropping the index
        # For now, we rely on the index being present or not based on config.use_index
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
                
                # Try with configured parameters first
                try:
                    index_build_time, index_size = self.create_index(
                        dimensions=dimensions,
                        metric=metric,
                        index_type='hnsw',
                        m=self.config.index_m,
                        ef_construction=self.config.index_ef_construction
                    )
                except Exception:
                    # Retry with smaller parameters if configured ones fail
                    print(f"    Retrying index creation with smaller parameters...")
                    time.sleep(0.2)
                    index_build_time, index_size = self.create_index(
                        dimensions=dimensions,
                        metric=metric,
                        index_type='hnsw',
                        m=max(4, self.config.index_m // 2),
                        ef_construction=max(50, self.config.index_ef_construction // 2)
                    )
                
                if index_size > 0:
                    print(f"    Index created successfully (size: {index_size / 1024 / 1024:.2f} MB)")
                    # Wait after index creation to let database stabilize
                    import time
                    time.sleep(2.0)
                    
                    # Check if database is in recovery mode and wait if needed
                    try:
                        if not self.db.wait_for_recovery(max_wait=30):
                            print(f"    WARNING: Database still in recovery mode after waiting")
                    except:
                        pass
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
            try:
                self.run_knn_query(
                    query_vectors[i], 
                    k, 
                    metric,
                    use_index=self.config.use_index
                )
            except Exception as e:
                print(f"    WARNING: Warmup query {i+1} failed: {e}")
                # If database is in recovery, wait for it
                if 'recovery' in str(e).lower():
                    print(f"    Database in recovery mode, waiting...")
                    if self.db.wait_for_recovery(max_wait=60):
                        print(f"    Database recovery complete, continuing...")
                    else:
                        print(f"    WARNING: Database still in recovery after timeout")
                continue
        
        # Actual benchmark
        print(f"    Benchmarking ({self.config.iterations} iterations)...")
        metrics = MetricsCollector()
        
        successful_queries = 0
        for i in range(
            self.config.warmup_iterations,
            self.config.warmup_iterations + self.config.iterations
        ):
            try:
                _, exec_time = self.run_knn_query(
                    query_vectors[i], 
                    k, 
                    metric,
                    use_index=self.config.use_index
                )
                metrics.add_timing(exec_time)
                successful_queries += 1
            except Exception as e:
                print(f"    WARNING: Query {i+1} failed: {e}")
                # If database is in recovery, wait for it
                if 'recovery' in str(e).lower():
                    print(f"    Database in recovery mode, waiting...")
                    if self.db.wait_for_recovery(max_wait=60):
                        print(f"    Database recovery complete, continuing...")
                    else:
                        print(f"    WARNING: Database still in recovery after timeout")
                continue
        
        if successful_queries == 0:
            raise RuntimeError("All benchmark queries failed. Database may be unavailable.")
        elif successful_queries < self.config.iterations:
            print(f"    WARNING: Only {successful_queries}/{self.config.iterations} queries succeeded")
        
        # Compute ground truth for first query (for accuracy)
        # Note: IDs in database start from 1, but numpy arrays are 0-indexed
        # So we need to adjust: ground_truth_indices are 0-indexed, but DB IDs are 1-indexed
        ground_truth_indices, ground_truth_distances = DataGenerator.compute_ground_truth(
            query_vectors[self.config.warmup_iterations],
            vectors,
            k,
            metric
        )
        # Convert 0-indexed to 1-indexed for comparison with DB results
        ground_truth_ids = ground_truth_indices + 1
        
        # Get results from first query for recall calculation
        first_results, _ = self.run_knn_query(
            query_vectors[self.config.warmup_iterations],
            k,
            metric,
            use_index=self.config.use_index
        )
        
        # Validate results
        if len(first_results) == 0:
            print(f"    WARNING: Query returned no results")
            recall = 0.0
        else:
            predicted_indices = np.array([r['id'] for r in first_results])
            recall = DataGenerator.compute_recall(predicted_indices, ground_truth_ids)
            
            # Additional validation: check if distances are reasonable
            predicted_distances = np.array([r['distance'] for r in first_results])
            if len(predicted_distances) > 0 and len(ground_truth_distances) > 0:
                # Check if the first result matches ground truth (within tolerance)
                if abs(predicted_distances[0] - ground_truth_distances[0]) > 1e-5:
                    print(f"    WARNING: Distance mismatch - predicted: {predicted_distances[0]:.6f}, ground truth: {ground_truth_distances[0]:.6f}")
        
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

