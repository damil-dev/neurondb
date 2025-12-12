"""
Base benchmark class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# Handle both package and script imports
try:
    from ..utils.database import DatabaseManager
    from ..utils.metrics import MetricsCollector
except ImportError:
    # Running as script, use absolute imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.database import DatabaseManager
    from utils.metrics import MetricsCollector


class Benchmark(ABC):
    """
    Abstract base class for all benchmarks.
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        config: 'BenchmarkConfig'
    ):
        """
        Initialize benchmark.
        
        Args:
            db_manager: Database manager instance
            config: Benchmark configuration
        """
        self.db = db_manager
        self.config = config
        self.results: List[Dict[str, Any]] = []
    
    @abstractmethod
    def setup(self) -> None:
        """
        Set up benchmark environment (create tables, indexes, etc.).
        
        This method should be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def run(self) -> List[Dict[str, Any]]:
        """
        Run the benchmark.
        
        Returns:
            List of result dictionaries
        
        This method should be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def teardown(self) -> None:
        """
        Clean up benchmark environment (drop tables, indexes, etc.).
        
        This method should be implemented by subclasses.
        """
        pass
    
    def collect_metrics(
        self,
        metrics_collector: MetricsCollector,
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Collect metrics from a metrics collector.
        
        Args:
            metrics_collector: MetricsCollector instance
            additional_metrics: Additional metrics to include
        
        Returns:
            Dictionary of collected metrics
        """
        stats = metrics_collector.get_statistics()
        result = {
            'latency_p50_ms': stats['p50'] * 1000,
            'latency_p95_ms': stats['p95'] * 1000,
            'latency_p99_ms': stats['p99'] * 1000,
            'latency_mean_ms': stats['mean'] * 1000,
            'latency_min_ms': stats['min'] * 1000,
            'latency_max_ms': stats['max'] * 1000,
            'throughput_qps': metrics_collector.get_throughput(),
            'query_count': stats['count'],
        }
        
        if additional_metrics:
            result.update(additional_metrics)
        
        return result

