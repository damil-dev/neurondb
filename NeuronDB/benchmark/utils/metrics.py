"""
Performance metrics collection and statistics.
"""

import statistics
from typing import List, Dict, Any, Optional
import time


class MetricsCollector:
    """
    Collect and compute performance metrics.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.timings: List[float] = []
        self.reset()
    
    def reset(self) -> None:
        """Reset all collected metrics."""
        self.timings = []
    
    def add_timing(self, duration: float) -> None:
        """
        Add a timing measurement.
        
        Args:
            duration: Duration in seconds
        """
        self.timings.append(duration)
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Compute statistics from collected timings.
        
        Returns:
            Dictionary with min, max, mean, median, p50, p95, p99
        """
        if not self.timings:
            return {
                'count': 0,
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'median': 0.0,
                'p50': 0.0,
                'p95': 0.0,
                'p99': 0.0,
            }
        
        sorted_timings = sorted(self.timings)
        n = len(sorted_timings)
        
        def percentile(p: float) -> float:
            """Compute percentile."""
            idx = int(p * (n - 1))
            return sorted_timings[idx]
        
        return {
            'count': n,
            'min': min(sorted_timings),
            'max': max(sorted_timings),
            'mean': statistics.mean(sorted_timings),
            'median': statistics.median(sorted_timings),
            'p50': percentile(0.50),
            'p95': percentile(0.95),
            'p99': percentile(0.99),
        }
    
    def get_throughput(self, total_queries: Optional[int] = None) -> float:
        """
        Compute queries per second (QPS).
        
        Args:
            total_queries: Total number of queries (default: len(timings))
        
        Returns:
            Queries per second
        """
        if not self.timings:
            return 0.0
        
        n = total_queries if total_queries is not None else len(self.timings)
        total_time = sum(self.timings)
        
        if total_time == 0:
            return 0.0
        
        return n / total_time
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        Format duration in human-readable format.
        
        Args:
            seconds: Duration in seconds
        
        Returns:
            Formatted string (e.g., "1.23ms", "45.6s")
        """
        if seconds < 0.001:
            return f"{seconds * 1_000_000:.2f}μs"
        elif seconds < 1.0:
            return f"{seconds * 1000:.2f}ms"
        else:
            return f"{seconds:.2f}s"
    
    @staticmethod
    def format_bytes(bytes_count: int) -> str:
        """
        Format bytes in human-readable format.
        
        Args:
            bytes_count: Size in bytes
        
        Returns:
            Formatted string (e.g., "1.5MB", "2.3GB")
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_count < 1024.0:
                return f"{bytes_count:.2f}{unit}"
            bytes_count /= 1024.0
        return f"{bytes_count:.2f}PB"





