"""
Output formatting for benchmark results.
"""

import json
import csv
import math
from typing import List, Dict, Any, Optional
from pathlib import Path


class OutputFormatter:
    """
    Format and export benchmark results.
    """
    
    @staticmethod
    def format_console_table(
        results: List[Dict[str, Any]],
        title: Optional[str] = None
    ) -> str:
        """
        Format results as a console table.
        
        Args:
            results: List of result dictionaries
            title: Optional table title
        
        Returns:
            Formatted table string
        """
        if not results:
            return "No results to display."
        
        # Get all unique keys from all results
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        
        # Order keys consistently
        ordered_keys = sorted(all_keys)
        
        # Build table
        lines = []
        if title:
            lines.append(f"\n{'=' * 80}")
            lines.append(f" {title}")
            lines.append(f"{'=' * 80}")
        
        # Header
        header = " | ".join(f"{k:>15}" for k in ordered_keys)
        lines.append(header)
        lines.append("-" * len(header))
        
        # Rows
        for result in results:
            row = " | ".join(
                f"{str(result.get(k, 'N/A')):>15}" for k in ordered_keys
            )
            lines.append(row)
        
        return "\n".join(lines)
    
    @staticmethod
    def format_comparison_table(
        neurondb_results: Dict[str, Any],
        pgvector_results: Optional[Dict[str, Any]],
        metrics: List[str]
    ) -> str:
        """
        Format side-by-side comparison table.
        
        Args:
            neurondb_results: NeuronDB benchmark results
            pgvector_results: pgvector benchmark results (optional)
            metrics: List of metric names to compare
        
        Returns:
            Formatted comparison table
        """
        lines = []
        lines.append(f"\n{'=' * 80}")
        lines.append(" Comparison: NeuronDB vs pgvector")
        lines.append(f"{'=' * 80}")
        lines.append(f"{'Metric':<30} | {'NeuronDB':>20} | {'pgvector':>20} | {'Speedup':>10}")
        lines.append("-" * 80)
        
        for metric in metrics:
            neurondb_val = neurondb_results.get(metric, 'N/A')
            pgvector_val = pgvector_results.get(metric, 'N/A') if pgvector_results else 'N/A'
            
            # Calculate speedup if both values are numeric
            speedup = 'N/A'
            try:
                ndb = float(neurondb_val)
                pgv = float(pgvector_val)
                if pgv > 0:
                    # For latency metrics, lower is better, so speedup = pgv/ndb
                    # For throughput, higher is better, so speedup = ndb/pgv
                    if 'latency' in metric.lower() or 'time' in metric.lower():
                        speedup = f"{pgv / ndb:.2f}x" if ndb > 0 else 'N/A'
                    elif 'throughput' in metric.lower() or 'qps' in metric.lower():
                        speedup = f"{ndb / pgv:.2f}x" if pgv > 0 else 'N/A'
                    else:
                        speedup = f"{pgv / ndb:.2f}x" if ndb > 0 else 'N/A'
            except (ValueError, TypeError):
                pass
            
            # Format values nicely
            neurondb_str = OutputFormatter._format_value(neurondb_val)
            pgvector_str = OutputFormatter._format_value(pgvector_val)
            
            lines.append(f"{metric:<30} | {neurondb_str:>20} | {pgvector_str:>20} | {speedup:>10}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_value(value: Any) -> str:
        """Format a value for display in tables."""
        if value == 'N/A':
            return 'N/A'
        try:
            fval = float(value)
            # Format based on magnitude
            if abs(fval) < 0.001:
                return f"{fval:.2e}"
            elif abs(fval) < 1:
                return f"{fval:.4f}"
            elif abs(fval) < 1000:
                return f"{fval:.2f}"
            elif abs(fval) < 1000000:
                return f"{fval/1000:.2f}K"
            else:
                return f"{fval/1000000:.2f}M"
        except (ValueError, TypeError):
            return str(value)
    
    @staticmethod
    def compute_summary_statistics(
        neurondb_results: List[Dict[str, Any]],
        pgvector_results: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Compute summary statistics across all benchmark results.
        
        Args:
            neurondb_results: List of NeuronDB benchmark results
            pgvector_results: Optional list of pgvector benchmark results
        
        Returns:
            Dictionary with summary statistics
        """
        stats = {}
        
        # Metrics to compute geomean for
        latency_metrics = ['latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms', 'latency_mean_ms']
        throughput_metrics = ['throughput_qps']
        
        def geomean(values: List[float]) -> float:
            """Compute geometric mean."""
            if not values or any(v <= 0 for v in values):
                return 0.0
            return math.exp(sum(math.log(v) for v in values) / len(values))
        
        # Compute geomean for NeuronDB
        for metric in latency_metrics + throughput_metrics:
            ndb_values = [float(r.get(metric, 0)) for r in neurondb_results if r.get(metric, 0) > 0]
            if ndb_values:
                stats[f'neurondb_{metric}_geomean'] = geomean(ndb_values)
        
        # Compute geomean for pgvector if available
        if pgvector_results:
            for metric in latency_metrics + throughput_metrics:
                pgv_values = [float(r.get(metric, 0)) for r in pgvector_results if r.get(metric, 0) > 0]
                if pgv_values:
                    stats[f'pgvector_{metric}_geomean'] = geomean(pgv_values)
                    
                    # Compute speedup
                    ndb_geomean = stats.get(f'neurondb_{metric}_geomean', 0)
                    if ndb_geomean > 0:
                        if 'latency' in metric or 'time' in metric:
                            stats[f'{metric}_speedup'] = stats[f'pgvector_{metric}_geomean'] / ndb_geomean
                        else:
                            stats[f'{metric}_speedup'] = ndb_geomean / stats[f'pgvector_{metric}_geomean']
        
        return stats
    
    @staticmethod
    def export_json(
        results: List[Dict[str, Any]],
        filepath: str
    ) -> None:
        """
        Export results to JSON file.
        
        Args:
            results: List of result dictionaries
            filepath: Output file path
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    @staticmethod
    def export_csv(
        results: List[Dict[str, Any]],
        filepath: str
    ) -> None:
        """
        Export results to CSV file.
        
        Args:
            results: List of result dictionaries
            filepath: Output file path
        """
        if not results:
            return
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get all keys
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        ordered_keys = sorted(all_keys)
        
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_keys)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
    
    @staticmethod
    def format_summary(
        results: List[Dict[str, Any]],
        system_name: str
    ) -> str:
        """
        Format a summary of benchmark results.
        
        Args:
            results: List of result dictionaries
            system_name: Name of the system being benchmarked
        
        Returns:
            Formatted summary string
        """
        lines = []
        lines.append(f"\n{'=' * 80}")
        lines.append(f" Benchmark Summary: {system_name}")
        lines.append(f"{'=' * 80}")
        
        if not results:
            lines.append("No results available.")
            return "\n".join(lines)
        
        # Group by test configuration and show key metrics
        for i, result in enumerate(results, 1):
            lines.append(f"\nTest {i}:")
            lines.append(f"  Configuration: dim={result.get('dimensions')}, "
                        f"size={result.get('dataset_size')}, "
                        f"metric={result.get('metric')}, k={result.get('k')}")
            
            # Format latency metrics
            p50 = result.get('latency_p50_ms', 'N/A')
            p95 = result.get('latency_p95_ms', 'N/A')
            p50_str = f"{p50:.2f}" if isinstance(p50, (int, float)) else str(p50)
            p95_str = f"{p95:.2f}" if isinstance(p95, (int, float)) else str(p95)
            lines.append(f"  Latency (p50): {p50_str} ms")
            lines.append(f"  Latency (p95): {p95_str} ms")
            
            # Format throughput
            qps = result.get('throughput_qps', 'N/A')
            qps_str = f"{qps:.2f}" if isinstance(qps, (int, float)) else str(qps)
            lines.append(f"  Throughput: {qps_str} QPS")
            
            # Format recall
            recall = result.get('recall', 'N/A')
            recall_str = f"{recall:.4f}" if isinstance(recall, (int, float)) else str(recall)
            lines.append(f"  Recall: {recall_str}")
            
            # Format index size
            idx_size = result.get('index_size_bytes', 0)
            if isinstance(idx_size, (int, float)) and idx_size > 0:
                lines.append(f"  Index Size: {idx_size / 1024 / 1024:.2f} MB")
        
        return "\n".join(lines)

