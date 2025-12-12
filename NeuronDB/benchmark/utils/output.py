"""
Output formatting for benchmark results.
"""

import json
import csv
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
                    speedup = f"{pgv / ndb:.2f}x"
            except (ValueError, TypeError):
                pass
            
            neurondb_str = f"{neurondb_val:>20}" if neurondb_val != 'N/A' else f"{'N/A':>20}"
            pgvector_str = f"{pgvector_val:>20}" if pgvector_val != 'N/A' else f"{'N/A':>20}"
            
            lines.append(f"{metric:<30} | {neurondb_str} | {pgvector_str} | {speedup:>10}")
        
        return "\n".join(lines)
    
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
        
        # Group by test configuration
        for i, result in enumerate(results, 1):
            lines.append(f"\nTest {i}:")
            for key, value in result.items():
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)

