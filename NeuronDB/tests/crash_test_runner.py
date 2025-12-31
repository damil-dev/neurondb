#!/usr/bin/env python3

"""
crash_test_runner.py
Master orchestrator for NeuronDB crash testing.

Runs SQL test files with core dump monitoring enabled.
Automatically detects crashes, analyzes core dumps, and generates reports.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    from crash_analyzer import CrashAnalyzer
except ImportError:
    CrashAnalyzer = None
    print("Warning: crash_analyzer.py not found, core dump analysis disabled")


# Configuration
TESTS_SQL_DIR = os.path.join(os.path.dirname(__file__), "sql")
CRASH_PREVENTION_DIR = os.path.join(TESTS_SQL_DIR, "crash_prevention")
CORE_DIR = "/tmp/core"
CORE_ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), "core_analysis")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "crash_reports")

DEFAULT_DB = "neurondb"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5432
DEFAULT_USER = "pge"


class CrashTestRunner:
    """Main crash test runner class."""
    
    def __init__(self, db: str = DEFAULT_DB, host: str = DEFAULT_HOST, 
                 port: int = DEFAULT_PORT, user: str = DEFAULT_USER,
                 verbose: bool = False):
        self.db = db
        self.host = host
        self.port = port
        self.user = user
        self.verbose = verbose
        self.setup_dirs()
        self.analyzer = CrashAnalyzer() if CrashAnalyzer else None
        
    def setup_dirs(self):
        """Create necessary directories."""
        os.makedirs(CORE_DIR, exist_ok=True)
        os.makedirs(CORE_ANALYSIS_DIR, exist_ok=True)
        os.makedirs(REPORT_DIR, exist_ok=True)
        
    def setup_core_dumps(self) -> bool:
        """Setup core dump generation."""
        script_path = os.path.join(os.path.dirname(__file__), "setup_core_dumps.sh")
        if not os.path.exists(script_path):
            print(f"Warning: {script_path} not found, setting up manually...")
            try:
                # Set ulimit -c unlimited
                subprocess.run(["ulimit", "-c", "unlimited"], shell=True, check=False)
                # Create core directory
                os.makedirs(CORE_DIR, exist_ok=True)
                return True
            except Exception as e:
                print(f"Error setting up core dumps: {e}")
                return False
        else:
            try:
                result = subprocess.run(
                    ["bash", script_path],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode == 0:
                    if self.verbose:
                        print(result.stdout)
                    return True
                else:
                    print(f"Warning: setup_core_dumps.sh failed: {result.stderr}")
                    return False
            except Exception as e:
                print(f"Error running setup_core_dumps.sh: {e}")
                return False
    
    def get_core_dumps_before(self) -> set:
        """Get list of core dump files before test."""
        pattern = os.path.join(CORE_DIR, "core.*")
        return set(glob.glob(pattern))
    
    def get_new_core_dumps(self, before: set) -> List[str]:
        """Get new core dump files created during test."""
        pattern = os.path.join(CORE_DIR, "core.*")
        after = set(glob.glob(pattern))
        new = after - before
        return sorted(list(new))
    
    def run_sql_test(self, sql_file: str) -> Tuple[bool, str, Optional[str]]:
        """
        Run a SQL test file and check for crashes.
        
        Returns:
            (success, output, core_dump_path)
        """
        if not os.path.exists(sql_file):
            return False, f"File not found: {sql_file}", None
        
        # Get core dumps before test
        cores_before = self.get_core_dumps_before()
        
        # Run psql
        cmd = [
            "psql",
            "-h", self.host,
            "-p", str(self.port),
            "-U", self.user,
            "-d", self.db,
            "-f", sql_file,
            "-v", "ON_ERROR_STOP=off"  # Don't stop on errors
        ]
        
        try:
            env = os.environ.copy()
            # Ensure ulimit -c unlimited is set
            env["SHELL"] = "/bin/bash"
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=600,  # 10 minute timeout
                check=False  # Don't raise on non-zero exit
            )
            
            # Check for new core dumps
            new_cores = self.get_new_core_dumps(cores_before)
            
            if new_cores:
                core_path = new_cores[0]  # Take first core dump
                return False, result.stdout + result.stderr, core_path
            else:
                # Test passed (no crash), even if SQL had errors
                return True, result.stdout + result.stderr, None
                
        except subprocess.TimeoutExpired:
            return False, "Test timed out after 10 minutes", None
        except Exception as e:
            return False, f"Error running test: {e}", None
    
    def analyze_crash(self, core_path: str, sql_file: str) -> Optional[Dict]:
        """Analyze a core dump and return crash information."""
        if not self.analyzer:
            return None
        
        return self.analyzer.analyze(core_path, sql_file)
    
    def run_test_category(self, category: str) -> Dict:
        """Run all tests in a category."""
        category_dir = os.path.join(CRASH_PREVENTION_DIR, category)
        if not os.path.exists(category_dir):
            # Try numbered files in crash_prevention directory
            pattern = os.path.join(CRASH_PREVENTION_DIR, f"*{category}*.sql")
            sql_files = glob.glob(pattern)
            if not sql_files:
                return {
                    "category": category,
                    "tests": [],
                    "total": 0,
                    "passed": 0,
                    "crashed": 0,
                    "errors": []
                }
        else:
            sql_files = sorted(glob.glob(os.path.join(category_dir, "*.sql")))
        
        results = {
            "category": category,
            "tests": [],
            "total": len(sql_files),
            "passed": 0,
            "crashed": 0,
            "errors": []
        }
        
        print(f"\n{'='*80}")
        print(f"Running category: {category}")
        print(f"Found {len(sql_files)} test files")
        print(f"{'='*80}\n")
        
        for sql_file in sql_files:
            test_name = os.path.basename(sql_file)
            print(f"Running: {test_name}...", end=" ", flush=True)
            
            start_time = time.time()
            success, output, core_path = self.run_sql_test(sql_file)
            elapsed = time.time() - start_time
            
            test_result = {
                "file": test_name,
                "path": sql_file,
                "success": success,
                "elapsed": elapsed,
                "core_dump": core_path
            }
            
            if not success and core_path:
                # Crash detected
                results["crashed"] += 1
                print(f"❌ CRASH (core: {os.path.basename(core_path)})")
                
                # Analyze crash
                crash_info = self.analyze_crash(core_path, sql_file)
                if crash_info:
                    test_result["crash_info"] = crash_info
            elif success:
                results["passed"] += 1
                print(f"✓ Passed ({elapsed:.2f}s)")
            else:
                print(f"⚠ Error (no crash)")
                results["errors"].append({
                    "file": test_name,
                    "error": output[:500]  # First 500 chars
                })
            
            results["tests"].append(test_result)
        
        return results
    
    def run_all(self) -> Dict:
        """Run all crash prevention tests."""
        # Ensure core dumps are enabled
        if not self.setup_core_dumps():
            print("Warning: Could not setup core dumps, continuing anyway...")
        
        # Find all test categories
        test_categories = []
        
        # Check for numbered test files (001_, 002_, etc.)
        pattern = os.path.join(CRASH_PREVENTION_DIR, "*.sql")
        numbered_files = glob.glob(pattern)
        if numbered_files:
            # Extract category names from filenames
            categories = set()
            for f in numbered_files:
                basename = os.path.basename(f)
                # Extract category from filename like "001_null_parameters.sql"
                parts = basename.replace(".sql", "").split("_", 1)
                if len(parts) > 1:
                    categories.add(parts[1])
            test_categories = sorted(categories)
        else:
            # Try subdirectories
            if os.path.exists(CRASH_PREVENTION_DIR):
                test_categories = [
                    d for d in os.listdir(CRASH_PREVENTION_DIR)
                    if os.path.isdir(os.path.join(CRASH_PREVENTION_DIR, d))
                ]
        
        if not test_categories:
            # Fallback: use predefined categories from plan
            test_categories = [
                "null_parameters",
                "invalid_models",
                "spi_failures",
                "memory_stress",
                "array_bounds",
                "overflow",
                "gpu_failures",
                "index_crashes",
                "type_confusion",
                "concurrency",
                "algorithm_boundaries"
            ]
        
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "categories": [],
            "summary": {
                "total_tests": 0,
                "total_passed": 0,
                "total_crashed": 0,
                "total_errors": 0
            }
        }
        
        for category in test_categories:
            category_results = self.run_test_category(category)
            all_results["categories"].append(category_results)
            all_results["summary"]["total_tests"] += category_results["total"]
            all_results["summary"]["total_passed"] += category_results["passed"]
            all_results["summary"]["total_crashed"] += category_results["crashed"]
            all_results["summary"]["total_errors"] += len(category_results["errors"])
        
        return all_results
    
    def generate_report(self, results: Dict) -> str:
        """Generate a human-readable report."""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("NEURONDB CRASH TEST REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Timestamp: {results['timestamp']}")
        report_lines.append("")
        
        summary = results["summary"]
        report_lines.append("SUMMARY")
        report_lines.append("-"*80)
        report_lines.append(f"Total Tests:  {summary['total_tests']}")
        report_lines.append(f"Passed:       {summary['total_passed']} ✓")
        report_lines.append(f"CRASHED:      {summary['total_crashed']} ❌")
        report_lines.append(f"Errors:       {summary['total_errors']} ⚠")
        report_lines.append("")
        
        # Category details
        report_lines.append("CATEGORY DETAILS")
        report_lines.append("-"*80)
        for cat_result in results["categories"]:
            report_lines.append(f"\n{cat_result['category']}:")
            report_lines.append(f"  Tests: {cat_result['total']}")
            report_lines.append(f"  Passed: {cat_result['passed']}")
            report_lines.append(f"  Crashed: {cat_result['crashed']}")
            
            # List crashes
            for test in cat_result["tests"]:
                if test.get("core_dump"):
                    report_lines.append(f"    ❌ {test['file']}: {test['core_dump']}")
                    if test.get("crash_info"):
                        info = test["crash_info"]
                        if "stack_trace" in info:
                            # Show top frame
                            frames = info["stack_trace"]
                            if frames:
                                top_frame = frames[0]
                                report_lines.append(f"      → {top_frame.get('function', 'unknown')}")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        return "\n".join(report_lines)
    
    def save_report(self, results: Dict, format: str = "both"):
        """Save report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format in ("json", "both"):
            json_path = os.path.join(REPORT_DIR, f"crash_report_{timestamp}.json")
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nJSON report saved: {json_path}")
        
        if format in ("text", "both"):
            text_path = os.path.join(REPORT_DIR, f"crash_report_{timestamp}.txt")
            report_text = self.generate_report(results)
            with open(text_path, "w") as f:
                f.write(report_text)
            print(f"Text report saved: {text_path}")
            print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(description="Run NeuronDB crash tests")
    parser.add_argument("--category", help="Run specific test category")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--db", default=DEFAULT_DB, help="Database name")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Database host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Database port")
    parser.add_argument("--user", default=DEFAULT_USER, help="Database user")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--format", choices=["json", "text", "both"], default="both",
                       help="Report format")
    
    args = parser.parse_args()
    
    runner = CrashTestRunner(
        db=args.db,
        host=args.host,
        port=args.port,
        user=args.user,
        verbose=args.verbose
    )
    
    if args.category:
        results = runner.run_test_category(args.category)
        # Wrap single category in full results format
        full_results = {
            "timestamp": datetime.now().isoformat(),
            "categories": [results],
            "summary": {
                "total_tests": results["total"],
                "total_passed": results["passed"],
                "total_crashed": results["crashed"],
                "total_errors": len(results["errors"])
            }
        }
        runner.save_report(full_results, args.format)
        
        # Exit with error code if crashes detected
        if results["crashed"] > 0:
            sys.exit(1)
    elif args.all:
        results = runner.run_all()
        runner.save_report(results, args.format)
        
        # Exit with error code if crashes detected
        if results["summary"]["total_crashed"] > 0:
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()


