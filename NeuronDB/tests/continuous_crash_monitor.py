#!/usr/bin/env python3

"""
continuous_crash_monitor.py
Continuous crash monitoring for CI/CD integration.

Monitors for new core dumps, analyzes them, and can create GitHub issues.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

try:
    from crash_analyzer import CrashAnalyzer
except ImportError:
    CrashAnalyzer = None
    print("Warning: crash_analyzer.py not found")


CORE_DIR = "/tmp/core"
REPORT_DIR = os.path.join(os.path.dirname(__file__), "crash_reports")
CRASH_LOG = os.path.join(os.path.dirname(__file__), "crash_monitor.log")


class ContinuousCrashMonitor:
    """Monitor for crashes continuously."""
    
    def __init__(self, interval: int = 3600):
        self.interval = interval  # Check every hour by default
        self.analyzer = CrashAnalyzer() if CrashAnalyzer else None
        self.known_cores = set()
        self.load_known_cores()
    
    def load_known_cores(self):
        """Load list of known core dumps."""
        log_file = CRASH_LOG
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                for line in f:
                    if line.strip():
                        self.known_cores.add(line.strip())
    
    def save_known_core(self, core_path: str):
        """Mark a core dump as known."""
        self.known_cores.add(core_path)
        with open(CRASH_LOG, "a") as f:
            f.write(f"{core_path}\n")
    
    def get_new_cores(self) -> List[str]:
        """Get new core dump files."""
        import glob
        
        pattern = os.path.join(CORE_DIR, "core.*")
        all_cores = set(glob.glob(pattern))
        new_cores = all_cores - self.known_cores
        return sorted(list(new_cores))
    
    def analyze_new_crashes(self) -> List[dict]:
        """Analyze new crashes and return crash reports."""
        new_cores = self.get_new_cores()
        crashes = []
        
        for core_path in new_cores:
            print(f"Analyzing new crash: {core_path}")
            
            if self.analyzer:
                crash_info = self.analyzer.analyze(core_path, None)
                if crash_info:
                    crashes.append(crash_info)
            
            self.save_known_core(core_path)
        
        return crashes
    
    def generate_crash_report(self, crashes: List[dict]) -> dict:
        """Generate a summary report of crashes."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_crashes": len(crashes),
            "crashes": crashes,
            "summary": {
                "signals": {},
                "functions": {},
                "files": {}
            }
        }
        
        for crash in crashes:
            # Count signals
            signal = crash.get("signal", "UNKNOWN")
            report["summary"]["signals"][signal] = report["summary"]["signals"].get(signal, 0) + 1
            
            # Count functions
            if crash.get("stack_trace"):
                func = crash["stack_trace"][0].get("function", "unknown")
                report["summary"]["functions"][func] = report["summary"]["functions"].get(func, 0) + 1
                
                file = crash["stack_trace"][0].get("file", "unknown")
                report["summary"]["files"][file] = report["summary"]["files"].get(file, 0) + 1
        
        return report
    
    def save_report(self, report: dict):
        """Save crash report to file."""
        os.makedirs(REPORT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(REPORT_DIR, f"monitor_report_{timestamp}.json")
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved: {report_path}")
        return report_path
    
    def run_once(self) -> bool:
        """Run one check cycle. Returns True if crashes found."""
        crashes = self.analyze_new_crashes()
        
        if crashes:
            report = self.generate_crash_report(crashes)
            self.save_report(report)
            
            print(f"\n{'='*80}")
            print(f"CRASHES DETECTED: {len(crashes)}")
            print(f"{'='*80}")
            for crash in crashes:
                signal = crash.get("signal", "UNKNOWN")
                func = "unknown"
                if crash.get("stack_trace"):
                    func = crash["stack_trace"][0].get("function", "unknown")
                print(f"  Signal: {signal}, Function: {func}")
            print(f"{'='*80}\n")
            
            return True
        
        print(f"No new crashes (checked at {datetime.now()})")
        return False
    
    def run_continuous(self):
        """Run continuous monitoring loop."""
        print(f"Starting continuous crash monitor (check interval: {self.interval}s)")
        print(f"Monitoring: {CORE_DIR}")
        print(f"Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.run_once()
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print("\nStopped by user")


def main():
    parser = argparse.ArgumentParser(description="Continuous crash monitor for NeuronDB")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval", type=int, default=3600, help="Check interval in seconds")
    
    args = parser.parse_args()
    
    monitor = ContinuousCrashMonitor(interval=args.interval)
    
    if args.once:
        found = monitor.run_once()
        sys.exit(1 if found else 0)
    else:
        monitor.run_continuous()


if __name__ == "__main__":
    main()


