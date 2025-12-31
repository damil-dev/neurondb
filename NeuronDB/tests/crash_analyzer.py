#!/usr/bin/env python3

"""
crash_analyzer.py
Automated core dump analysis using GDB.

Extracts stack traces, registers, and crash information from core dumps.
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


class CrashAnalyzer:
    """Analyze PostgreSQL core dumps with GDB."""
    
    def __init__(self, postgres_bin: Optional[str] = None):
        self.postgres_bin = postgres_bin or self.find_postgres_binary()
        self.core_analysis_dir = os.path.join(os.path.dirname(__file__), "core_analysis")
        os.makedirs(self.core_analysis_dir, exist_ok=True)
    
    def find_postgres_binary(self) -> Optional[str]:
        """Find PostgreSQL binary for core dump analysis."""
        # Common locations
        common_paths = [
            "/usr/local/pgsql/bin/postgres",
            "/usr/local/pgsql.18-pge/bin/postgres",
            "/usr/local/pgsql.17-pge/bin/postgres",
            "/usr/bin/postgres",
            "/usr/lib/postgresql/17/bin/postgres",
            "/usr/lib/postgresql/18/bin/postgres",
        ]
        
        # Try which first
        try:
            result = subprocess.run(
                ["which", "postgres"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                path = result.stdout.strip()
                if os.path.exists(path):
                    return path
        except Exception:
            pass
        
        # Try common paths
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def analyze(self, core_path: str, sql_file: Optional[str] = None) -> Optional[Dict]:
        """
        Analyze a core dump file.
        
        Returns:
            Dictionary with crash information:
            {
                "signal": "SIGSEGV",
                "stack_trace": [...],
                "registers": {...},
                "locals": {...},
                "error_message": "..."
            }
        """
        if not os.path.exists(core_path):
            return None
        
        if not self.postgres_bin:
            print("Warning: PostgreSQL binary not found, cannot analyze core dump")
            return None
        
        # Generate output filename
        core_name = os.path.basename(core_path)
        output_file = os.path.join(self.core_analysis_dir, f"{core_name}.txt")
        
        # Run GDB
        gdb_commands = [
            "set pagination off",
            f"file {self.postgres_bin}",
            f"core-file {core_path}",
            "bt",  # Backtrace
            "info registers",
            "info locals",
            "thread apply all bt",  # All threads
            "quit"
        ]
        
        try:
            cmd = ["gdb", "-batch"] + [arg for cmd in gdb_commands for arg in ["-ex", cmd]]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False
            )
            
            # Save full output
            with open(output_file, "w") as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n--- STDERR ---\n")
                    f.write(result.stderr)
            
            # Parse output
            crash_info = self.parse_gdb_output(result.stdout, result.stderr)
            crash_info["core_dump"] = core_path
            crash_info["sql_file"] = sql_file
            crash_info["analysis_file"] = output_file
            
            return crash_info
            
        except subprocess.TimeoutExpired:
            return {"error": "GDB analysis timed out", "core_dump": core_path}
        except Exception as e:
            return {"error": f"GDB analysis failed: {e}", "core_dump": core_path}
    
    def parse_gdb_output(self, stdout: str, stderr: str) -> Dict:
        """Parse GDB output to extract crash information."""
        info = {
            "signal": None,
            "stack_trace": [],
            "registers": {},
            "error_message": None
        }
        
        # Look for signal information
        signal_patterns = [
            r"Program received signal (\S+)",
            r"Program terminated with signal (\S+)",
        ]
        for pattern in signal_patterns:
            match = re.search(pattern, stdout + stderr)
            if match:
                info["signal"] = match.group(1)
                break
        
        # Parse stack trace
        lines = stdout.split("\n")
        in_backtrace = False
        current_frame = None
        
        for line in lines:
            # Start of backtrace
            if re.match(r"^#0\s", line):
                in_backtrace = True
                current_frame = self.parse_frame(line)
                if current_frame:
                    info["stack_trace"].append(current_frame)
            elif in_backtrace and re.match(r"^#\d+\s", line):
                if current_frame:
                    info["stack_trace"].append(current_frame)
                current_frame = self.parse_frame(line)
            elif in_backtrace and line.strip() == "":
                # End of backtrace
                if current_frame:
                    info["stack_trace"].append(current_frame)
                break
            elif in_backtrace and current_frame:
                # Continuation of current frame
                current_frame["details"] = current_frame.get("details", "") + " " + line.strip()
        
        # Look for error messages
        error_patterns = [
            r"Assertion.*failed",
            r"ERROR:.*",
            r"FATAL:.*",
        ]
        for pattern in error_patterns:
            match = re.search(pattern, stdout + stderr, re.IGNORECASE)
            if match:
                info["error_message"] = match.group(0)
                break
        
        return info
    
    def parse_frame(self, line: str) -> Optional[Dict]:
        """Parse a stack frame line from GDB."""
        # Pattern: #0  function_name (args) at file:line
        pattern = r"#(\d+)\s+(.+?)\s+\([^)]*\)\s+at\s+(.+?):(\d+)"
        match = re.search(pattern, line)
        if match:
            return {
                "frame": int(match.group(1)),
                "function": match.group(2),
                "file": match.group(3),
                "line": int(match.group(4)),
                "full": line.strip()
            }
        
        # Pattern without file:line: #0  function_name (args)
        pattern2 = r"#(\d+)\s+(.+?)\s+\([^)]*\)"
        match = re.search(pattern2, line)
        if match:
            return {
                "frame": int(match.group(1)),
                "function": match.group(2),
                "full": line.strip()
            }
        
        # Minimal pattern: #0  address in function_name
        pattern3 = r"#(\d+)\s+0x[0-9a-f]+\s+in\s+(.+)"
        match = re.search(pattern3, line)
        if match:
            return {
                "frame": int(match.group(1)),
                "function": match.group(2),
                "full": line.strip()
            }
        
        return None
    
    def get_crashing_function(self, crash_info: Dict) -> Optional[str]:
        """Extract the function name where the crash occurred."""
        if crash_info.get("stack_trace"):
            top_frame = crash_info["stack_trace"][0]
            return top_frame.get("function")
        return None
    
    def get_crashing_file(self, crash_info: Dict) -> Optional[str]:
        """Extract the source file where the crash occurred."""
        if crash_info.get("stack_trace"):
            top_frame = crash_info["stack_trace"][0]
            return top_frame.get("file")
        return None


def main():
    """CLI for analyzing a single core dump."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze a PostgreSQL core dump")
    parser.add_argument("core_dump", help="Path to core dump file")
    parser.add_argument("--postgres-bin", help="Path to PostgreSQL binary")
    parser.add_argument("--sql-file", help="SQL file that caused the crash")
    
    args = parser.parse_args()
    
    analyzer = CrashAnalyzer(postgres_bin=args.postgres_bin)
    crash_info = analyzer.analyze(args.core_dump, args.sql_file)
    
    if crash_info:
        import json
        print(json.dumps(crash_info, indent=2))
    else:
        print("Failed to analyze core dump")
        sys.exit(1)


if __name__ == "__main__":
    main()

