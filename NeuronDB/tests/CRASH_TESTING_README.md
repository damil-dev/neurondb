# Crash Testing README (Regenerated)

This directory includes scripts and SQL designed to find and prevent crashes.

## Philosophy

- Prefer reproducible SQL that triggers the issue.
- Add regression tests for every crash fix.
- Include boundary condition tests (nulls, empty arrays, overflow, invalid models).

## Where to look

- SQL crash suite: `NeuronDB/tests/sql/crash_prevention/`
- Harness scripts: `NeuronDB/tests/*.py`

# NeuronDB Crash Testing Suite

This directory contains comprehensive crash testing tools and test cases for NeuronDB to achieve zero crash tolerance.

## Quick Start

### 1. Setup Core Dumps

```bash
# Run the setup script (may require sudo)
./tests/setup_core_dumps.sh

# Verify core dumps are enabled
ulimit -c unlimited
```

### 2. Run All Crash Tests

```bash
# Run all crash prevention tests
python3 tests/crash_test_runner.py --all

# Run specific category
python3 tests/crash_test_runner.py --category null_parameters

# Generate reports
python3 tests/crash_test_runner.py --all --format both
```

### 3. Analyze Core Dumps

```bash
# Manual analysis of a specific core dump
python3 tests/crash_analyzer.py /tmp/core/core.postgres.12345 --sql-file tests/sql/crash_prevention/001_null_parameters.sql

# Batch analysis of all core dumps
./tests/analyze_core_dumps.sh
```

## Test Files

All crash prevention test files are in `tests/sql/crash_prevention/`:

1. **001_null_parameters_comprehensive.sql** - NULL parameter injection tests
2. **002_invalid_models_exhaustive.sql** - Invalid model handling tests
3. **003_spi_failures_complete.sql** - SPI failure scenario tests
4. **004_memory_stress_extreme.sql** - Memory stress tests
5. **005_array_bounds_fuzzing.sql** - Array bounds and dimension tests
6. **006_overflow_attacks.sql** - Integer overflow tests
7. **007_gpu_failures.sql** - GPU-specific failure tests
8. **008_index_crashes.sql** - Index build crash tests
9. **009_type_confusion.sql** - Type casting tests
10. **010_concurrency_hammer.sql** - Concurrency stress tests
11. **011_algorithm_boundaries.sql** - Algorithm boundary condition tests

## Tools

### crash_test_runner.py

Master test orchestrator that:
- Runs SQL test files
- Monitors for core dumps
- Analyzes crashes automatically
- Generates comprehensive reports

Usage:
```bash
python3 tests/crash_test_runner.py --all
python3 tests/crash_test_runner.py --category null_parameters --verbose
python3 tests/crash_test_runner.py --all --format json --output report.json
```

### crash_analyzer.py

Automated GDB-based core dump analyzer.

Usage:
```bash
python3 tests/crash_analyzer.py /path/to/core.dump
python3 tests/crash_analyzer.py /path/to/core.dump --postgres-bin /usr/local/pgsql/bin/postgres
```

### fuzz_test_generator.py

Generates random fuzz tests for comprehensive coverage.

Usage:
```bash
# Generate 1000 random tests
python3 tests/fuzz_test_generator.py --count 1000 --output fuzz_tests.sql

# Generate with seed for reproducibility
python3 tests/fuzz_test_generator.py --count 10000 --seed 42
```

### continuous_crash_monitor.py

Continuous monitoring tool for CI/CD integration.

Usage:
```bash
# Run once and exit
python3 tests/continuous_crash_monitor.py --once

# Run continuously (check every hour)
python3 tests/continuous_crash_monitor.py --interval 3600
```

## Test Categories

### NULL Parameter Tests

Tests all functions with NULL parameters to ensure graceful error handling.

```bash
python3 tests/crash_test_runner.py --category null_parameters
```

### Invalid Model Tests

Tests handling of invalid/non-existent models.

```bash
python3 tests/crash_test_runner.py --category invalid_models
```

### SPI Failure Tests

Tests Server Programming Interface failure scenarios.

```bash
python3 tests/crash_test_runner.py --category spi_failures
```

### Memory Stress Tests

Tests memory handling under extreme conditions.

```bash
python3 tests/crash_test_runner.py --category memory_stress
```

## Reports

Test reports are saved in `tests/crash_reports/`:

- **JSON reports**: Machine-readable format with full crash details
- **Text reports**: Human-readable summary reports

Example report structure:
```
crash_report_20250116_123456.json  # Full JSON report
crash_report_20250116_123456.txt   # Text summary
```

## Core Dump Analysis

Core dumps are saved to `/tmp/core/` with the pattern:
```
core.<executable>.<pid>.<timestamp>
```

Analysis output is saved to `tests/core_analysis/`:
```
core_analysis/postgres.postgres.12345.txt
```

## Integration with CI/CD

Add to your CI pipeline:

```yaml
# Example GitHub Actions workflow
- name: Run crash tests
  run: |
    ./tests/setup_core_dumps.sh
    python3 tests/crash_test_runner.py --all --format json
  continue-on-error: true

- name: Check for crashes
  run: |
    if [ -n "$(find /tmp/core -name 'core.*' -mmin -5)" ]; then
      echo "Crashes detected!"
      python3 tests/crash_test_runner.py --all --format json
      exit 1
    fi
```

## Manual Testing Protocol

1. **Pre-test Setup**
   ```bash
   ./tests/setup_core_dumps.sh
   ```

2. **Run Tests**
   ```bash
   python3 tests/crash_test_runner.py --all --verbose
   ```

3. **Review Reports**
   ```bash
   cat tests/crash_reports/crash_report_*.txt
   ```

4. **Analyze Crashes**
   ```bash
   ./tests/analyze_core_dumps.sh
   cat tests/core_analysis/*.txt
   ```

5. **Fix Issues**
   - Review stack traces in core analysis
   - Implement fixes in C code
   - Add regression tests

6. **Verify Fixes**
   ```bash
   python3 tests/crash_test_runner.py --category [affected_category]
   ```

## Code Audit Checklist

See `tests/c_code_audit_checklist.md` for systematic code review checklist.

## Crash Report Template

Use `tests/crash_report_template.md` when reporting new crashes.

## Success Metrics

Target metrics:
- **Crash-Free Rate**: 100% (0 crashes in 100,000 tests)
- **Test Coverage**: 100% of exported SQL functions
- **Code Coverage**: >90% of C code in src/ml/, src/index/
- **Core Dumps**: Zero new core dumps in 30-day test period

## Troubleshooting

### Core dumps not being generated

1. Check ulimit:
   ```bash
   ulimit -c
   # Should show "unlimited"
   ```

2. Check core pattern:
   ```bash
   cat /proc/sys/kernel/core_pattern
   # Should point to /tmp/core/
   ```

3. Check directory permissions:
   ```bash
   ls -ld /tmp/core
   # Should be writable
   ```

### GDB not found

Install gdb:
```bash
# Ubuntu/Debian
sudo apt-get install gdb

# RHEL/CentOS
sudo yum install gdb
```

### PostgreSQL binary not found

Specify path manually:
```bash
python3 tests/crash_analyzer.py core.dump --postgres-bin /usr/local/pgsql/bin/postgres
```

## Contributing

When adding new crash tests:

1. Add test to appropriate category file in `tests/sql/crash_prevention/`
2. Update this README if adding new categories
3. Run tests to ensure they work: `python3 tests/crash_test_runner.py --category [your_category]`
4. Add to CI/CD pipeline if critical

## Related Documentation

- `tests/c_code_audit_checklist.md` - Code review checklist
- `tests/crash_report_template.md` - Crash reporting template
- `tests/sql/crash_prevention/readme.md` - Test suite overview


