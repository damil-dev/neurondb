# Crash Report Template (Regenerated)

## Summary

- What crashed:
- When:
- Environment:

## Reproduction

- Steps:
- Minimal SQL / command:

## Logs

- Postgres logs:
- Stack traces:

## Suspected root cause

- Module:
- Code path:

## Fix plan

- Proposed fix:
- Tests to add:

# NeuronDB Crash Report Template

## Crash Information

- **Date/Time**: [YYYY-MM-DD HH:MM:SS]
- **Test File**: `tests/sql/crash_prevention/[test_file].sql`
- **Core Dump**: `/tmp/core/core.[executable].[pid].[timestamp]`
- **Signal**: [SIGSEGV/SIGABRT/SIGFPE/etc.]

## Stack Trace

```
#0  [function] at [file]:[line]
#1  [function] at [file]:[line]
#2  [function] at [file]:[line]
...
```

**Crashing Function**: `[function_name]`  
**Crashing File**: `[file_path]:[line_number]`

## Reproduction Steps

1. [Step 1]
2. [Step 2]
3. [Step 3]

## SQL Query

```sql
[SQL query that caused the crash]
```

## Environment

- **PostgreSQL Version**: [version]
- **NeuronDB Version**: [version]
- **OS**: [OS version]
- **CPU**: [CPU info]
- **GPU**: [GPU info if applicable]
- **Memory**: [Memory info]

## Analysis

### Root Cause
[Description of the root cause]

### Code Location
- File: `[path/to/file.c]`
- Function: `[function_name]`
- Line: `[line_number]`

### Issue Type
- [ ] NULL pointer dereference
- [ ] Memory leak
- [ ] Double free
- [ ] Use after free
- [ ] Buffer overflow
- [ ] Integer overflow
- [ ] Array bounds violation
- [ ] Type confusion
- [ ] Uninitialized memory
- [ ] Race condition
- [ ] Other: [specify]

## Suggested Fix

```c
// Before (unsafe):
[code snippet]

// After (safe):
[code snippet]
```

## Test Case

Add to: `tests/sql/crash_prevention/[category].sql`

```sql
-- Test case to prevent regression
[SQL test case]
```

## Related Issues

- Related crash reports: [#issue1, #issue2]
- Related code changes: [commit hash]

## Status

- [ ] Reported
- [ ] Reproduced
- [ ] Root cause identified
- [ ] Fix implemented
- [ ] Test case added
- [ ] Verified fixed
- [ ] Closed

## Notes

[Additional notes]


