# NeuronMCP SQL Migrations

This directory contains all SQL migration files for NeuronMCP.

## File Naming Convention

Files are named with the pattern: `NNN_description.sql` where:
- `NNN` is a 3-digit zero-padded sequence number (001, 002, 003, ...)
- `description` is a descriptive name using underscores (e.g., `initial_schema`, `functions`)

## Migration Order

Migrations are applied in numerical order. Each migration file should be:
- Idempotent (safe to run multiple times)
- Reversible (when possible)
- Well-documented with comments

## Current Migrations

- 000_validate_setup.sql - Validation script (run this first to verify setup)
- 001_initial_schema.sql - Initial schema setup (tables, indexes, views)
- 002_functions.sql - Management functions and stored procedures

## Usage

### Quick Validation
Run the validation script to verify if everything is set up correctly:
```bash
psql -d neurondb -f sql/000_validate_setup.sql
```

### Full Setup
If validation fails, run the setup scripts in order:
```bash
psql -d neurondb -f sql/001_initial_schema.sql
psql -d neurondb -f sql/002_functions.sql
```
