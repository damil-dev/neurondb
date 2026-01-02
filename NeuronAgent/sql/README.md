# NeuronAgent SQL Migrations

This directory contains all SQL migration files for NeuronAgent.

## File Naming Convention

Files are named with the pattern: `NNN_description.sql` where:
- `NNN` is a 3-digit zero-padded sequence number (001, 002, 003, ...)
- `description` is a descriptive name using underscores (e.g., `initial_schema`, `add_indexes`)

## Migration Order

Migrations are applied in numerical order. Each migration file should be:
- Idempotent (safe to run multiple times)
- Reversible (when possible)
- Well-documented with comments

## Current Migrations

- 001_initial_schema.sql
- 002_add_indexes.sql
- 003_add_triggers.sql
- 004_advanced_features.sql
- 005_budget_schema.sql
- 006_webhooks_schema.sql
- 007_human_in_loop_schema.sql
- 008_principals_and_permissions.sql
- 009_execution_snapshots.sql
- 010_evaluation_framework.sql
- 011_workflow_engine.sql
- 012_browser_sessions.sql
- 013_collaboration_workspace.sql
- 014_hierarchical_memory.sql
- 015_event_stream.sql
- 016_verification_agent.sql
- 017_virtual_filesystem.sql
- 018_async_tasks.sql
- 019_sub_agents.sql
- 020_task_alerts.sql
