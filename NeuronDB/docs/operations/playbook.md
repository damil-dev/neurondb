# Operational Playbook

This playbook covers operational tasks including backup/restore, upgrades, observability, and capacity planning for NeuronDB in production.

## Backup and Restore

### Impact on Indexes

**Important:** Vector indexes (HNSW and IVF) are stored on disk and included in PostgreSQL backups, but they may need to be rebuilt after restore in some scenarios.

### Backup Strategies

#### 1. pg_dump (Logical Backup)

**Recommended for:**
- Small to medium databases (< 100GB)
- Cross-version upgrades
- Selective table backups

```bash
# Full database backup
pg_dump -h localhost -U postgres -d neurondb -F c -f neurondb_backup.dump

# Schema-only (for structure)
pg_dump -h localhost -U postgres -d neurondb --schema-only -f schema.sql

# Data-only (for content)
pg_dump -h localhost -U postgres -d neurondb --data-only -f data.sql
```

**Index handling:**
- Index definitions are included in schema backup
- Index data is included in full backup
- Indexes are automatically recreated on restore

#### 2. pg_basebackup (Physical Backup)

**Recommended for:**
- Large databases (> 100GB)
- Point-in-time recovery
- Full cluster backups

```bash
# Base backup
pg_basebackup -h localhost -U postgres -D /backup/neurondb -Ft -z -P

# With WAL archiving
pg_basebackup -h localhost -U postgres -D /backup/neurondb -Ft -z -P -X stream
```

**Index handling:**
- All index files are included in physical backup
- No rebuild required on restore

#### 3. Continuous Archiving (WAL)

**Recommended for:**
- Zero-downtime requirements
- Point-in-time recovery
- High availability setups

```sql
-- Enable WAL archiving
archive_mode = on
archive_command = 'cp %p /backup/wal/%f'
```

### Restore Procedures

#### Restore from pg_dump

```bash
# Drop existing database (if needed)
dropdb -h localhost -U postgres neurondb

# Create new database
createdb -h localhost -U postgres neurondb

# Restore
pg_restore -h localhost -U postgres -d neurondb -v neurondb_backup.dump

# Verify extension
psql -h localhost -U postgres -d neurondb -c "SELECT neurondb.version();"

# Rebuild indexes if needed
psql -h localhost -U postgres -d neurondb <<EOF
REINDEX INDEX documents_hnsw_idx;
ANALYZE documents;
EOF
```

#### Restore from Physical Backup

```bash
# Stop PostgreSQL
systemctl stop postgresql

# Restore base backup
rm -rf /var/lib/postgresql/data/*
tar -xzf /backup/neurondb/base.tar.gz -C /var/lib/postgresql/data/

# Restore WAL files (if using PITR)
cp /backup/wal/* /var/lib/postgresql/data/pg_wal/

# Start PostgreSQL
systemctl start postgresql

# Verify
psql -h localhost -U postgres -d neurondb -c "SELECT neurondb.version();"
```

### Backup Best Practices

1. **Regular backups:** Daily full backups + hourly WAL archives
2. **Test restores:** Regularly test restore procedures
3. **Offsite storage:** Store backups in separate location
4. **Encryption:** Encrypt backups containing sensitive data
5. **Retention:** Keep backups for at least 30 days, longer for compliance

## Upgrades

### Extension Version Upgrades

#### Minor/Patch Upgrades (e.g., 1.0.0 → 1.0.1)

**Process:**
1. Backup database
2. Stop services using NeuronDB
3. Install new extension version
4. Run upgrade SQL (if provided)
5. Restart services
6. Verify functionality

```bash
# 1. Backup
pg_dump -h localhost -U postgres -d neurondb -F c -f backup.dump

# 2. Stop services
systemctl stop neuronagent
systemctl stop neuronmcp

# 3. Install new version
cd NeuronDB
make install

# 4. Upgrade extension (if needed)
psql -h localhost -U postgres -d neurondb <<EOF
ALTER EXTENSION neurondb UPDATE TO '1.0.1';
EOF

# 5. Restart services
systemctl start neuronagent
systemctl start neuronmcp

# 6. Verify
./scripts/verify_neurondb_integration.sh --tier 0
```

#### Major Upgrades (e.g., 1.0.x → 2.0.0)

**Process:**
1. Review migration guide
2. Backup database
3. Test upgrade on staging
4. Schedule maintenance window
5. Perform upgrade
6. Run migration scripts
7. Verify all functionality
8. Monitor for issues

### PostgreSQL Major Upgrades

#### Upgrade from PostgreSQL 16 → 17

**Method 1: pg_upgrade (Recommended)**

```bash
# 1. Install PostgreSQL 17 alongside 16
apt-get install postgresql-17

# 2. Stop both versions
systemctl stop postgresql@16-main
systemctl stop postgresql@17-main

# 3. Run pg_upgrade
/usr/lib/postgresql/17/bin/pg_upgrade \
  --old-datadir=/var/lib/postgresql/16/main \
  --new-datadir=/var/lib/postgresql/17/main \
  --old-bindir=/usr/lib/postgresql/16/bin \
  --new-bindir=/usr/lib/postgresql/17/bin \
  --check

# 4. If check passes, run actual upgrade
/usr/lib/postgresql/17/bin/pg_upgrade \
  --old-datadir=/var/lib/postgresql/16/main \
  --new-datadir=/var/lib/postgresql/17/main \
  --old-bindir=/usr/lib/postgresql/16/bin \
  --new-bindir=/usr/lib/postgresql/17/bin

# 5. Rebuild NeuronDB extension
cd NeuronDB
PG_CONFIG=/usr/lib/postgresql/17/bin/pg_config make install

# 6. Start PostgreSQL 17
systemctl start postgresql@17-main

# 7. Verify
psql -h localhost -U postgres -d neurondb -c "SELECT neurondb.version();"
```

**Method 2: Logical Replication (Zero Downtime)**

```bash
# 1. Set up logical replication
# On source (PG 16):
ALTER SYSTEM SET wal_level = logical;
SELECT pg_reload_conf();

# 2. Create publication
CREATE PUBLICATION neurondb_pub FOR ALL TABLES;

# 3. On target (PG 17), create subscription
CREATE SUBSCRIPTION neurondb_sub 
CONNECTION 'host=source_host dbname=neurondb user=replicator'
PUBLICATION neurondb_pub;

# 4. Wait for sync, then switchover
```

### Upgrade Checklist

- [ ] Backup created and verified
- [ ] Migration guide reviewed
- [ ] Staging environment tested
- [ ] Maintenance window scheduled
- [ ] Rollback plan prepared
- [ ] Services stopped (if needed)
- [ ] Extension upgraded
- [ ] Database upgraded (if PG major version)
- [ ] Indexes verified
- [ ] Functionality tested
- [ ] Monitoring enabled
- [ ] Documentation updated

## Observability

### Metrics Exposed

#### PostgreSQL Standard Metrics

**pg_stat_statements:**
```sql
-- Enable extension
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Query performance
SELECT 
  query,
  calls,
  mean_exec_time,
  max_exec_time,
  total_exec_time
FROM pg_stat_statements
WHERE query LIKE '%neurondb%'
ORDER BY total_exec_time DESC;
```

**pg_stat_user_tables:**
```sql
-- Table statistics
SELECT 
  schemaname,
  tablename,
  seq_scan,
  idx_scan,
  n_tup_ins,
  n_tup_upd,
  n_tup_del,
  n_live_tup,
  n_dead_tup
FROM pg_stat_user_tables
WHERE schemaname = 'neurondb';
```

**pg_stat_user_indexes:**
```sql
-- Index statistics
SELECT 
  schemaname,
  tablename,
  indexname,
  idx_scan,
  idx_tup_read,
  idx_tup_fetch
FROM pg_stat_user_indexes
WHERE indexname LIKE '%hnsw%' OR indexname LIKE '%ivf%';
```

#### Custom NeuronDB Metrics

**Extension version:**
```sql
SELECT neurondb.version();
```

**GPU status:**
```sql
SELECT neurondb.gpu_enabled();
SELECT neurondb.gpu_device_count();
```

**Index information:**
```sql
SELECT 
  indexname,
  indexdef
FROM pg_indexes
WHERE indexdef LIKE '%hnsw%' OR indexdef LIKE '%ivf%';
```

### Logging Configuration

**PostgreSQL logging:**
```sql
-- Enable query logging
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_min_duration_statement = 1000;  -- Log queries > 1s
SELECT pg_reload_conf();
```

**NeuronDB logging:**
```sql
-- Set log level
SET neurondb.log_level = 'info';  -- or 'debug', 'warning', 'error'
```

### Slow Query Hooks

**Create slow query log table:**
```sql
CREATE TABLE slow_queries (
  id SERIAL PRIMARY KEY,
  query TEXT,
  duration_ms FLOAT,
  created_at TIMESTAMP DEFAULT NOW()
);
```

**Enable slow query tracking:**
```sql
-- Using pg_stat_statements
SELECT 
  query,
  mean_exec_time,
  calls
FROM pg_stat_statements
WHERE mean_exec_time > 1000  -- > 1 second
ORDER BY mean_exec_time DESC;
```

### Monitoring Setup

**Prometheus exporters:**
- `postgres_exporter` for PostgreSQL metrics
- Custom exporter for NeuronDB-specific metrics

**Grafana dashboards:**
- Query performance over time
- Index usage statistics
- Vector operation latency
- Error rates

## Capacity Planning

### Memory Requirements

**PostgreSQL shared_buffers:**
```
shared_buffers = RAM * 0.25  # 25% of total RAM
```

**work_mem for index builds:**
```
work_mem = 256MB  # For large index builds
```

**NeuronDB index memory:**
- HNSW: ~3-4x vector data size
- IVF: ~1.5-2x vector data size

**Example calculation:**
- 10M vectors, 768 dimensions = ~30GB data
- HNSW index: ~90-120GB memory
- Total RAM needed: ~150GB+ (including PostgreSQL overhead)

### Disk I/O Patterns

**Index build:**
- High sequential write during build
- Random reads during construction
- Plan for 2-3x data size temporary space

**Query patterns:**
- Random reads from index
- Sequential reads for vector data
- SSD recommended for production

**WAL impact:**
- Vector inserts generate WAL
- Large batch inserts: consider `UNLOGGED` tables during load, then convert

### Connection Pooling

**Recommended pool sizes:**
- **NeuronAgent:** 10-50 connections (depending on load)
- **NeuronMCP:** 5-20 connections
- **Application:** 20-100 connections

**PostgreSQL max_connections:**
```
max_connections = sum_of_all_pools + 20  # +20 for admin connections
```

**Connection pooler (PgBouncer):**
- Use transaction pooling for better efficiency
- Pool size: 2-3x application connections

### Scaling Strategies

**Vertical scaling:**
- Increase RAM for larger indexes
- Faster CPU for query performance
- SSD storage for I/O performance

**Horizontal scaling:**
- Read replicas for query distribution
- Partitioning for very large tables
- Sharding for multi-tenant scenarios

## Operational Checklist

- [ ] **Backups:** Configured and tested
- [ ] **Monitoring:** Metrics and logging enabled
- [ ] **Alerts:** Set up for critical issues
- [ ] **Documentation:** Runbooks and procedures documented
- [ ] **Capacity:** Resources sized appropriately
- [ ] **Security:** Access controls and encryption configured
- [ ] **Performance:** Baseline metrics established
- [ ] **Disaster recovery:** Plan tested and documented

## Related Documentation

- [Configuration Guide](configuration.md) - All configuration parameters
- [Performance Playbook](performance/playbook.md) - Performance optimization
- [Troubleshooting Guide](troubleshooting.md) - Common issues and solutions
- [Security Guide](security/overview.md) - Security best practices

