# NeuronDB Deployment Complete Guide

**Complete deployment guide for NeuronDB ecosystem with all Docker profiles and production considerations.**

> **Version:** 1.0  
> **Last Updated:** 2025-01-01

## Table of Contents

- [Docker Deployment](#docker-deployment)
- [Native Installation](#native-installation)
- [Production Considerations](#production-considerations)
- [Scaling Strategies](#scaling-strategies)
- [Monitoring Setup](#monitoring-setup)
- [Backup and Recovery](#backup-and-recovery)

---

## Docker Deployment

### Docker Compose Profiles

NeuronDB provides multiple Docker Compose profiles for different deployment scenarios:

#### CPU Profile

**Profile:** `cpu` or `default`

**Service:** `neurondb`

**Configuration:**
```yaml
services:
  neurondb:
    profiles:
      - default
      - cpu
    image: neurondb:cpu-pg17
    environment:
      POSTGRES_USER: neurondb
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: neurondb
      NEURONDB_GPU_BACKEND_TYPE: 0
      NEURONDB_COMPUTE_MODE: 0
    ports:
      - "5433:5432"
    volumes:
      - neurondb-data:/var/lib/postgresql/data
```

**Start:**
```bash
docker compose --profile cpu up -d
```

---

#### CUDA Profile

**Profile:** `cuda` or `gpu`

**Service:** `neurondb-cuda`

**Configuration:**
```yaml
services:
  neurondb-cuda:
    profiles:
      - cuda
      - gpu
    image: neurondb:cuda-pg17
    environment:
      POSTGRES_USER: neurondb
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: neurondb
      NVIDIA_VISIBLE_DEVICES: all
      NVIDIA_DRIVER_CAPABILITIES: compute,utility
      NEURONDB_GPU_ENABLED: on
      NEURONDB_GPU_BACKEND_TYPE: 1
      NEURONDB_COMPUTE_MODE: 1
    ports:
      - "5434:5432"
    volumes:
      - neurondb-cuda-data:/var/lib/postgresql/data
    # runtime: nvidia  # Requires nvidia-container-toolkit
```

**Requirements:**
- NVIDIA GPU with CUDA support
- nvidia-container-toolkit installed
- NVIDIA drivers

**Start:**
```bash
docker compose --profile cuda up -d
```

---

#### ROCm Profile

**Profile:** `rocm` or `gpu`

**Service:** `neurondb-rocm`

**Configuration:**
```yaml
services:
  neurondb-rocm:
    profiles:
      - rocm
      - gpu
    image: neurondb:rocm-pg17
    environment:
      POSTGRES_USER: neurondb
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: neurondb
      NEURONDB_GPU_ENABLED: on
      NEURONDB_GPU_BACKEND_TYPE: 2
      NEURONDB_COMPUTE_MODE: 1
    ports:
      - "5435:5432"
    volumes:
      - neurondb-rocm-data:/var/lib/postgresql/data
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
```

**Requirements:**
- AMD GPU with ROCm support
- ROCm drivers installed

**Start:**
```bash
docker compose --profile rocm up -d
```

---

### Environment Variables

**PostgreSQL:**
- `POSTGRES_USER`: Database user (default: `neurondb`)
- `POSTGRES_PASSWORD`: Database password (**REQUIRED for production**)
- `POSTGRES_DB`: Database name (default: `neurondb`)
- `POSTGRES_PORT`: External port (default: `5433`)

**NeuronDB:**
- `NEURONDB_GPU_BACKEND_TYPE`: GPU backend (0=CUDA, 1=ROCm, 2=Metal)
- `NEURONDB_COMPUTE_MODE`: Compute mode (0=CPU, 1=GPU, 2=Auto)
- `NEURONDB_GPU_ENABLED`: Enable GPU (on/off)

**Security:**
- **ALWAYS** set `POSTGRES_PASSWORD` in `.env` file
- Generate secure password: `openssl rand -base64 32`

---

### Docker Compose Examples

#### Basic CPU Deployment

```bash
# Create .env file
cat > .env << EOF
POSTGRES_PASSWORD=$(openssl rand -base64 32)
POSTGRES_USER=neurondb
POSTGRES_DB=neurondb
POSTGRES_PORT=5433
EOF

# Start services
docker compose --profile cpu up -d
```

#### CUDA Deployment

```bash
# Create .env file
cat > .env << EOF
POSTGRES_PASSWORD=$(openssl rand -base64 32)
POSTGRES_USER=neurondb
POSTGRES_DB=neurondb
POSTGRES_CUDA_PORT=5434
NVIDIA_VISIBLE_DEVICES=all
NEURONDB_GPU_ENABLED=on
NEURONDB_COMPUTE_MODE=1
EOF

# Start services
docker compose --profile cuda up -d
```

#### Full Stack Deployment

```bash
# Start all services (NeuronDB + NeuronAgent + NeuronMCP + NeuronDesktop)
docker compose --profile full up -d
```

---

## Native Installation

### PostgreSQL Extension Installation

**Requirements:**
- PostgreSQL 16, 17, or 18
- Build tools (gcc, make, cmake)
- Development headers

**Build:**
```bash
cd NeuronDB
make
make install
```

**Enable Extension:**
```sql
CREATE EXTENSION neurondb;
```

### GPU Backend Installation

#### CUDA

**Requirements:**
- NVIDIA GPU
- CUDA Toolkit 12.2+
- cuDNN

**Build:**
```bash
cd NeuronDB
make CUDA=1
make install
```

#### ROCm

**Requirements:**
- AMD GPU
- ROCm 5.7+

**Build:**
```bash
cd NeuronDB
make ROCm=1
make install
```

#### Metal

**Requirements:**
- Apple Silicon (M1/M2/M3)
- macOS 13+

**Build:**
```bash
cd NeuronDB
make Metal=1
make install
```

---

## Production Considerations

### Security

#### Password Management

**ALWAYS** use strong passwords in production:

```bash
# Generate secure password
openssl rand -base64 32

# Set in .env file
POSTGRES_PASSWORD=<generated_password>
```

#### Network Security

- Use firewall rules to restrict access
- Enable SSL/TLS for connections
- Use VPN or private networks
- Restrict database access to application servers

#### Authentication

- Use strong authentication methods
- Enable SSL/TLS
- Use connection pooling
- Implement rate limiting

---

### Resource Limits

#### CPU Profile

**Recommended:**
- **CPU:** 4+ cores
- **Memory:** 4GB+ RAM
- **Storage:** 20GB+ SSD

**Docker:**
```yaml
deploy:
  resources:
    limits:
      cpus: "4"
      memory: 4G
```

#### GPU Profile

**Recommended:**
- **CPU:** 8+ cores
- **Memory:** 8GB+ RAM
- **GPU:** NVIDIA/AMD GPU with 8GB+ VRAM
- **Storage:** 50GB+ SSD

**Docker:**
```yaml
deploy:
  resources:
    limits:
      cpus: "8"
      memory: 8G
```

---

### High Availability

#### Replication

**PostgreSQL Streaming Replication:**
```sql
-- Primary
CREATE USER replicator REPLICATION;
ALTER SYSTEM SET wal_level = replica;

-- Standby
-- Configure in postgresql.conf
```

#### Backup Strategy

**Automated Backups:**
```bash
# Daily backups
0 2 * * * pg_dump -U neurondb neurondb > /backups/neurondb_$(date +%Y%m%d).sql
```

**Point-in-Time Recovery:**
- Enable WAL archiving
- Regular base backups
- Test recovery procedures

---

## Scaling Strategies

### Horizontal Scaling

#### Read Replicas

**Setup:**
1. Create streaming replicas
2. Route read queries to replicas
3. Route write queries to primary

**Configuration:**
```sql
-- Primary
ALTER SYSTEM SET max_wal_senders = 3;
ALTER SYSTEM SET wal_keep_size = '1GB';

-- Replica
-- Configure in postgresql.conf
```

#### Connection Pooling

**Use PgBouncer:**
```yaml
services:
  pgbouncer:
    image: pgbouncer/pgbouncer
    environment:
      DATABASES_HOST: neurondb
      DATABASES_PORT: 5432
      DATABASES_USER: neurondb
      DATABASES_PASSWORD: ${POSTGRES_PASSWORD}
      DATABASES_DBNAME: neurondb
      POOL_MODE: transaction
      MAX_CLIENT_CONN: 1000
      DEFAULT_POOL_SIZE: 25
```

---

### Vertical Scaling

#### Resource Allocation

**Increase Resources:**
```yaml
deploy:
  resources:
    limits:
      cpus: "16"
      memory: 16G
    reservations:
      cpus: "8"
      memory: 8G
```

#### Index Optimization

**Tune Index Parameters:**
```sql
-- Increase HNSW ef_construction for better quality
SET neurondb.ef_construction = 400;

-- Increase ef_search for higher recall
SET neurondb.hnsw_ef_search = 128;
```

---

## Monitoring Setup

### Prometheus Metrics

**NeuronDB exposes Prometheus metrics:**

**Endpoint:** `http://localhost:5432/metrics`

**Metrics:**
- `neurondb_queries_total`: Total queries
- `neurondb_gpu_queries_total`: GPU queries
- `neurondb_gpu_fallback_total`: GPU fallbacks
- `neurondb_index_builds_total`: Index builds
- `neurondb_llm_requests_total`: LLM requests

**Prometheus Configuration:**
```yaml
scrape_configs:
  - job_name: 'neurondb'
    static_configs:
      - targets: ['localhost:5432']
```

---

### PostgreSQL Statistics

**View Statistics:**
```sql
-- Query statistics
SELECT * FROM pg_stat_neurondb;

-- GPU statistics
SELECT * FROM pg_stat_neurondb_gpu;

-- Index statistics
SELECT * FROM pg_stat_neurondb_indexes;
```

---

### Health Checks

**Docker Health Check:**
```yaml
healthcheck:
  test: ["CMD-SHELL", "pg_isready -U neurondb -d neurondb"]
  interval: 10s
  timeout: 5s
  retries: 5
  start_period: 40s
```

**Custom Health Check:**
```sql
-- Check extension
SELECT * FROM pg_extension WHERE extname = 'neurondb';

-- Check GPU
SELECT * FROM neurondb_gpu_info();

-- Check LLM
SELECT neurondb_llm_gpu_available();
```

---

## Backup and Recovery

### Backup Strategies

#### Full Backup

**pg_dump:**
```bash
pg_dump -U neurondb -d neurondb -F c -f neurondb_backup.dump
```

**Docker:**
```bash
docker exec neurondb-cpu pg_dump -U neurondb neurondb > backup.sql
```

#### Incremental Backup

**WAL Archiving:**
```sql
-- Enable WAL archiving
ALTER SYSTEM SET archive_mode = on;
ALTER SYSTEM SET archive_command = 'cp %p /backups/wal/%f';
```

#### Automated Backups

**Cron Job:**
```bash
# Daily backup
0 2 * * * docker exec neurondb-cpu pg_dump -U neurondb neurondb > /backups/neurondb_$(date +%Y%m%d).sql

# Weekly full backup
0 3 * * 0 docker exec neurondb-cpu pg_dump -U neurondb neurondb -F c > /backups/neurondb_$(date +%Y%m%d).dump
```

---

### Recovery Procedures

#### Point-in-Time Recovery

**Restore from Backup:**
```bash
# Restore base backup
pg_restore -U neurondb -d neurondb -F c neurondb_backup.dump

# Replay WAL
# Configure recovery.conf
```

#### Disaster Recovery

**Full Recovery:**
1. Restore base backup
2. Replay WAL archives
3. Verify data integrity
4. Test application connectivity

---

## Related Documentation

- [Quick Start Guide](../../QUICKSTART.md)
- [Configuration Reference](../reference/configuration-complete.md)
- [Troubleshooting](../getting-started/troubleshooting.md)

---

**Last Updated:** 2025-01-01  
**Documentation Version:** 1.0.0



