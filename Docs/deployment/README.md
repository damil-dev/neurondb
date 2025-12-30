# NeuronDB Deployment Overview

**Complete deployment strategies for the NeuronDB ecosystem**

This guide covers all deployment options, from development to production, including single-server, multi-server, containerized, and orchestrated deployments.

---

## 🎯 Deployment Options Summary

| Option | Use Case | Complexity | Scalability | HA | Cost |
|--------|----------|------------|-------------|-------|------|
| **[Docker Compose](#docker-compose)** | Development, small production | ⭐ Easy | ⭐ Limited | ❌ No | 💰 Low |
| **[Individual Containers](#individual-containers)** | Medium production | ⭐⭐ Moderate | ⭐⭐ Good | ⚠️ Partial | 💰💰 Medium |
| **[Kubernetes](#kubernetes)** | Large-scale production | ⭐⭐⭐ Complex | ⭐⭐⭐ Excellent | ✅ Yes | 💰💰💰 High |
| **[Bare Metal](#bare-metal)** | High-performance | ⭐⭐ Moderate | ⭐⭐ Good | ⚠️ Manual | 💰💰 Medium |

---

## 🐳 Docker Compose

**Best for:** Development, testing, small production deployments

### Quick Start

```bash
# Start all services
docker compose up -d

# Verify services
docker compose ps

# Test
./scripts/smoke-test.sh
```

### Configuration

**Default Configuration (CPU):**
```yaml
services:
  neurondb:
    image: neurondb:cpu-pg17
    ports: ["5433:5432"]
    
  neuronagent:
    image: neuronagent:latest
    ports: ["8080:8080"]
    
  neurondb-mcp:
    image: neurondb-mcp:latest
    
  neurondesk-api:
    image: neurondesk-api:latest
    ports: ["8081:8081"]
    
  neurondesk-frontend:
    image: neurondesk-frontend:latest
    ports: ["3000:3000"]
```

**GPU Configuration (CUDA):**
```bash
# Start with CUDA profile
docker compose --profile cuda up -d

# Or modify docker-compose.yml
services:
  neurondb:
    image: neurondb:cuda-pg17
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Advantages

✅ **Simple setup** - Single command to start everything  
✅ **Easy configuration** - Environment variables and YAML  
✅ **Quick iteration** - Fast rebuild and restart  
✅ **Integrated networking** - Automatic service discovery  
✅ **Volume management** - Easy data persistence  

### Limitations

❌ **No HA** - Single point of failure  
❌ **Limited scaling** - Can't scale beyond single host  
❌ **Manual updates** - No rolling updates  
❌ **Resource limits** - Single host constraints  

### Production Considerations

```yaml
services:
  neurondb:
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
    volumes:
      - neurondb-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "neurondb"]
      interval: 10s
      timeout: 5s
      retries: 5
```

---

## 📦 Individual Containers

**Best for:** Medium-scale production, service isolation

### Architecture

```
┌─────────────────────────────────────────────┐
│              Load Balancer                  │
│         (nginx/haproxy/traefik)             │
└──────┬────────────────────────────┬─────────┘
       │                            │
┌──────▼──────────┐      ┌─────────▼─────────┐
│  NeuronAgent    │      │  NeuronDesktop    │
│  (3 replicas)   │      │  (2 replicas)     │
└──────┬──────────┘      └─────────┬─────────┘
       │                            │
       └────────────┬───────────────┘
                    │
         ┌──────────▼──────────┐
         │      NeuronDB       │
         │   (Primary + HA)    │
         └─────────────────────┘
```

### Deployment Steps

**1. Setup Network:**
```bash
# Create bridge network
docker network create neurondb-network
```

**2. Deploy Database:**
```bash
# Primary database
docker run -d \
  --name neurondb-primary \
  --network neurondb-network \
  -p 5433:5432 \
  -e POSTGRES_PASSWORD=secret \
  -v neurondb-data:/var/lib/postgresql/data \
  neurondb:cpu-pg17

# Replica (read-only)
docker run -d \
  --name neurondb-replica \
  --network neurondb-network \
  -p 5434:5432 \
  -e POSTGRES_PASSWORD=secret \
  -e PRIMARY_HOST=neurondb-primary \
  -v neurondb-replica-data:/var/lib/postgresql/data \
  neurondb:cpu-pg17
```

**3. Deploy Agent:**
```bash
# Agent instance 1
docker run -d \
  --name neuronagent-1 \
  --network neurondb-network \
  -p 8080:8080 \
  -e DB_HOST=neurondb-primary \
  -e DB_PORT=5432 \
  -e DB_NAME=neurondb \
  neuronagent:latest

# Agent instance 2
docker run -d \
  --name neuronagent-2 \
  --network neurondb-network \
  -p 8081:8080 \
  -e DB_HOST=neurondb-primary \
  neuronagent:latest
```

**4. Deploy MCP:**
```bash
docker run -d \
  --name neurondb-mcp \
  --network neurondb-network \
  -e NEURONDB_HOST=neurondb-primary \
  neurondb-mcp:latest
```

**5. Deploy Desktop:**
```bash
# Backend API
docker run -d \
  --name neurondesk-api \
  --network neurondb-network \
  -p 8082:8081 \
  -e DB_HOST=neurondb-primary \
  neurondesk-api:latest

# Frontend
docker run -d \
  --name neurondesk-frontend \
  --network neurondb-network \
  -p 3000:3000 \
  -e NEXT_PUBLIC_API_URL=http://localhost:8082 \
  neurondesk-frontend:latest
```

**6. Setup Load Balancer:**
```nginx
upstream neuronagent {
    server 127.0.0.1:8080;
    server 127.0.0.1:8081;
}

server {
    listen 80;
    server_name agent.example.com;
    
    location / {
        proxy_pass http://neuronagent;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Advantages

✅ **Service isolation** - Independent scaling  
✅ **Better resource control** - Per-service limits  
✅ **Easier updates** - Update one service at a time  
✅ **Load balancing** - Multiple instances per service  

### Scaling Example

```bash
# Scale agent instances
for i in {3..5}; do
  docker run -d \
    --name neuronagent-$i \
    --network neurondb-network \
    -e DB_HOST=neurondb-primary \
    neuronagent:latest
done

# Update load balancer
```

---

## ☸️ Kubernetes

**Best for:** Large-scale production, high availability

### Architecture

```
┌─────────────────────────────────────────────┐
│              Ingress Controller             │
│           (nginx/traefik/istio)             │
└──────┬──────────────────────────────────────┘
       │
┌──────▼───────────────────────────────────┐
│         Kubernetes Cluster                │
│                                           │
│  ┌─────────────────────────────────────┐ │
│  │  NeuronAgent Deployment (HPA)       │ │
│  │  - Replicas: 3-10 (autoscale)      │ │
│  │  - Service: LoadBalancer           │ │
│  └─────────────────────────────────────┘ │
│                                           │
│  ┌─────────────────────────────────────┐ │
│  │  NeuronDesktop Deployment           │ │
│  │  - API Replicas: 2                  │ │
│  │  - Frontend Replicas: 3             │ │
│  └─────────────────────────────────────┘ │
│                                           │
│  ┌─────────────────────────────────────┐ │
│  │  NeuronDB StatefulSet               │ │
│  │  - Primary: 1                       │ │
│  │  - Replicas: 2 (read-only)         │ │
│  │  - PersistentVolumeClaims          │ │
│  └─────────────────────────────────────┘ │
└───────────────────────────────────────────┘
```

### Deployment Manifests

**1. Namespace:**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: neurondb
```

**2. NeuronDB StatefulSet:**
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: neurondb
  namespace: neurondb
spec:
  serviceName: neurondb
  replicas: 3
  selector:
    matchLabels:
      app: neurondb
  template:
    metadata:
      labels:
        app: neurondb
    spec:
      containers:
      - name: neurondb
        image: neurondb:cpu-pg17
        ports:
        - containerPort: 5432
          name: postgres
        env:
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: neurondb-secret
              key: password
        volumeMounts:
        - name: data
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
```

**3. NeuronAgent Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neuronagent
  namespace: neurondb
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neuronagent
  template:
    metadata:
      labels:
        app: neuronagent
    spec:
      containers:
      - name: neuronagent
        image: neuronagent:latest
        ports:
        - containerPort: 8080
        env:
        - name: DB_HOST
          value: "neurondb-0.neurondb.neurondb.svc.cluster.local"
        - name: DB_PORT
          value: "5432"
        - name: DB_NAME
          value: "neurondb"
        - name: DB_USER
          value: "neurondb"
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: neurondb-secret
              key: password
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: neuronagent
  namespace: neurondb
spec:
  type: LoadBalancer
  ports:
  - port: 8080
    targetPort: 8080
  selector:
    app: neuronagent
```

**4. Horizontal Pod Autoscaler:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: neuronagent-hpa
  namespace: neurondb
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: neuronagent
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Deployment

```bash
# Apply manifests
kubectl apply -f namespace.yaml
kubectl apply -f secrets.yaml
kubectl apply -f neurondb-statefulset.yaml
kubectl apply -f neuronagent-deployment.yaml
kubectl apply -f neuronagent-hpa.yaml
kubectl apply -f neurondesktop-deployment.yaml
kubectl apply -f ingress.yaml

# Verify
kubectl get pods -n neurondb
kubectl get services -n neurondb
kubectl get hpa -n neurondb
```

### Advantages

✅ **High availability** - Automatic failover  
✅ **Auto-scaling** - Based on metrics  
✅ **Rolling updates** - Zero-downtime deployments  
✅ **Service discovery** - Built-in DNS  
✅ **Load balancing** - Automatic distribution  
✅ **Self-healing** - Automatic pod replacement  

---

## 🖥️ Bare Metal

**Best for:** High-performance requirements, full control

### Architecture

```
Server 1 (Database Primary)
├── PostgreSQL 17 + NeuronDB
├── 32GB RAM, 16 cores
└── NVMe SSD array (RAID 10)

Server 2 (Database Replica)
├── PostgreSQL 17 + NeuronDB (read-only)
├── 32GB RAM, 16 cores
└── NVMe SSD array (RAID 10)

Server 3 (Application 1)
├── NeuronAgent
├── NeuronDesktop API
└── 16GB RAM, 8 cores

Server 4 (Application 2)
├── NeuronAgent
├── NeuronDesktop API
└── 16GB RAM, 8 cores

Server 5 (Frontend)
├── NeuronDesktop Frontend
├── nginx
└── 8GB RAM, 4 cores
```

### Installation Steps

**1. Install PostgreSQL + NeuronDB:**
```bash
# On database servers
sudo apt-get install postgresql-17

# Build and install NeuronDB
cd NeuronDB
make PG_CONFIG=/usr/lib/postgresql/17/bin/pg_config
sudo make install PG_CONFIG=/usr/lib/postgresql/17/bin/pg_config

# Enable extension
psql -d neurondb -c "CREATE EXTENSION neurondb;"
```

**2. Configure Replication:**
```bash
# Primary (server 1)
# postgresql.conf
wal_level = replica
max_wal_senders = 3
max_replication_slots = 3

# pg_hba.conf
host replication neurondb 10.0.0.2/32 md5

# Replica (server 2)
# Create replication slot
pg_basebackup -h 10.0.0.1 -U neurondb -D /var/lib/postgresql/17/main -P -R
```

**3. Install NeuronAgent:**
```bash
# On application servers
cd NeuronAgent
go build ./cmd/agent-server

# Create systemd service
sudo cat > /etc/systemd/system/neuronagent.service <<EOF
[Unit]
Description=NeuronAgent Service
After=network.target

[Service]
Type=simple
User=neuronagent
Environment="DB_HOST=10.0.0.1"
Environment="DB_PORT=5432"
Environment="SERVER_PORT=8080"
ExecStart=/opt/neuronagent/agent-server
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable neuronagent
sudo systemctl start neuronagent
```

**4. Setup Load Balancing:**
```nginx
# On frontend server
upstream neuronagent {
    server 10.0.0.3:8080;
    server 10.0.0.4:8080;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;
    
    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;
    
    location / {
        proxy_pass http://neuronagent;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Advantages

✅ **Maximum performance** - No containerization overhead  
✅ **Full control** - Complete system access  
✅ **Hardware optimization** - Direct GPU access  
✅ **Cost effective** - No orchestration overhead  

### Limitations

❌ **Manual management** - Requires system administration  
❌ **Slower updates** - Manual deployment process  
❌ **Complex scaling** - Manual server provisioning  

---

## 🔒 Security Best Practices

### Network Security

```bash
# Firewall rules
sudo ufw allow 443/tcp    # HTTPS
sudo ufw allow 5432/tcp from 10.0.0.0/24  # PostgreSQL (internal only)
sudo ufw deny 5432/tcp    # Block external PostgreSQL access
```

### Database Security

```sql
-- Create restricted user
CREATE USER neuronapp WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE neurondb TO neuronapp;
GRANT USAGE ON SCHEMA neurondb TO neuronapp;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA neurondb TO neuronapp;

-- Enable SSL
ALTER SYSTEM SET ssl = 'on';
ALTER SYSTEM SET ssl_cert_file = '/etc/ssl/certs/server.crt';
ALTER SYSTEM SET ssl_key_file = '/etc/ssl/private/server.key';
```

### Application Security

- **API Keys:** Rotate regularly, use strong random keys
- **Rate Limiting:** Implement per-key and global limits
- **Input Validation:** Validate all user inputs
- **HTTPS Only:** Enforce TLS for all connections
- **Secrets Management:** Use vault systems (HashiCorp Vault, AWS Secrets Manager)

---

## 📊 Monitoring & Observability

### Prometheus + Grafana

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'neuronagent'
    static_configs:
      - targets: ['neuronagent:8080']
  
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
```

### Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| **Database Connections** | Active connections | > 80% of max_connections |
| **Query Latency** | 95th percentile query time | > 100ms |
| **API Response Time** | 95th percentile API latency | > 500ms |
| **Vector Search Latency** | HNSW search time | > 50ms |
| **CPU Usage** | Per-service CPU | > 80% |
| **Memory Usage** | Per-service memory | > 85% |
| **Disk I/O** | Database disk IOPS | > 90% capacity |

---

## 🆘 Troubleshooting

See comprehensive troubleshooting guides:
- **[NeuronDB Troubleshooting](../NeuronDB/docs/troubleshooting.md)**
- **[Docker Troubleshooting](docker.md#troubleshooting)**

---

## 📚 Additional Resources

- **[Docker Deployment Guide](docker.md)** - Detailed Docker documentation
- **[Component Documentation](../components/README.md)** - Component-specific guides
- **[Performance Tuning](../NeuronDB/docs/performance/)** - Optimization guides
- **[Security Policy](../SECURITY.md)** - Security best practices

---

**Last Updated:** 2025-01-30  
**Deployment Version:** 1.0.0
