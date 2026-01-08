# Kubernetes Helm Deployment Guide

Complete guide for deploying NeuronDB ecosystem on Kubernetes using Helm.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Upgrading](#upgrading)
- [Accessing Services](#accessing-services)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Kubernetes cluster (1.24+)
- Helm 3.8+
- kubectl configured to access your cluster
- Storage class configured (for persistent volumes)
- Ingress controller (optional, for external access)

## Features

This Helm chart provides a complete cloud-native deployment with:

- **StatefulSet** for NeuronDB with persistent storage
- **Deployments** for all services with configurable replicas
- **Horizontal Pod Autoscaling** for NeuronAgent
- **Pod Disruption Budgets** for high availability
- **Init Containers** for proper startup ordering
- **ServiceAccounts** for security
- **Network Policies** (optional) for network security
- **Health Checks** (liveness and readiness probes)
- **Resource Limits** and requests
- **ConfigMaps** for configuration management
- **Secrets** for sensitive data
- **Ingress** support with TLS
- **Complete Observability Stack** (Prometheus, Grafana, Jaeger)

### Verify Prerequisites

```bash
# Check Kubernetes version
kubectl version --client --short

# Check Helm version
helm version

# Check cluster access
kubectl cluster-info

# List available storage classes
kubectl get storageclass
```

## Quick Start

### 1. Create Namespace

```bash
kubectl create namespace neurondb
```

### 2. Set PostgreSQL Password

Create a secret with your PostgreSQL password:

```bash
# Generate a secure password
POSTGRES_PASSWORD=$(openssl rand -base64 32)

# Create secret
kubectl create secret generic neurondb-secrets \
  --from-literal=postgres-password="$POSTGRES_PASSWORD" \
  --namespace=neurondb
```

Or set it in values.yaml:

```yaml
secrets:
  create: true
  postgresPassword: "your-secure-password-here"
```

### 3. Install with Helm

```bash
# Install from local chart
helm install neurondb ./helm/neurondb \
  --namespace neurondb \
  --create-namespace \
  --set secrets.postgresPassword="$POSTGRES_PASSWORD"
```

### 4. Verify Installation

```bash
# Check all pods are running
kubectl get pods -n neurondb

# Check services
kubectl get svc -n neurondb

# Check persistent volumes
kubectl get pvc -n neurondb

# Run Helm test (if available)
helm test neurondb -n neurondb

# Validate chart
./scripts/validate-helm-chart.sh
```

## Installation

### Basic Installation

```bash
helm install neurondb ./helm/neurondb \
  --namespace neurondb \
  --create-namespace
```

### Installation with Custom Values

Create a custom values file:

```yaml
# my-values.yaml
neurondb:
  persistence:
    size: 100Gi
    storageClass: "fast-ssd"

neuronagent:
  replicas: 3
  autoscaling:
    maxReplicas: 20

monitoring:
  enabled: true
  grafana:
    adminPassword: "secure-password"
```

Install with custom values:

```bash
helm install neurondb ./helm/neurondb \
  --namespace neurondb \
  --create-namespace \
  --values my-values.yaml
```

### Installation with Ingress

```yaml
# ingress-values.yaml
ingress:
  enabled: true
  className: "nginx"
  hosts:
    - host: neurondb.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: neurondb-tls
      hosts:
        - neurondb.yourdomain.com
```

```bash
helm install neurondb ./helm/neurondb \
  --namespace neurondb \
  --create-namespace \
  --values ingress-values.yaml
```

## Configuration

### Key Configuration Options

#### NeuronDB (PostgreSQL)

```yaml
neurondb:
  enabled: true
  image:
    repository: ghcr.io/neurondb/neurondb-postgres
    tag: "2.0.0-pg17-cpu"
  
  postgresql:
    database: "neurondb"
    username: "neurondb"
    port: 5432
  
  persistence:
    enabled: true
    size: 50Gi
    storageClass: ""  # Uses default storage class if empty
  
  resources:
    requests:
      memory: "4Gi"
      cpu: "2"
    limits:
      memory: "8Gi"
      cpu: "4"
```

#### NeuronAgent

```yaml
neuronagent:
  enabled: true
  replicas: 2
  logLevel: "info"
  
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
```

#### NeuronMCP

```yaml
neuronmcp:
  enabled: true
  replicas: 1
  logLevel: "info"
```

#### NeuronDesktop

```yaml
neurondesktop:
  enabled: true
  api:
    replicas: 2
    database: "neurondesk"
    logLevel: "info"
  frontend:
    replicas: 2
```

#### Monitoring

```yaml
monitoring:
  enabled: true
  prometheus:
    enabled: true
    retention: "30d"
    persistence:
      enabled: true
      size: "20Gi"
  grafana:
    enabled: true
    adminPassword: "admin"  # Change in production!
    persistence:
      enabled: true
      size: "10Gi"
  jaeger:
    enabled: true
```

## Upgrading

### Upgrade to New Version

```bash
# Update values if needed
helm upgrade neurondb ./helm/neurondb \
  --namespace neurondb \
  --values my-values.yaml \
  --set neurondb.image.tag="2.0.0-pg17-cpu"
```

### Rolling Back

```bash
# List releases
helm list -n neurondb

# Rollback to previous version
helm rollback neurondb -n neurondb

# Rollback to specific revision
helm rollback neurondb 2 -n neurondb
```

## Accessing Services

### Port Forwarding

#### NeuronDesktop Frontend

```bash
kubectl port-forward -n neurondb \
  svc/neurondb-neurondesktop-frontend 3000:3000
```

Access at: http://localhost:3000

#### NeuronDesktop API

```bash
kubectl port-forward -n neurondb \
  svc/neurondb-neurondesktop-api 8081:8081
```

#### NeuronAgent

```bash
kubectl port-forward -n neurondb \
  svc/neurondb-neuronagent 8080:8080
```

#### Grafana

```bash
kubectl port-forward -n neurondb \
  svc/neurondb-grafana 3001:3000
```

Access at: http://localhost:3001 (admin/admin)

#### Prometheus

```bash
kubectl port-forward -n neurondb \
  svc/neurondb-prometheus 9090:9090
```

Access at: http://localhost:9090

#### Jaeger

```bash
kubectl port-forward -n neurondb \
  svc/neurondb-jaeger 16686:16686
```

Access at: http://localhost:16686

### Using Ingress

If ingress is enabled, access services via the configured host:

```bash
# Get ingress address
kubectl get ingress -n neurondb

# Access via hostname
curl https://neurondb.yourdomain.com
```

## Monitoring

### Prometheus Metrics

Services expose metrics at `/metrics`:

- NeuronAgent: `http://neurondb-neuronagent:8080/metrics`
- NeuronDesktop API: `http://neurondb-neurondesktop-api:8081/metrics`

### Grafana Dashboards

Grafana is pre-configured with:

- Prometheus datasource
- Default dashboard provisioning

Access Grafana and create custom dashboards for:
- Service health
- Request rates and latencies
- Database connection metrics
- Resource utilization

### Jaeger Tracing

Jaeger is available for distributed tracing:

- UI: Port 16686
- OTLP gRPC: Port 4317
- OTLP HTTP: Port 4318

## Troubleshooting

### Pods Not Starting

#### Check Pod Status

```bash
kubectl get pods -n neurondb
kubectl describe pod <pod-name> -n neurondb
kubectl logs <pod-name> -n neurondb
```

#### Common Issues

**Pending Pods (Storage Issues)**

```bash
# Check PVC status
kubectl get pvc -n neurondb

# Check storage class
kubectl get storageclass

# If PVC is pending, check events
kubectl describe pvc <pvc-name> -n neurondb
```

**CrashLoopBackOff**

```bash
# Check logs
kubectl logs <pod-name> -n neurondb --previous

# Common causes:
# - Database connection failures
# - Missing secrets
# - Resource limits too low
```

### Database Connection Issues

#### Verify Database is Ready

```bash
# Check NeuronDB pod
kubectl get pod -n neurondb -l app.kubernetes.io/component=neurondb

# Check logs
kubectl logs -n neurondb -l app.kubernetes.io/component=neurondb

# Test connection
kubectl exec -it -n neurondb \
  $(kubectl get pod -n neurondb -l app.kubernetes.io/component=neurondb -o jsonpath='{.items[0].metadata.name}') \
  -- psql -U neurondb -d neurondb -c "SELECT version();"
```

#### Verify Service Connectivity

```bash
# Check service endpoints
kubectl get endpoints -n neurondb

# Test from another pod
kubectl run -it --rm debug --image=postgres:17 --restart=Never -n neurondb -- \
  psql -h neurondb-neurondb -U neurondb -d neurondb
```

### Health Check Failures

#### Check Probe Configuration

```bash
# View pod spec
kubectl get pod <pod-name> -n neurondb -o yaml | grep -A 10 probes
```

#### Common Fixes

- Increase `initialDelaySeconds` if service takes time to start
- Verify health endpoint is accessible: `/health`
- Check resource limits aren't causing OOM kills

### Resource Issues

#### Check Resource Usage

```bash
# View resource requests/limits
kubectl describe pod <pod-name> -n neurondb | grep -A 5 "Limits\|Requests"

# Check node resources
kubectl top nodes
kubectl top pods -n neurondb
```

#### Adjust Resources

Update values.yaml and upgrade:

```yaml
neuronagent:
  resources:
    requests:
      memory: "1Gi"  # Increase if needed
      cpu: "1"
    limits:
      memory: "4Gi"
      cpu: "4"
```

### Uninstalling

```bash
# Uninstall release
helm uninstall neurondb -n neurondb

# Delete namespace (removes all resources)
kubectl delete namespace neurondb

# WARNING: This deletes all data in PVCs!
# Backup data before uninstalling if needed
```

### Backup and Restore

#### Manual Backup

```bash
# Backup database
kubectl exec -n neurondb \
  $(kubectl get pod -n neurondb -l app.kubernetes.io/component=neurondb -o jsonpath='{.items[0].metadata.name}') \
  -- pg_dump -U neurondb neurondb > backup.sql
```

#### Restore

```bash
# Copy backup to pod
kubectl cp backup.sql neurondb/<pod-name>:/tmp/backup.sql

# Restore
kubectl exec -n neurondb <pod-name> -- \
  psql -U neurondb -d neurondb < /tmp/backup.sql
```

## Advanced Configuration

### Custom PostgreSQL Configuration

```yaml
neurondb:
  postgresql:
    config: |
      shared_buffers = 256MB
      max_connections = 200
      shared_preload_libraries = 'neurondb'
```

### Service Account and RBAC

```yaml
serviceAccount:
  create: true
  name: "neurondb-sa"
  annotations:
    eks.amazonaws.com/role-arn: "arn:aws:iam::ACCOUNT:role/neurondb-role"
```

### Node Selectors and Affinity

Add to deployment templates or use values:

```yaml
neuronagent:
  nodeSelector:
    workload-type: "compute"
  affinity:
    podAntiAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchExpressions:
            - key: app.kubernetes.io/component
              operator: In
              values:
              - neuronagent
          topologyKey: kubernetes.io/hostname
```

## Production Recommendations

1. **Secrets Management**: Use external secret management (AWS Secrets Manager, HashiCorp Vault)
2. **Backup Strategy**: Implement automated backups using CronJob or external backup tools
3. **Monitoring**: Enable full observability stack and set up alerting
4. **Resource Limits**: Set appropriate requests/limits based on workload
5. **High Availability**: Use multiple replicas and pod anti-affinity
6. **Storage**: Use fast, reliable storage class for production
7. **Ingress**: Enable TLS/SSL for external access
8. **Network Policies**: Implement network policies for security

## Support

For issues and questions:

- GitHub Issues: https://github.com/neurondb/neurondb/issues
- Documentation: https://docs.neurondb.ai
- Community: https://discord.gg/neurondb

