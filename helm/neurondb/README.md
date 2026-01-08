# NeuronDB Helm Chart

Complete cloud-native Helm chart for deploying the NeuronDB ecosystem on Kubernetes.

## Quick Start

```bash
# Create namespace
kubectl create namespace neurondb

# Install with default values
helm install neurondb ./helm/neurondb \
  --namespace neurondb \
  --create-namespace \
  --set secrets.postgresPassword="$(openssl rand -base64 32)"
```

## Components

This chart deploys:

- **NeuronDB**: PostgreSQL with NeuronDB extension (StatefulSet)
- **NeuronAgent**: AI agent service (Deployment with HPA)
- **NeuronMCP**: Model Context Protocol server (Deployment)
- **NeuronDesktop**: Web UI (API + Frontend Deployments)
- **Prometheus**: Metrics collection
- **Grafana**: Dashboards and visualization
- **Jaeger**: Distributed tracing

## Configuration

See `values.yaml` for all configuration options.

### Key Values

```yaml
# Database
neurondb:
  persistence:
    size: 50Gi
    storageClass: ""

# Services
neuronagent:
  replicas: 2
  autoscaling:
    enabled: true
    maxReplicas: 10

# Monitoring
monitoring:
  enabled: true
  prometheus:
    retention: "30d"
  grafana:
    adminPassword: "change-me"

# Ingress
ingress:
  enabled: true
  className: "nginx"
  hosts:
    - host: neurondb.example.com
```

## Requirements

- Kubernetes 1.24+
- Helm 3.8+
- Storage class for persistent volumes
- Ingress controller (optional)

## Installation

See [Docs/deployment/kubernetes-helm.md](../../Docs/deployment/kubernetes-helm.md) for complete installation guide.

## Upgrading

```bash
helm upgrade neurondb ./helm/neurondb \
  --namespace neurondb \
  --values my-values.yaml
```

## Uninstalling

```bash
helm uninstall neurondb -n neurondb
```

**Warning**: This will delete all data in persistent volumes!

## Support

- Documentation: [Docs/deployment/kubernetes-helm.md](../../Docs/deployment/kubernetes-helm.md)
- Issues: https://github.com/neurondb/neurondb/issues

