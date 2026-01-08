# NeuronDB Production Deployment Documentation

Complete production deployment guide for NeuronDB on Kubernetes.

## Quick Links

- [Production Installation Guide](./production-install.md) - Complete production setup
- [Backup and Restore Guide](./backup-restore.md) - Backup/restore procedures
- [Upgrade and Rollback Guide](./upgrade-rollback.md) - Upgrade procedures
- [Sizing Guide](./sizing-guide.md) - Resource sizing recommendations

## Features

### Security
- ✅ Per-component RBAC with minimal permissions
- ✅ NetworkPolicies with default deny
- ✅ Pod Security Standards enforcement
- ✅ External Secrets Operator integration
- ✅ CSI Secrets Store support

### High Availability
- ✅ Zero-downtime upgrades with migration hooks
- ✅ StatefulSet rolling updates
- ✅ Pod Disruption Budgets
- ✅ PriorityClasses for critical components
- ✅ Health checks with SLO focus

### Observability
- ✅ ServiceMonitor for Prometheus Operator
- ✅ PrometheusRule alerts
- ✅ OpenTelemetry exporter config
- ✅ Structured logging

### Operations
- ✅ Automated backups (S3/GCS/Azure)
- ✅ Restore procedures
- ✅ Migration management
- ✅ External PostgreSQL support
- ✅ Advanced autoscaling (HPA/KEDA)

### GitOps
- ✅ Argo CD examples
- ✅ Flux examples
- ✅ Declarative configuration

## Quick Start

### Production Installation

```bash
# 1. Create namespace
kubectl create namespace neurondb

# 2. Create external PostgreSQL secret (if using external DB)
kubectl create secret generic neurondb-external-postgres-secret \
  --from-literal=host=postgres.example.com \
  --from-literal=port=5432 \
  --from-literal=database=neurondb \
  --from-literal=username=neurondb \
  --from-literal=password=<secure-password> \
  -n neurondb

# 3. Install with production values
helm install neurondb ./helm/neurondb \
  -f helm/neurondb/values-production-external-postgres.yaml \
  -n neurondb
```

### Enable Production Features

```bash
helm upgrade neurondb ./helm/neurondb \
  --set rbac.enabled=true \
  --set networkPolicy.enabled=true \
  --set backup.enabled=true \
  --set backup.s3.enabled=true \
  --set backup.s3.bucket=neurondb-backups \
  --set backup.s3.region=us-east-1 \
  --set observability.prometheusOperator.enabled=true \
  -n neurondb
```

## Example Values Files

- `values-minimal.yaml` - Minimal configuration for development
- `values-production-external-postgres.yaml` - Production with external PostgreSQL
- `values-observability-external.yaml` - With external observability stack
- `values-external-postgres.yaml` - External PostgreSQL example

## CI/CD Integration

All CI workflows are configured:
- Image signing (cosign)
- SBOM generation (Syft)
- SLSA provenance
- Trivy security scanning
- Helm lint and unittest
- Chart testing
- OCI registry publishing

## Support

For issues or questions:
- GitHub Issues: https://github.com/neurondb/neurondb2/issues
- Documentation: https://docs.neurondb.ai
- Email: support@neurondb.ai
