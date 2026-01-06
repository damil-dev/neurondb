# Upgrade and Rollback Guide

Complete procedures for upgrading and rolling back NeuronDB.

## Pre-Upgrade Checklist

- [ ] Review release notes for breaking changes
- [ ] Backup database (see [Backup Guide](./backup-restore.md))
- [ ] Verify current version: `helm list -n neurondb`
- [ ] Check migration status: `kubectl get jobs -n neurondb | grep migration`
- [ ] Ensure sufficient cluster resources
- [ ] Review values.yaml for deprecated options

## Upgrade Procedure

### Step 1: Backup

```bash
# Ensure backup is recent
kubectl get cronjob neurondb-backup -n neurondb
kubectl create job --from=cronjob/neurondb-backup manual-backup-$(date +%s) -n neurondb
```

### Step 2: Review Changes

```bash
# Dry-run upgrade
helm upgrade neurondb ./helm/neurondb \
  --version <new-version> \
  --dry-run --debug \
  -f values-production-external-postgres.yaml \
  -n neurondb
```

### Step 3: Run Migrations

Migrations run automatically via pre-upgrade hooks. Monitor:

```bash
# Watch migration job
kubectl get jobs -n neurondb -w | grep migration
kubectl logs -f job/neurondb-migration-<revision> -n neurondb
```

### Step 4: Upgrade Chart

```bash
helm upgrade neurondb ./helm/neurondb \
  --version <new-version> \
  -f values-production-external-postgres.yaml \
  -n neurondb
```

### Step 5: Monitor Upgrade

```bash
# Watch pods
kubectl get pods -n neurondb -w

# Check rollout status
kubectl rollout status statefulset/neurondb-neurondb -n neurondb
kubectl rollout status deployment/neurondb-neuronagent -n neurondb

# Check logs
kubectl logs -f deployment/neurondb-neuronagent -n neurondb
```

### Step 6: Verify Upgrade

```bash
# Check versions
helm list -n neurondb
kubectl get pods -n neurondb -o jsonpath='{.items[*].spec.containers[*].image}'

# Test functionality
kubectl port-forward svc/neurondb-neuronagent 8080:8080 -n neurondb
curl http://localhost:8080/health
```

## Rollback Procedure

### Step 1: Identify Previous Revision

```bash
# List release history
helm history neurondb -n neurondb

# Note the revision number to rollback to
```

### Step 2: Rollback Helm Release

```bash
# Rollback to previous revision
helm rollback neurondb <revision> -n neurondb

# Or rollback to previous version
helm rollback neurondb -n neurondb
```

### Step 3: Database Rollback (if needed)

If database migrations were applied, rollback them:

```bash
# Connect to database
kubectl exec -it statefulset/neurondb-neurondb -n neurondb -- \
  psql -U neurondb -d neurondb

# Run rollback script (if provided)
\i /path/to/rollback.sql
```

### Step 4: Verify Rollback

```bash
# Check pod versions
kubectl get pods -n neurondb -o jsonpath='{.items[*].spec.containers[*].image}'

# Verify functionality
kubectl port-forward svc/neurondb-neuronagent 8080:8080 -n neurondb
curl http://localhost:8080/health
```

## Zero-Downtime Upgrades

### StatefulSet Rolling Update

Configure update strategy:

```yaml
neurondb:
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      partition: null  # Update all pods
      maxUnavailable: 1
```

### Canary Deployment

Gradual rollout using partition:

```yaml
neurondb:
  replicas: 3
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      partition: 2  # Only update pod 2 and 3
```

After verification, reduce partition:
```bash
helm upgrade neurondb ./helm/neurondb \
  --set neurondb.updateStrategy.rollingUpdate.partition=1 \
  -n neurondb
```

## Migration Management

### Check Migration Status

```bash
# View migration history
kubectl get jobs -n neurondb | grep migration

# Check database migration version
kubectl exec -it statefulset/neurondb-neurondb -n neurondb -- \
  psql -U neurondb -d neurondb -c \
  "SELECT version, name, applied_at FROM neurondb_agent.schema_migrations ORDER BY version DESC;"
```

### Manual Migration

If automatic migration fails:

```bash
# Create migration job manually
kubectl create job --from=cronjob/neurondb-migration manual-migration -n neurondb

# Or run migration script directly
kubectl exec -it statefulset/neurondb-neurondb -n neurondb -- \
  psql -U neurondb -d neurondb -f /path/to/migration.sql
```

## Troubleshooting

### Upgrade Fails

1. Check migration job:
```bash
kubectl describe job neurondb-migration-<revision> -n neurondb
kubectl logs job/neurondb-migration-<revision> -n neurondb
```

2. Check pod status:
```bash
kubectl describe pod <pod-name> -n neurondb
kubectl logs <pod-name> -n neurondb
```

3. Rollback if needed:
```bash
helm rollback neurondb -n neurondb
```

### Migration Fails

1. Check migration logs:
```bash
kubectl logs job/neurondb-migration-<revision> -n neurondb
```

2. Fix migration issues and retry:
```bash
# Delete failed job
kubectl delete job neurondb-migration-<revision> -n neurondb

# Retry upgrade
helm upgrade neurondb ./helm/neurondb --version <version> -n neurondb
```

### Rollback Fails

1. Check release history:
```bash
helm history neurondb -n neurondb
```

2. Manual rollback:
```bash
# Delete current resources
helm uninstall neurondb -n neurondb

# Reinstall previous version
helm install neurondb ./helm/neurondb \
  --version <previous-version> \
  -f values-production-external-postgres.yaml \
  -n neurondb
```

## Best Practices

1. **Always backup before upgrade**
2. **Test upgrades in staging first**
3. **Review breaking changes in release notes**
4. **Monitor during and after upgrade**
5. **Keep rollback plan ready**
6. **Document custom migrations**
7. **Use canary deployments for major upgrades**

