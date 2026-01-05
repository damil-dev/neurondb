# Version 2.0.0 Release

All components have been updated to version 2.0.0 for the main branch.

## Updated Components

### Core Extensions
- **NeuronDB**: `2.0` (PostgreSQL extension)
  - Control file updated
  - SQL schema files: `neurondb--2.0.sql`, `neurondb--2.0.sql.linux`, `neurondb--2.0.sql.macos`
  - Upgrade path: `neurondb--1.0--2.0.sql`

### Services
- **NeuronMCP**: `2.0.0` (Model Context Protocol server)
  - Package version updated
  - All Go modules updated
  
- **NeuronAgent**: `2.0.0` (Agent service)
  - OpenAPI specification updated

- **NeuronDesktop**: `2.0.0` (Web UI)
  - Frontend package updated

### Infrastructure
- **Docker Images**: All Dockerfiles updated to version 2.0
  - NeuronDB (CPU, CUDA, ROCm, Metal variants)
  - NeuronAgent
  - NeuronMCP

- **Docker Compose**: `docker-compose.prod.yml` updated with 2.0 tags

- **Helm Chart**: Version `2.0.0`
  - Chart version: `2.0.0`

## Upgrade Path

For existing installations running version 1.0, PostgreSQL will automatically use the upgrade script:

```sql
ALTER EXTENSION neurondb UPDATE TO '2.0';
```

The upgrade script (`neurondb--1.0--2.0.sql`) ensures compatibility between versions.

## Files Modified

### Configuration Files
- `NeuronDB/neurondb.control`
- `NeuronMCP/package.json`
- `NeuronDesktop/frontend/package.json`
- `NeuronAgent/openapi/openapi.yaml`
- `helm/neurondb/Chart.yaml`

### Docker Files
- `dockers/neurondb/Dockerfile.package`
- `dockers/neurondb/Dockerfile.package.cuda`
- `dockers/neurondb/Dockerfile.package.rocm`
- `dockers/neuronagent/Dockerfile.package`
- `dockers/neuronmcp/Dockerfile.package`
- `docker-compose.prod.yml`

### Source Code
- Multiple Go files in NeuronMCP and NeuronDesktop with version strings

### SQL Files
- Renamed: `neurondb--1.0.sql` → `neurondb--2.0.sql`
- Renamed: `neurondb--1.0.sql.linux` → `neurondb--2.0.sql.linux`
- Renamed: `neurondb--1.0.sql.macos` → `neurondb--2.0.sql.macos`
- Added: `neurondb--1.0--2.0.sql` (upgrade script)

## Verification

To verify the version:

```bash
# Check NeuronDB extension
psql -c "SELECT extversion FROM pg_extension WHERE extname = 'neurondb';"

# Check Docker images
docker images | grep neurondb

# Check NeuronMCP
neurondb-mcp --version
```

## Next Steps

1. Rebuild Docker images with new version tags
2. Update documentation references to version 2.0.0
3. Create release tags in git
4. Update package repositories

---

**Release Date**: January 5, 2026  
**Branch**: main  
**Commit**: 6dc4af7

