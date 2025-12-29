# NeuronDB Setup Script Documentation

## Overview

The `neurondb-setup.sh` script provides a professional, comprehensive, and detailed one-command setup for the entire NeuronDB ecosystem. It supports multiple installation modes, flexible component selection, and complete uninstall functionality.

## Features

### ✨ Professional & Detailed
- **Color-coded output** with clear status indicators
- **Detailed logging** to `/tmp/neurondb-setup-*.log`
- **Step-by-step progress** with verbose mode support
- **Comprehensive error handling** with helpful messages
- **Dependency resolution** - automatically includes required components

### 🎯 Flexible Component Selection
- Install all components with `--all`
- Install specific components with `--components`
- Mix and match any combination
- Dependencies automatically resolved

### 📦 Multiple Installation Modes
- **Docker** - Recommended for quick start and development
- **DEB packages** - Ubuntu/Debian systems
- **RPM packages** - RHEL/CentOS/Rocky/Fedora
- **macOS packages** - .pkg installer files

### 🗑️ Uninstall Support
- **Complete uninstall** with `--uninstall` flag
- **Selective removal** - uninstall specific components
- **Data removal** option with `--remove-data` (use with caution!)
- **Service cleanup** - stops services before removal
- **Docker cleanup** - removes containers and optionally volumes

## Quick Start

### Install All Components (Docker)
```bash
./scripts/neurondb-setup.sh --mode docker --all
```

### Install Specific Components
```bash
# Just NeuronDB and NeuronAgent
./scripts/neurondb-setup.sh --mode deb --components NeuronDB NeuronAgent

# NeuronDB and NeuronMCP
./scripts/neurondb-setup.sh --mode rpm --components NeuronDB NeuronMCP

# All except NeuronDesktop
./scripts/neurondb-setup.sh --mode docker --components NeuronDB NeuronAgent NeuronMCP
```

### Uninstall Components
```bash
# Uninstall all (Docker)
./scripts/neurondb-setup.sh --mode docker --all --uninstall

# Uninstall specific components
./scripts/neurondb-setup.sh --mode deb --components NeuronDB NeuronAgent --uninstall

# Uninstall with data removal (destructive!)
./scripts/neurondb-setup.sh --mode deb --components NeuronDB --uninstall --remove-data
```

## Usage Examples

### Example 1: Docker Setup (All Components)
```bash
./scripts/neurondb-setup.sh --mode docker --all
```
This will:
- Check Docker prerequisites
- Start all services via Docker Compose
- Verify each component
- Display comprehensive summary

### Example 2: DEB Package Installation (Selected Components)
```bash
./scripts/neurondb-setup.sh \
    --mode deb \
    --components NeuronDB NeuronAgent \
    --db-host localhost \
    --db-port 5432 \
    --db-name neurondb \
    --db-user postgres \
    --db-password mypassword
```
This will:
- Check prerequisites (PostgreSQL, dpkg)
- Install DEB packages for selected components
- Setup database schemas
- Start system services
- Verify installation

### Example 3: RPM Installation with Custom Database
```bash
./scripts/neurondb-setup.sh \
    --mode rpm \
    --components NeuronDB NeuronMCP NeuronDesktop \
    --db-host db.example.com \
    --db-password secret123
```

### Example 4: Uninstall (Docker)
```bash
# Keep data volumes
./scripts/neurondb-setup.sh --mode docker --all --uninstall

# Remove everything including data
./scripts/neurondb-setup.sh --mode docker --all --uninstall --remove-data
```

### Example 5: Uninstall (DEB Packages)
```bash
# Remove packages only
./scripts/neurondb-setup.sh --mode deb --components NeuronDB NeuronAgent --uninstall

# Remove packages and database schemas
./scripts/neurondb-setup.sh --mode deb --components NeuronDB --uninstall --remove-data
```

### Example 6: Dry Run (See What Would Happen)
```bash
./scripts/neurondb-setup.sh --mode docker --all --dry-run
./scripts/neurondb-setup.sh --mode deb --uninstall --all --dry-run
```

### Example 7: Verbose Output
```bash
./scripts/neurondb-setup.sh --mode docker --all --verbose
```

## Component Selection

### Available Components
- **NeuronDB** - PostgreSQL extension for vector search and ML
- **NeuronAgent** - REST API server for AI agent runtime
- **NeuronMCP** - MCP protocol server for desktop clients
- **NeuronDesktop** - Unified web interface

### Dependencies (Automatically Resolved)
- NeuronAgent requires NeuronDB
- NeuronMCP requires NeuronDB
- NeuronDesktop requires NeuronDB

When you select a component, its dependencies are automatically included.

## Command-Line Options

### Required Options
- `--mode MODE` - Installation mode: `docker`, `deb`, `rpm`, or `mac`
- Component selection (choose one):
  - `--all` - Install all components
  - `--components COMP1 [COMP2 ...]` - Install specific components

### Database Options
- `--db-host HOST` - Database host (default: `localhost`)
- `--db-port PORT` - Database port (default: `5432`, `5433` for docker)
- `--db-name NAME` - Database name (default: `neurondb`)
- `--db-user USER` - Database user (default: `postgres`, `neurondb` for docker)
- `--db-password PASS` - Database password (default: `neurondb`)

### Other Options
- `--skip-setup` - Skip database schema setup (packages only)
- `--skip-services` - Skip starting services (packages only)
- `--uninstall` - Uninstall selected components
- `--remove-data` - Remove data/schemas during uninstall (use with caution!)
- `--verbose, -v` - Enable verbose output
- `--dry-run` - Show what would be done without making changes
- `--help, -h` - Show help message

## Uninstall Behavior

### Docker Mode
When uninstalling with Docker mode:
1. Stops all running containers
2. Removes containers
3. Optionally removes volumes (with `--remove-data`)
4. Preserves Docker images (remove manually if needed)

### Package Modes (deb/rpm/mac)
When uninstalling with package modes:
1. Stops system services (systemd)
2. Removes packages
3. Optionally removes database schemas (with `--remove-data`)
4. Preserves configuration files (unless purged)

**Warning**: `--remove-data` flag will:
- Drop database extensions and schemas
- Remove Docker volumes (all data)
- This is **irreversible**!

## Output Format

The script provides professional, color-coded output:

- ✓ **Green** - Success messages
- ✗ **Red** - Error messages
- ⚠ **Yellow** - Warning messages
- ℹ **Cyan** - Information messages
- → **Blue** - Progress indicators

### Sections
1. **Header** - Script title and version
2. **Component Selection** - Shows selected components and dependencies
3. **Prerequisites Checking** - Validates required tools and dependencies
4. **Package Installation** / **Docker Setup** - Performs installation
5. **Database Schema Setup** - Configures database schemas
6. **Service Management** - Starts system services
7. **Verification** - Verifies each component is working
8. **Summary** - Final status table and next steps

For uninstall mode, the sections are:
1. **Header** - Script title and version
2. **Component Uninstallation** - Shows components to uninstall
3. **Stopping Services** - Stops running services
4. **Removing Packages** / **Removing Docker Services** - Removes components
5. **Removing Database Schemas** - (if `--remove-data` is used)
6. **Uninstallation Summary** - Final status table

## Logging

All operations are logged to `/tmp/neurondb-setup-YYYYMMDD-HHMMSS.log` with:
- Timestamps for each operation
- Success/failure status
- Detailed error messages
- Component status tracking

Enable verbose mode (`--verbose`) to see log file location and additional details.

## Exit Codes

- `0` - Success
- `1` - Error (missing prerequisites, installation failure, etc.)
- `2` - Invalid arguments

## Prerequisites by Mode

### Docker Mode
- Docker 20.10+
- Docker Compose 2.0+
- Docker daemon running

### Package Modes (deb/rpm/mac)
- PostgreSQL 16, 17, or 18 (client tools)
- Package manager (dpkg/rpm/pkgutil)
- sudo (usually required)
- Network access to database

### Component-Specific
- **NeuronDB**: PostgreSQL with development headers (for source builds)
- **NeuronAgent**: Go 1.23+ (for source builds)
- **NeuronMCP**: Go 1.23+ (for source builds)
- **NeuronDesktop**: Node.js/npm (for frontend builds)

Note: Package installations include pre-built binaries, so build tools aren't required.

## Testing on Debian

Since you're on Debian, you can test with DEB packages:

```bash
# First, build the DEB packages (if not already built)
cd packaging/deb
./build-all-deb.sh

# Then install using the script
cd ../..
./scripts/neurondb-setup.sh --mode deb --all

# Test uninstall
./scripts/neurondb-setup.sh --mode deb --all --uninstall --dry-run
```

## Troubleshooting

### Common Issues

**Docker daemon not running**
```bash
# Linux
sudo systemctl start docker

# macOS/Windows
# Start Docker Desktop application
```

**Database connection failed**
- Verify PostgreSQL is running
- Check connection parameters (host, port, user, password)
- Ensure user has necessary privileges
- Check firewall rules

**Package not found**
- Build packages first: `cd packaging/{mode} && ./build-all-{mode}.sh`
- Check package directory exists
- Verify you're in the correct repository root

**Permission denied**
- Package installations typically require sudo
- Ensure you have appropriate permissions

**Uninstall fails**
- Ensure services are stopped first
- Check if packages are actually installed
- For Docker: ensure containers exist
- Use `--dry-run` to see what would happen

### Getting Help

1. **Check logs**: View `/tmp/neurondb-setup-*.log` for detailed information
2. **Enable verbose mode**: Add `--verbose` flag
3. **Dry run**: Use `--dry-run` to see what would happen
4. **Read documentation**: See README.md and QUICKSTART.md

## Advanced Usage

### Skip Setup Steps
```bash
# Install packages but skip database setup
./scripts/neurondb-setup.sh --mode deb --components NeuronDB --skip-setup

# Install but don't start services
./scripts/neurondb-setup.sh --mode rpm --components NeuronDB --skip-services
```

### Custom Configuration
```bash
# Use environment variables (some components support this)
export DB_HOST=prod-db.example.com
export DB_PASSWORD=super-secret
./scripts/neurondb-setup.sh --mode docker --all
```

### Partial Uninstall
```bash
# Uninstall only NeuronAgent, keep NeuronDB
./scripts/neurondb-setup.sh --mode deb --components NeuronAgent --uninstall

# Uninstall Docker services but keep packages
# (Use Docker commands directly: docker compose down)
```

## Integration with Other Scripts

This script integrates with:
- `scripts/verify_neurondb_integration.sh` - Comprehensive verification tests
- Component-specific setup scripts in each component directory
- Docker Compose configurations in repository root
- Package build scripts in `packaging/` directory

## Best Practices

1. **Start with Docker mode** for quick evaluation
2. **Use dry-run first** to understand what will happen
3. **Check prerequisites** before running (script does this automatically)
4. **Review logs** if issues occur
5. **Use package modes** for production deployments
6. **Install dependencies first** if building from source
7. **Backup data** before using `--remove-data` flag
8. **Test uninstall** with `--dry-run` first

## Related Documentation

- [README.md](../README.md) - Complete project documentation
- [QUICKSTART.md](../QUICKSTART.md) - Quick start guide
- [NeuronDB/INSTALL.md](../NeuronDB/INSTALL.md) - Detailed installation guide
- [Component-specific READMEs](../) - Individual component documentation
