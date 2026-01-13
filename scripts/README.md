# NeuronDB Scripts

**Professional, self-sufficient automation scripts for the NeuronDB ecosystem**

This directory contains comprehensive mega-scripts for managing, deploying, monitoring, and maintaining all NeuronDB components. All scripts are **completely self-sufficient** with no external dependencies.

---

## 📋 Quick Reference

| Script | Purpose | Quick Example |
|--------|---------|---------------|
| **neurondb-docker.sh** | All Docker operations | `./neurondb-docker.sh run --component neurondb` |
| **neurondb-database.sh** | Database backup/restore/setup | `./neurondb-database.sh backup --format custom` |
| **neurondb-setup.sh** | Installation and setup | `./neurondb-setup.sh install --mode docker` |
| **neurondb-healthcheck.sh** | Health checks and testing | `./neurondb-healthcheck.sh quick` |
| **neurondb-monitor.sh** | Monitoring and logs | `./neurondb-monitor.sh status` |
| **neurondb-workflows.sh** | Release and git operations | `./neurondb-workflows.sh release --version 2.0.0` |
| **neurondb-pkgs.sh** | Package management | `./neurondb-pkgs.sh verify --os ubuntu` |
| **neurondb-blogs.sh** | Blog maintenance | `./neurondb-blogs.sh fix-markdown` |
| **neurondb-cleanup.sh** | Cleanup operations | `./neurondb-cleanup.sh --all --dry-run` |

---

## 🎯 Script Architecture

All scripts follow a **unified command structure**:

```bash
./neurondb-<category>.sh COMMAND [OPTIONS]
```

**Common Options (available in all scripts):**
- `-h, --help` - Show detailed help message
- `-v, --verbose` - Enable verbose output
- `-V, --version` - Show version information
- `--dry-run` - Preview changes without applying (where applicable)

---

## 🐳 Docker Management: `neurondb-docker.sh`

**Purpose:** Complete Docker container management for all NeuronDB components. Handles building, running, testing, verifying, and monitoring Docker containers.

### Commands

#### `run` - Run Docker Containers
Build, clean, or start Docker containers for NeuronDB components.

```bash
# Start NeuronDB CPU variant
./neurondb-docker.sh run --component neurondb --variant cpu

# Start NeuronDB with CUDA support
./neurondb-docker.sh run --component neurondb --variant cuda

# Build NeuronDB image
./neurondb-docker.sh run --component neurondb --variant cpu --action build

# Clean NeuronDB containers and volumes
./neurondb-docker.sh run --component neurondb --variant cpu --action clean

# Start NeuronAgent
./neurondb-docker.sh run --component neuronagent

# Start NeuronMCP
./neurondb-docker.sh run --component neuronmcp
```

**Options:**
- `--component COMPONENT` - Component name: `neurondb`, `neuronagent`, `neuronmcp` (required)
- `--variant VARIANT` - Variant for neurondb: `cpu`, `cuda`, `rocm`, `metal` (default: `cpu`)
- `--action ACTION` - Action to perform: `build`, `clean`, `run` (default: `run`)

**What it does:**
- Builds Docker images with proper build arguments
- Manages Docker Compose profiles
- Handles environment variables for database connections
- Provides connection information after startup

---

#### `test` - Run Docker Tests
Comprehensive testing suite for Docker deployments.

```bash
# Basic connectivity tests
./neurondb-docker.sh test --type basic --variant cpu

# Integration tests between components
./neurondb-docker.sh test --type integration

# Comprehensive test suite
./neurondb-docker.sh test --type comprehensive --variant cuda

# Detailed test suite
./neurondb-docker.sh test --type detailed --variant cpu --quick

# Deep test suite (REL1_STABLE)
./neurondb-docker.sh test --type deep --variant cpu --stop-on-fail
```

**Options:**
- `--type TYPE` - Test type: `basic`, `integration`, `comprehensive`, `detailed`, `deep` (required)
- `--variant VARIANT` - Container variant: `cpu`, `cuda`, `rocm`, `metal`, `all` (default: `cpu`)
- `--component COMPONENT` - Component to test: `neurondb`, `neuronagent`, `neuronmcp`, `all` (default: `neurondb`)
- `--stop-on-fail` - Stop testing on first failure
- `--quick` - Run in quick mode (reduced test set)

**Test Types:**
- **basic** - Container connectivity, PostgreSQL connection, extension installation
- **integration** - Component integration (NeuronDB ↔ NeuronAgent ↔ NeuronMCP)
- **comprehensive** - Full test suite covering all features
- **detailed** - Detailed testing with verbose output
- **deep** - Deep testing for stable releases

---

#### `verify` - Verify Docker Setup
Verify Docker dependencies and ecosystem health.

```bash
# Verify Docker dependencies
./neurondb-docker.sh verify --dependencies

# Verify entire Docker ecosystem
./neurondb-docker.sh verify --ecosystem

# Verify both
./neurondb-docker.sh verify --dependencies --ecosystem
```

**Options:**
- `--dependencies` - Verify Docker and Docker Compose installation
- `--ecosystem` - Verify all running containers and services

**What it checks:**
- Docker installation and daemon status
- Docker Compose availability
- Container running status
- Service health

---

#### `logs` - View Container Logs
View and follow logs from Docker containers.

```bash
# View last 50 lines of NeuronDB logs
./neurondb-docker.sh logs --component neurondb

# Follow NeuronAgent logs in real-time
./neurondb-docker.sh logs --component neuronagent --follow

# View last 100 lines
./neurondb-docker.sh logs --component neurondb --lines 100
```

**Options:**
- `--component COMPONENT` - Component name: `neurondb`, `neuronagent`, `neuronmcp` (required)
- `--follow, -f` - Follow log output (like `tail -f`)
- `--lines N` - Number of lines to show (default: 50)

---

#### `status` - Show Container Status
Display status of all Docker containers.

```bash
./neurondb-docker.sh status
```

**Output:**
- Lists all NeuronDB-related containers
- Shows running/stopped status
- Displays container status details
- Shows Docker Compose service status

---

#### `build` - Build Docker Images
Build Docker images for components.

```bash
# Build NeuronDB CPU image
./neurondb-docker.sh build --component neurondb --variant cpu

# Build NeuronAgent
./neurondb-docker.sh build --component neuronagent
```

---

#### `clean` - Clean Docker Resources
Clean up Docker containers and volumes.

```bash
# Clean NeuronDB containers
./neurondb-docker.sh clean --component neurondb --variant cpu
```

**⚠️ Warning:** This will remove containers and volumes, resulting in data loss!

---

#### `exec` - Execute Command in Container
Execute commands inside running containers.

```bash
# Execute SQL query in NeuronDB
./neurondb-docker.sh exec --component neurondb --command "psql -U neurondb -d neurondb -c 'SELECT version();'"

# Check NeuronAgent status
./neurondb-docker.sh exec --component neuronagent --command "curl localhost:8080/health"
```

**Options:**
- `--component COMPONENT` - Component name (required)
- `--command COMMAND` - Command to execute (required)

---

## 💾 Database Management: `neurondb-database.sh`

**Purpose:** Complete database operations including backup, restore, setup, maintenance, and verification.

### Commands

#### `backup` - Create Database Backup
Create backups in multiple formats with retention policies.

```bash
# Custom format backup (recommended - compressed, supports parallel restore)
./neurondb-database.sh backup --format custom

# SQL format with compression
./neurondb-database.sh backup --format sql --compress

# Directory format for large databases (parallel dump)
./neurondb-database.sh backup --format directory --output /backups/neurondb

# Custom retention policy (keep backups for 7 days)
./neurondb-database.sh backup --format custom --retention 7
```

**Options:**
- `--format FORMAT` - Backup format: `sql`, `custom`, `directory` (default: `custom`)
- `--output DIR` - Output directory (default: `./backups`)
- `--compress` - Compress SQL backups
- `--retention DAYS` - Keep backups for N days (default: 30)

**Backup Formats:**
- **sql** - Plain SQL dump (text format, portable, can be compressed)
- **custom** - PostgreSQL custom format (compressed, supports parallel restore, recommended)
- **directory** - Directory format (parallel dump/restore, best for large databases)

**Database Configuration:**
Set via environment variables or command-line options:
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=neurondb
export DB_USER=postgres
export DB_PASSWORD=your_password

# Or use command-line options
./neurondb-database.sh backup --host localhost --port 5432 --database neurondb --user postgres --password secret
```

**What it does:**
- Creates timestamped backups
- Automatically cleans up old backups based on retention policy
- Provides detailed backup information (size, location, restore instructions)
- Supports multiple backup formats for different use cases

---

#### `restore` - Restore Database from Backup
Restore databases from any backup format.

```bash
# Restore from custom format (auto-detected)
./neurondb-database.sh restore --backup neurondb_backup_20250101_120000.dump

# Restore from SQL backup
./neurondb-database.sh restore --backup neurondb_backup_20250101_120000.sql

# Restore from directory format with 8 parallel jobs
./neurondb-database.sh restore --backup neurondb_backup_20250101_120000_dir --jobs 8

# Drop database before restore (⚠️ destructive!)
./neurondb-database.sh restore --backup backup.dump --drop

# Clean database objects before restore
./neurondb-database.sh restore --backup backup.dump --clean
```

**Options:**
- `--backup PATH` - Backup file or directory to restore (required)
- `--format FORMAT` - Force backup format (auto-detected if not specified)
- `--drop` - Drop existing database before restore (requires confirmation)
- `--clean` - Clean (drop) database objects before recreating
- `--jobs N` - Number of parallel jobs for directory format (default: 4)

**What it does:**
- Auto-detects backup format (SQL, custom, or directory)
- Handles compressed backups automatically
- Supports parallel restore for directory format
- Verifies restore completion
- Checks for NeuronDB extension after restore

---

#### `setup` - Setup Database and Extensions
Initialize database and install required extensions.

```bash
# Setup database with extensions
./neurondb-database.sh setup

# With custom database configuration
./neurondb-database.sh setup --host localhost --port 5432 --database neurondb --user postgres --password secret
```

**What it does:**
- Creates database if it doesn't exist
- Installs `vector` extension
- Installs `neurondb` extension
- Verifies installation
- Displays database information

---

#### `status` - Check Database Status
Display database status and information.

```bash
# Check database status
./neurondb-database.sh status

# With custom connection
./neurondb-database.sh status --host localhost --port 5432 --database neurondb
```

**Output includes:**
- Database name, host, port
- Database size
- Table count
- Extension count and versions
- Connection information
- Installed extensions list

---

#### `vacuum` - Run VACUUM and ANALYZE
Perform database maintenance operations.

```bash
# Run VACUUM ANALYZE
./neurondb-database.sh vacuum
```

**What it does:**
- Runs `VACUUM ANALYZE` on the database
- Updates table statistics
- Reclaims storage space
- Improves query performance

---

#### `list-backups` - List Available Backups
List all available backups in the backup directory.

```bash
# List all backups
./neurondb-database.sh list-backups

# With custom backup directory
./neurondb-database.sh list-backups --output /backups/neurondb
```

**Output:**
- SQL backups with sizes
- Custom format backups with sizes
- Directory format backups with sizes

---

#### `verify` - Verify Database Integrity
Verify database integrity and extensions.

```bash
# Verify database
./neurondb-database.sh verify
```

**What it checks:**
- Database size
- NeuronDB extension installation and version
- Vector extension installation
- Table count
- Overall database health

---

## 🚀 Setup & Installation: `neurondb-setup.sh`

**Purpose:** Complete setup and installation of the NeuronDB ecosystem in various modes.

### Commands

#### `install` - Install NeuronDB Ecosystem
Install NeuronDB components using different methods.

```bash
# Install with Docker (recommended)
./neurondb-setup.sh install --mode docker

# Install specific components
./neurondb-setup.sh install --mode docker --components neurondb,neuronagent

# Install with DEB packages
./neurondb-setup.sh install --mode deb

# Install with RPM packages
./neurondb-setup.sh install --mode rpm
```

**Options:**
- `--mode MODE` - Installation mode: `docker`, `deb`, `rpm` (default: `docker`)
- `--components LIST` - Comma-separated list: `neurondb`, `neuronagent`, `neuronmcp`, `neurondesktop` (default: `all`)

**Installation Modes:**
- **docker** - Uses Docker Compose to start all services
- **deb** - Installs DEB packages (Ubuntu/Debian)
- **rpm** - Installs RPM packages (RHEL/CentOS/Rocky)

**What it does:**
- Checks prerequisites
- Installs components based on mode
- Configures services
- Starts services
- Verifies installation

---

#### `vagrant-deps` - Install Vagrant Dependencies
Install VirtualBox and Vagrant for package testing.

```bash
# Install Vagrant dependencies
./neurondb-setup.sh vagrant-deps
```

**What it does:**
- Detects operating system
- Installs VirtualBox
- Installs Vagrant
- Verifies installation

**Supported Platforms:**
- Ubuntu/Debian (apt-get)
- RHEL/CentOS/Rocky (yum/dnf)
- macOS (Homebrew)

---

#### `ecosystem` - Setup Complete Ecosystem
One-command setup for the complete NeuronDB ecosystem.

```bash
# Setup complete ecosystem
./neurondb-setup.sh ecosystem
```

**What it does:**
- Checks all prerequisites
- Installs all components
- Configures services
- Starts all services
- Verifies complete setup

---

#### `verify` - Verify Installation
Verify that installation completed successfully.

```bash
# Verify installation
./neurondb-setup.sh verify
```

**What it checks:**
- Docker installation and daemon
- Container running status
- Service health

---

#### `generate-passwords` - Generate Secure Passwords
Generate secure passwords for deployment configuration.

```bash
# Generate secure passwords
./neurondb-setup.sh generate-passwords > .env.secure

# Review and copy to .env file
cat .env.secure
```

**What it generates:**
- Secure PostgreSQL password (base64 encoded, 32 bytes)
- Environment variable template
- Instructions for use

**Output format:**
```bash
# Secure passwords generated on 2025-01-01 12:00:00
# Copy these values to your .env file

# PostgreSQL / NeuronDB
POSTGRES_USER=neurondb
POSTGRES_PASSWORD=<secure-random-password>
POSTGRES_DB=neurondb

# NeuronAgent (must match POSTGRES_PASSWORD)
DB_PASSWORD=${POSTGRES_PASSWORD}

# NeuronMCP (must match POSTGRES_PASSWORD)
NEURONDB_PASSWORD=${POSTGRES_PASSWORD}

# Generate NeuronAgent API key (if needed)
# Use: openssl rand -hex 32
```

---

## ✅ Health Checking: `neurondb-healthcheck.sh`

**Purpose:** Health checks and integration testing for all NeuronDB components.

### Commands

#### `quick` - Quick Health Check
Fast health check (30 seconds).

```bash
# Quick health check
./neurondb-healthcheck.sh quick

# With custom configuration
./neurondb-healthcheck.sh quick --host localhost --port 5432 --database neurondb
```

**What it checks:**
- Database connection
- NeuronAgent health endpoint

**Output:**
```
✓ Database connection
✓ NeuronAgent health
All checks passed!
```

---

#### `health` - Full Health Check
Comprehensive health check of all components.

```bash
# Full health check
./neurondb-healthcheck.sh health

# With custom agent URL
./neurondb-healthcheck.sh health --agent-url http://localhost:8080
```

**What it checks:**
- NeuronDB (PostgreSQL) connection
- NeuronDB extension installation and version
- NeuronAgent REST API health
- Component integration

**Options:**
- `--host HOST` - Database host (default: `localhost`)
- `--port PORT` - Database port (default: `5432`)
- `--database NAME` - Database name (default: `neurondb`)
- `--user USER` - Database user (default: `neurondb`)
- `--password PASS` - Database password (or use `DB_PASSWORD` env var)
- `--agent-url URL` - NeuronAgent URL (default: `http://localhost:8080`)

---

#### `integration` - Integration Tests
Run integration tests between components.

```bash
# Run integration tests
./neurondb-healthcheck.sh integration
```

**What it tests:**
- NeuronDB → NeuronAgent integration
- Component communication
- End-to-end workflows

---

#### `smoke` - Smoke Tests
Quick smoke tests for basic functionality.

```bash
# Run smoke tests
./neurondb-healthcheck.sh smoke
```

**What it tests:**
- NeuronDB SQL queries
- NeuronAgent REST API
- Basic component functionality

---

## 📊 Monitoring: `neurondb-monitor.sh`

**Purpose:** Real-time monitoring, log viewing, and metrics collection for all components.

### Commands

#### `status` - Show Component Status
Display status of all NeuronDB components.

```bash
# Show status
./neurondb-monitor.sh status

# Specify deployment mode
./neurondb-monitor.sh status --mode docker
```

**Options:**
- `--mode [docker|native]` - Deployment mode (auto-detected if not specified)

**Output:**
- Component running status
- Container/service status
- Deployment mode detection

---

#### `logs` - View Container Logs
View and follow logs from components.

```bash
# View NeuronDB logs
./neurondb-monitor.sh logs --component neurondb

# Follow NeuronAgent logs
./neurondb-monitor.sh logs --component neuronagent --follow

# View last 100 lines
./neurondb-monitor.sh logs --component neurondb --lines 100
```

**Options:**
- `--component COMPONENT` - Component name: `neurondb`, `neuronagent`, `neuronmcp` (required)
- `--follow, -f` - Follow log output in real-time
- `--lines N` - Number of lines to show (default: 50)

---

#### `watch` - Watch Status Continuously
Continuously monitor component status.

```bash
# Watch status (updates every 5 seconds)
./neurondb-monitor.sh watch

# Custom refresh interval
./neurondb-monitor.sh watch --interval 3
```

**Options:**
- `--interval SECONDS` - Refresh interval in seconds (default: 5)

**What it does:**
- Continuously displays component status
- Updates at specified interval
- Clears screen between updates
- Press Ctrl+C to exit

---

#### `metrics` - Show Metrics and Statistics
Display component metrics and resource usage.

```bash
# Show metrics
./neurondb-monitor.sh metrics
```

**Output:**
- Container CPU usage
- Memory usage
- Network I/O
- Resource statistics

---

## 🔄 Workflows: `neurondb-workflows.sh`

**Purpose:** Release management, version synchronization, and git operations.

### Commands

#### `release` - Create a New Release
Create a new release with version updates.

```bash
# Create release
./neurondb-workflows.sh release --version 2.0.0

# Dry run to preview changes
./neurondb-workflows.sh release --version 2.0.0 --dry-run
```

**Options:**
- `--version VERSION` - Version number in semver format (required, e.g., `1.0.0`, `2.0.0-beta`)
- `--dry-run` - Preview changes without applying

**What it does:**
- Updates version in all component files:
  - `env.example`
  - `package.json`
  - `NeuronDB/neurondb.control`
  - `NeuronAgent/go.mod`
  - `NeuronDesktop/frontend/package.json`
  - `NeuronMCP/go.mod`
- Validates version format (semver)
- Provides detailed output of changes

---

#### `sync` - Sync Version Branches
Synchronize version branches (e.g., REL1_STABLE and main).

```bash
# Sync branches (main -> REL1_STABLE, 2.0.0 -> 1.0.0)
./neurondb-workflows.sh sync

# Custom branch sync
./neurondb-workflows.sh sync --from-branch main --to-branch REL1_STABLE --from-version 2.0.0 --to-version 1.0.0

# Dry run
./neurondb-workflows.sh sync --dry-run
```

**Options:**
- `--from-branch BRANCH` - Source branch (default: `main`)
- `--to-branch BRANCH` - Target branch (default: `REL1_STABLE`)
- `--from-version VER` - Source version (default: `2.0.0`)
- `--to-version VER` - Target version (default: `1.0.0`)
- `--dry-run` - Preview changes without applying

**What it does:**
- Fetches latest changes
- Merges source branch into target branch
- Replaces version numbers in all files
- Updates Helm charts, Docker files, package files
- Handles merge conflicts gracefully

---

#### `update-refs` - Update Markdown References
Update references to renamed markdown files.

```bash
# Update references after file renames
./neurondb-workflows.sh update-refs

# Dry run
./neurondb-workflows.sh update-refs --dry-run
```

**What it does:**
- Detects renamed files from git status
- Updates references in all file types:
  - Markdown files (`.md`)
  - Text files (`.txt`)
  - Scripts (`.sh`)
  - Go files (`.go`)
  - Python files (`.py`)
  - TypeScript/JavaScript files (`.ts`, `.tsx`, `.js`)
  - JSON files (`.json`)

---

#### `pull` - Safe Git Pull
Perform safe git pull with rebase (avoids merge commits).

```bash
# Safe git pull
./neurondb-workflows.sh pull
```

**What it does:**
- Checks for uncommitted changes
- Stashes changes if needed
- Fetches latest changes
- Rebases on remote branch (maintains linear history)
- Restores stashed changes
- Handles conflicts gracefully

**Benefits:**
- Maintains linear git history
- Avoids unnecessary merge commits
- Preserves uncommitted work
- Safe and reversible

---

## 📦 Package Management: `neurondb-pkgs.sh`

**Purpose:** Package verification, SDK generation, and package testing.

### Commands

#### `verify` - Verify DEB/RPM Packages
Verify package files and integrity.

```bash
# Verify packages for Ubuntu
./neurondb-pkgs.sh verify --os ubuntu

# Verify packages for Rocky Linux
./neurondb-pkgs.sh verify --os rocky

# Verify specific package
./neurondb-pkgs.sh verify --os ubuntu --package neurondb_2.0.0_amd64.deb
```

**Options:**
- `--os OS` - OS type: `ubuntu`, `rocky` (default: `ubuntu`)
- `--package PATH` - Specific package file path

**What it checks:**
- Package file existence
- Package format validity
- Package metadata

---

#### `generate-sdk` - Generate Client SDKs
Generate client SDKs from OpenAPI specification.

```bash
# Generate Python SDK
./neurondb-pkgs.sh generate-sdk --language python

# Generate JavaScript SDK
./neurondb-pkgs.sh generate-sdk --language javascript

# Generate Go SDK
./neurondb-pkgs.sh generate-sdk --language go --output ./sdks

# Generate Java SDK
./neurondb-pkgs.sh generate-sdk --language java
```

**Options:**
- `--language LANG` - Target language: `python`, `javascript`, `go`, `java` (default: `python`)
- `--output DIR` - Output directory (default: `./sdks`)

**Prerequisites:**
- OpenAPI Generator CLI: `npm install -g @openapitools/openapi-generator-cli`

**What it does:**
- Finds OpenAPI specification file
- Generates SDK in specified language
- Creates output directory structure
- Provides SDK location

---

#### `test-vagrant` - Test Packages in Vagrant VM
Test packages in a Vagrant virtual machine.

```bash
# Test packages in Ubuntu VM
./neurondb-pkgs.sh test-vagrant --os ubuntu

# Test packages in Rocky Linux VM
./neurondb-pkgs.sh test-vagrant --os rocky

# Destroy VM after testing
./neurondb-pkgs.sh test-vagrant --os ubuntu --destroy-vm
```

**Options:**
- `--os OS` - OS type: `ubuntu`, `rocky` (default: `ubuntu`)
- `--destroy-vm` - Destroy VM after testing

**Prerequisites:**
- Vagrant installed (use `neurondb-setup.sh vagrant-deps`)
- VirtualBox installed

**What it does:**
- Starts Vagrant VM
- Copies packages to VM
- Installs packages
- Runs basic verification
- Optionally destroys VM

---

#### `validate-helm` - Validate Helm Charts
Validate Helm chart syntax and structure.

```bash
# Validate Helm chart
./neurondb-pkgs.sh validate-helm
```

**Prerequisites:**
- Helm installed: https://helm.sh/docs/intro/install/

**What it checks:**
- Chart syntax
- Required files
- Values validation
- Template validation

---

## 📝 Blog Management: `neurondb-blogs.sh`

**Purpose:** Complete blog maintenance operations including formatting fixes, image path updates, and content extraction.

### Commands

#### `fix-markdown` - Fix Markdown Formatting
Fix escaped backticks and markdown formatting issues.

```bash
# Fix markdown formatting
./neurondb-blogs.sh fix-markdown

# Dry run to preview changes
./neurondb-blogs.sh fix-markdown --dry-run

# Custom blog directory
./neurondb-blogs.sh fix-markdown --blog-dir ./content/blog
```

**Options:**
- `--blog-dir DIR` - Blog directory (default: `./blog`)
- `--dry-run` - Preview changes without applying

**What it fixes:**
- Escaped backticks (`\`` → `` ` ``)
- Triple backticks (`\`\`\`` → ` ``` `)
- Markdown formatting issues

---

#### `fix-image-paths` - Fix Image Paths
Update image paths in blog files.

```bash
# Fix image paths
./neurondb-blogs.sh fix-image-paths

# Dry run
./neurondb-blogs.sh fix-image-paths --dry-run
```

**What it does:**
- Updates paths from `/blog/{slug}/` to `assets/{slug}/`
- Removes version query parameters
- Updates all markdown files in blog directory

---

#### `fix-code-blocks` - Fix Code Block Formatting
Ensure code blocks have correct formatting.

```bash
# Fix code blocks
./neurondb-blogs.sh fix-code-blocks
```

**What it fixes:**
- Code block language tags
- Code block formatting
- Ensures proper syntax highlighting

---

#### `copy` - Copy Blogs from Source
Copy blog markdown and assets from neurondb-www repository.

```bash
# Copy blogs from neurondb-www
./neurondb-blogs.sh copy --source /path/to/neurondb-www

# Custom destination
./neurondb-blogs.sh copy --source /path/to/neurondb-www --output ./blog
```

**Options:**
- `--source DIR` - Source directory (default: `/Users/pgedge/pge/neurondb-www`)
- `--output DIR` - Output directory (default: `./blog`)

**What it does:**
- Extracts markdown from React components
- Copies blog assets (images, etc.)
- Organizes files by blog slug

---

#### `convert-html` - Convert HTML to Markdown
Convert HTML blog content to markdown format.

```bash
# Convert HTML to markdown
./neurondb-blogs.sh convert-html --input blog.html --output ./blog
```

**Options:**
- `--input FILE` - Input HTML file (required)
- `--output DIR` - Output directory (default: `./blog`)

**What it converts:**
- HTML headings to markdown headings
- HTML paragraphs to markdown
- HTML images to markdown images
- HTML lists to markdown lists
- HTML code blocks to markdown code blocks

---

#### `extract` - Extract Blog Content
Extract blog content from React components.

```bash
# Extract blogs from neurondb-www
./neurondb-blogs.sh extract --source /path/to/neurondb-www
```

**Options:**
- `--source DIR` - Source directory (default: `/Users/pgedge/pge/neurondb-www`)

**What it extracts:**
- Markdown content from React `page.tsx` files
- Blog assets
- Organizes by blog slug

---

#### `svg-to-png` - Convert SVG to PNG
Convert SVG images to PNG format.

```bash
# Convert SVGs to PNGs
./neurondb-blogs.sh svg-to-png

# Custom dimensions
./neurondb-blogs.sh svg-to-png --width 1920 --height 1080

# Force conversion (overwrite existing)
./neurondb-blogs.sh svg-to-png --force
```

**Options:**
- `--width WIDTH` - PNG width in pixels (default: 1200)
- `--height HEIGHT` - PNG height in pixels (default: 800)
- `--force` - Force conversion even if PNG exists

**Prerequisites:**
- One of: `rsvg-convert`, `inkscape`, or ImageMagick `convert`
- On macOS: `brew install librsvg`

---

## 🧹 Cleanup: `neurondb-cleanup.sh`

**Purpose:** Clean build artifacts, Docker resources, logs, and temporary files.

### Usage

```bash
# Preview cleanup (safe)
./neurondb-cleanup.sh --all --dry-run

# Clean everything
./neurondb-cleanup.sh --all

# Clean only Docker resources
./neurondb-cleanup.sh --docker

# Clean logs and build artifacts
./neurondb-cleanup.sh --logs --build

# Clean cache directories
./neurondb-cleanup.sh --cache
```

**Options:**
- `--all` - Clean everything (Docker, logs, build artifacts, cache)
- `--docker` - Clean Docker containers, images, volumes (⚠️ destructive!)
- `--logs` - Clean log files
- `--build` - Clean build artifacts
- `--cache` - Clean cache directories (node_modules, venv, etc.)
- `--dry-run` - Preview without making changes

**What it cleans:**
- **Docker**: Containers, images, volumes
- **Logs**: All log files and log directories
- **Build**: Compiled binaries, object files, build artifacts
- **Cache**: node_modules, Python venv, Go vendor, build caches

**⚠️ Warning:** `--docker` will stop and remove all containers and volumes, resulting in data loss!

---

## 🔧 Common Workflows

### Complete Fresh Installation

```bash
# 1. Setup complete ecosystem
./neurondb-setup.sh ecosystem

# 2. Verify installation
./neurondb-setup.sh verify

# 3. Quick health check
./neurondb-healthcheck.sh quick

# 4. Monitor status
./neurondb-monitor.sh status
```

---

### Docker Deployment

```bash
# 1. Start all services
./neurondb-docker.sh run --component neurondb --variant cpu
./neurondb-docker.sh run --component neuronagent
./neurondb-docker.sh run --component neuronmcp

# 2. Verify Docker setup
./neurondb-docker.sh verify --dependencies --ecosystem

# 3. Run tests
./neurondb-docker.sh test --type basic

# 4. Monitor
./neurondb-monitor.sh watch
```

---

### Database Backup & Restore

```bash
# 1. Create backup
./neurondb-database.sh backup --format custom --retention 30

# 2. List backups
./neurondb-database.sh list-backups

# 3. Restore if needed
./neurondb-database.sh restore --backup backups/neurondb_backup_20250101_120000.dump

# 4. Verify restore
./neurondb-database.sh verify
```

---

### Development Testing

```bash
# Quick smoke test
./neurondb-healthcheck.sh smoke

# Full health check
./neurondb-healthcheck.sh health

# Integration tests
./neurondb-healthcheck.sh integration

# Monitor during development
./neurondb-monitor.sh watch --interval 3
```

---

### Release Workflow

```bash
# 1. Create release
./neurondb-workflows.sh release --version 2.0.0 --dry-run
./neurondb-workflows.sh release --version 2.0.0

# 2. Sync version branches
./neurondb-workflows.sh sync --from-branch main --to-branch REL1_STABLE

# 3. Update references after file renames
./neurondb-workflows.sh update-refs

# 4. Safe git pull
./neurondb-workflows.sh pull
```

---

### Package Testing

```bash
# 1. Verify packages
./neurondb-pkgs.sh verify --os ubuntu

# 2. Test in Vagrant
./neurondb-pkgs.sh test-vagrant --os rocky

# 3. Generate SDKs
./neurondb-pkgs.sh generate-sdk --language python

# 4. Validate Helm charts
./neurondb-pkgs.sh validate-helm
```

---

## 📚 Environment Variables

All scripts support environment variables for configuration:

### Database Configuration
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=neurondb
export DB_USER=postgres
export DB_PASSWORD=your_password
```

### Docker Configuration
```bash
export POSTGRES_USER=neurondb
export POSTGRES_PASSWORD=neurondb
export POSTGRES_DB=neurondb
export PG_MAJOR=18
export CUDA_VERSION=12.4.1
export ONNX_VERSION=1.17.0
```

### Service Configuration
```bash
export SERVER_PORT=8080
export AGENT_URL=http://localhost:8080
export NEURONDB_HOST=neurondb
export NEURONDB_PORT=5432
```

---

## 🆘 Troubleshooting

### Script Permission Denied
```bash
# Make script executable
chmod +x scripts/neurondb-*.sh
```

### Database Connection Failed
```bash
# Test connection manually
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT 1;"

# Check Docker container
./neurondb-docker.sh status
./neurondb-docker.sh logs --component neurondb
```

### Docker Issues
```bash
# Check Docker status
./neurondb-docker.sh verify --dependencies

# View container logs
./neurondb-docker.sh logs --component neurondb --follow

# Restart containers
./neurondb-docker.sh run --component neurondb --variant cpu --action clean
./neurondb-docker.sh run --component neurondb --variant cpu
```

### Help and Documentation
```bash
# Get help for any script
./neurondb-*.sh --help

# Verbose output
./neurondb-*.sh COMMAND --verbose

# Dry run (where applicable)
./neurondb-*.sh COMMAND --dry-run
```

---

## 📊 Script Statistics

| Category | Scripts | Description |
|----------|---------|-------------|
| **Docker Management** | 1 | Complete Docker operations |
| **Database Management** | 1 | Backup, restore, setup, maintenance |
| **Setup & Installation** | 1 | Installation and setup |
| **Health Checking** | 1 | Health checks and testing |
| **Monitoring** | 1 | Status monitoring and logs |
| **Workflows** | 1 | Release and git operations |
| **Package Management** | 1 | Package verification and SDK generation |
| **Blog Management** | 1 | Blog maintenance operations |
| **Utilities** | 2 | Cleanup and password generation |
| **Total** | 10 | Professional production scripts |

---

## 🎓 Learning Path

### New Users
1. Start with setup: `./neurondb-setup.sh ecosystem`
2. Verify: `./neurondb-healthcheck.sh quick`
3. Monitor: `./neurondb-monitor.sh watch`
4. Explore logs: `./neurondb-monitor.sh logs --component neurondb`

### Developers
1. Use health checks: `./neurondb-healthcheck.sh integration`
2. Monitor during development: `./neurondb-monitor.sh watch`
3. Test Docker: `./neurondb-docker.sh test --type basic`
4. Backup test data: `./neurondb-database.sh backup`

### DevOps/Operators
1. Deploy: `./neurondb-setup.sh install --mode docker`
2. Monitor: `./neurondb-monitor.sh metrics`
3. Backup strategy: `./neurondb-database.sh backup --retention 30`
4. Troubleshoot: `./neurondb-monitor.sh logs --follow`

---

## 🔗 Related Resources

- **[Main README](../README.md)** - Project overview
- **[Quick Start](../QUICKSTART.md)** - Get started quickly
- **[Documentation Index](../Docs/documentation.md)** - Complete documentation
- **[Contributing Guide](../CONTRIBUTING.md)** - Contribution guidelines

---

## 📝 Script Development

All scripts follow these principles:
- ✅ **Self-sufficient** - No external dependencies
- ✅ **Standard CLI** - `--help`, `--verbose`, `--version`, `--dry-run`
- ✅ **Modular** - Commands organized by functionality
- ✅ **Professional** - Consistent naming and structure
- ✅ **Well-documented** - Comprehensive help and examples

---

**Last Updated:** 2025-01-01  
**Scripts Version:** 2.0.0  
**Maintainer:** NeuronDB Team
