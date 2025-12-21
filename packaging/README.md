# NeuronDB Ecosystem Packaging

This directory contains scripts to build DEB (Debian/Ubuntu), RPM (RHEL/CentOS/Rocky), and macOS (.pkg) packages for the NeuronDB ecosystem components.

## Components

Three separate packages are provided:

- **neurondb**: PostgreSQL extension for vector search and ML
- **neuronagent**: AI agent runtime system with REST API
- **neuronmcp**: Model Context Protocol server

## Configuration

Before building packages, you can configure build options in `build-config.json`:

```json
{
  "version": "1.0.0.beta",
  "architecture": "amd64",
  "neurondb": {
    "postgresql_versions": ["16", "17", "18"],
    "gpu_backends": "none",
    "compute_mode": "cpu",
    "cuda": {
      "enabled": false,
      "path": "/usr/local/cuda",
      "version": "12.0"
    },
    "rocm": {
      "enabled": false,
      "path": "/opt/rocm",
      "version": "5.7"
    },
    "metal": {
      "enabled": false
    }
  }
}
```

**Configuration Options:**

- `version`: Package version (default: 1.0.0.beta)
- `architecture`: Target architecture (default: amd64)
- `neurondb.postgresql_versions`: Array of PostgreSQL versions to build for (default: ["16", "17", "18"])
- `neurondb.gpu_backends`: GPU backend selection - "none", "cuda", "rocm", "metal", or "auto" (default: "none")
- `neurondb.compute_mode`: Compute mode - "cpu", "cuda", "rocm", "metal" (default: "cpu")
- `neurondb.cuda.enabled`: Enable CUDA support (default: false)
- `neurondb.cuda.path`: Path to CUDA installation (default: auto-detect)
- `neurondb.cuda.version`: CUDA version (optional)
- `neurondb.rocm.enabled`: Enable ROCm support (default: false)
- `neurondb.rocm.path`: Path to ROCm installation (default: auto-detect)
- `neurondb.rocm.version`: ROCm version (optional)
- `neurondb.metal.enabled`: Enable Metal support for macOS (default: false)

**Example: Build with CUDA support**
```json
{
  "neurondb": {
    "gpu_backends": "cuda",
    "compute_mode": "cuda",
    "cuda": {
      "enabled": true,
      "path": "/usr/local/cuda-12.0"
    }
  }
}
```

**Example: Build for specific PostgreSQL versions only**
```json
{
  "neurondb": {
    "postgresql_versions": ["17", "18"]
  }
}
```

If `build-config.json` is not present, the build scripts will use default values. You can also override the config file location with:
```bash
export PACKAGING_CONFIG=/path/to/custom-config.json
```

**Pre-configured example files:**
- `build-config.json` - Default CPU-only build
- `build-config.example.json` - Template with all options documented
- `build-config.cuda.json` - Example for CUDA builds
- `build-config.rocm.json` - Example for ROCm builds
- `build-config.metal.json` - Example for Metal builds (macOS)

To use an example config:
```bash
cp build-config.cuda.json build-config.json
./build-all-deb.sh
```

## Prerequisites

### For DEB Packages

```bash
# Ubuntu/Debian
sudo apt-get install -y dpkg-dev fakeroot build-essential

# For NeuronDB
sudo apt-get install -y postgresql-server-dev-16 postgresql-server-dev-17 postgresql-server-dev-18

# For Go services (NeuronAgent, NeuronMCP)
sudo apt-get install -y golang-go  # Or install Go 1.23+ from golang.org
```

### For RPM Packages

```bash
# RHEL/CentOS/Rocky
sudo dnf install -y rpm-build gcc make

# For NeuronDB
sudo dnf install -y postgresql17-devel postgresql16-devel postgresql18-devel

# For Go services
sudo dnf install -y golang  # Or install Go 1.23+ from golang.org
```

### For macOS Packages (.pkg)

```bash
# macOS requires Xcode Command Line Tools
xcode-select --install

# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# For NeuronDB - install PostgreSQL via Homebrew
brew install postgresql@17  # or postgresql@16, postgresql@18

# For Go services - install Go
brew install go  # Or install Go 1.23+ from golang.org
```

## Building Packages

### Build All Packages

**DEB packages:**
```bash
cd packaging/deb
./build-all-deb.sh [VERSION]
```

**RPM packages:**
```bash
cd packaging/rpm
./build-all-rpm.sh [VERSION]
```

**macOS packages (.pkg):**
```bash
cd packaging/pkg
./build-all-pkg.sh [VERSION]
```

Default version is `1.0.0.beta`. Override by passing version as argument or setting `VERSION` environment variable.

### Build Individual Packages

**NeuronDB:**
```bash
cd packaging/deb/neurondb
./build.sh

# Or for RPM
cd packaging/rpm/neurondb
./build.sh
```

**NeuronAgent:**
```bash
cd packaging/deb/neuronagent
./build.sh

# Or for RPM
cd packaging/rpm/neuronagent
./build.sh
```

**NeuronMCP:**
```bash
cd packaging/deb/neuronmcp
./build.sh

# Or for RPM
cd packaging/rpm/neuronmcp
./build.sh

# Or for macOS
cd packaging/pkg/neuronmcp
./build.sh
```

## Package Contents

### NeuronDB Package

- **Binary**: `neurondb.so` (Linux) or `neurondb.dylib` (macOS) installed to PostgreSQL extension directory
- **SQL files**: Extension SQL and control files
- **Location**: Built for all installed PostgreSQL versions (16, 17, 18)
  - Linux: `/usr/lib/postgresql/*/lib/` and `/usr/share/postgresql/*/extension/`
  - macOS: `/opt/homebrew/opt/postgresql@*/lib/` and `/opt/homebrew/opt/postgresql@*/share/postgresql/extension/` (Apple Silicon)

### NeuronAgent Package

- **Binary**: `/usr/bin/neuronagent` (Linux) or `/usr/local/bin/neuronagent` (macOS)
- **Configuration**: `/etc/neuronagent/config.yaml.example`
- **Migrations**: `/usr/share/neuronagent/migrations/` (Linux) or `/usr/local/share/neuronagent/migrations/` (macOS)
- **Systemd service**: `/etc/systemd/system/neuronagent.service` (Linux only)
- **User**: `neuronagent` (Linux only, created automatically)
- **Directories**: `/var/lib/neuronagent`, `/var/log/neuronagent` (Linux only)

### NeuronMCP Package

- **Binary**: `/usr/bin/neurondb-mcp` (Linux) or `/usr/local/bin/neurondb-mcp` (macOS)
- **Configuration**: `/etc/neuronmcp/mcp-config.json.example`
- **User**: `neuronmcp` (Linux only, created automatically)
- **Directories**: `/var/lib/neuronmcp`, `/var/log/neuronmcp` (Linux only)

## Installation

### DEB Packages (Ubuntu/Debian)

```bash
# Install all packages
sudo dpkg -i neurondb_1.0.0.beta_amd64.deb
sudo dpkg -i neuronagent_1.0.0.beta_amd64.deb
sudo dpkg -i neuronmcp_1.0.0.beta_amd64.deb

# Fix any missing dependencies
sudo apt-get install -f
```

### RPM Packages (RHEL/CentOS/Rocky)

```bash
# Install all packages
sudo rpm -ivh neurondb-1.0.0.beta-1.el*.rpm
sudo rpm -ivh neuronagent-1.0.0.beta-1.el*.rpm
sudo rpm -ivh neuronmcp-1.0.0.beta-1.el*.rpm
```

### macOS Packages (.pkg)

```bash
# Install individual packages
sudo installer -pkg neurondb-1.0.0.beta-arm64.pkg -target /
sudo installer -pkg neuronagent-1.0.0.beta-arm64.pkg -target /
sudo installer -pkg neuronmcp-1.0.0.beta-arm64.pkg -target /

# Or use the installer GUI by double-clicking the .pkg file
```

## Post-Installation

### NeuronDB

1. Connect to your PostgreSQL database:
   ```bash
   psql -d your_database
   ```

2. Create the extension:
   ```sql
   CREATE EXTENSION neurondb;
   ```

3. Verify installation:
   ```sql
   SELECT neurondb.version();
   ```

### NeuronAgent

1. Configure database connection:
   ```bash
   # Linux
   sudo nano /etc/neuronagent/config.yaml
   
   # macOS
   sudo nano /etc/neuronagent/config.yaml
   ```

2. Start the service:
   ```bash
   # Linux (systemd)
   sudo systemctl start neuronagent
   sudo systemctl enable neuronagent
   
   # macOS (run manually or use launchd)
   /usr/local/bin/neuronagent
   ```

3. Check status:
   ```bash
   # Linux
   sudo systemctl status neuronagent
   curl http://localhost:8080/health
   
   # macOS
   curl http://localhost:8080/health
   ```

### NeuronMCP

1. Configure database connection:
   ```bash
   sudo nano /etc/neuronmcp/mcp-config.json
   ```

2. Configure your MCP client (e.g., Claude Desktop) to use:
   ```
   /usr/bin/neurondb-mcp
   ```

## Package Structure

```
packaging/
├── deb/
│   ├── neurondb/
│   │   ├── DEBIAN/
│   │   │   ├── control      # Package metadata
│   │   │   ├── postinst     # Post-install script
│   │   │   └── prerm        # Pre-removal script
│   │   └── build.sh         # Build script
│   ├── neuronagent/
│   │   ├── DEBIAN/
│   │   │   ├── control
│   │   │   ├── postinst
│   │   │   └── prerm
│   │   ├── neuronagent.service  # Systemd service file
│   │   └── build.sh
│   ├── neuronmcp/
│   │   ├── DEBIAN/
│   │   │   ├── control
│   │   │   ├── postinst
│   │   │   └── prerm
│   │   └── build.sh
│   └── build-all-deb.sh     # Master build script
├── rpm/
│   ├── neurondb/
│   │   ├── neurondb.spec    # RPM spec file
│   │   └── build.sh
│   ├── neuronagent/
│   │   ├── neuronagent.spec
│   │   └── build.sh
│   ├── neuronmcp/
│   │   ├── neuronmcp.spec
│   │   └── build.sh
│   └── build-all-rpm.sh     # Master build script
├── pkg/
│   ├── neurondb/
│   │   ├── scripts/
│   │   │   └── postinstall  # Post-install script
│   │   └── build.sh         # macOS package build script
│   ├── neuronagent/
│   │   ├── scripts/
│   │   │   └── postinstall
│   │   └── build.sh
│   ├── neuronmcp/
│   │   ├── scripts/
│   │   │   └── postinstall
│   │   └── build.sh
│   └── build-all-pkg.sh     # Master build script for macOS
└── README.md                 # This file
```

## Troubleshooting

### Build Errors

**"pg_config not found"**
- Install PostgreSQL development packages for the versions you want to support
- For DEB: `postgresql-server-dev-16`, `postgresql-server-dev-17`, etc.
- For RPM: `postgresql17-devel`, `postgresql16-devel`, etc.

**"Go not found"**
- Install Go 1.23 or later
- Ensure `go` is in your PATH
- Verify with: `go version`

**"rpmbuild not found"**
- Install: `sudo dnf install rpm-build` (RHEL/CentOS/Rocky)

**"fakeroot not found"**
- Install: `sudo apt-get install fakeroot` (Ubuntu/Debian)

### Installation Errors

**Missing dependencies**
- For DEB: `sudo apt-get install -f`
- For RPM: Check required packages with `rpm -qpR <package.rpm>`

**Service won't start**
- Check configuration files: `/etc/neuronagent/config.yaml`
- Check logs: `journalctl -u neuronagent`
- Verify database connection

## Version Management

Default version is `1.0.0.beta`. To build with a different version:

**Option 1: Edit build-config.json**
```json
{
  "version": "1.0.0"
}
```

**Option 2: Via environment variable**
```bash
export VERSION=1.0.0
./build-all-deb.sh
```

**Option 3: Via argument**
```bash
./build-all-deb.sh 1.0.0
```

**Option 4: Custom config file**
```bash
export PACKAGING_CONFIG=/path/to/custom-config.json
./build-all-deb.sh
```

## Support

For issues and questions:
- Documentation: https://www.neurondb.ai/docs
- Support: admin@neurondb.com

