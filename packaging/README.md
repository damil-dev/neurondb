# NeuronDB Ecosystem Packaging

This directory contains scripts to build DEB (Debian/Ubuntu), RPM (RHEL/CentOS/Rocky), and macOS (.pkg) packages for the NeuronDB ecosystem components.

## Components

The NeuronDB ecosystem consists of two types of components with different installation requirements:

Four separate packages are provided:

### PostgreSQL Extension

- **neurondb**: PostgreSQL extension for vector search and ML
  - **Installation Type**: Extension files placed in PostgreSQL directories
  - **Dependency**: Requires appropriate PostgreSQL version (16, 17, or 18)
  - **Location**: Files installed to PostgreSQL extension directories:
    - Linux: `/usr/lib/postgresql/*/lib/` and `/usr/share/postgresql/*/extension/`
    - macOS: `/opt/homebrew/opt/postgresql@*/lib/` and `/opt/homebrew/opt/postgresql@*/share/postgresql/extension/`
  - **Activation**: Must be enabled per database with `CREATE EXTENSION neurondb;`
  - **No Service**: Does not run as a service - it's part of PostgreSQL

### Applications (Services)

These are standalone applications that must be installed and configured as OS services:

- **neuronagent**: AI agent runtime system with REST API
  - **Installation Type**: Application binary + system service
  - **Service Type**: 
    - Linux: systemd service (`/etc/systemd/system/neuronagent.service`)
    - macOS: launchd service (plist file)
  - **Auto-start**: Service is enabled but not started by default (requires configuration)

- **neuronmcp**: Model Context Protocol server
  - **Installation Type**: Application binary (typically used via stdio, not as a service)
  - **Service Type**: Optional systemd/launchd service for background operation
  - **Usage**: Primarily used by MCP clients via stdio communication

- **neurondesktop**: Unified web interface dashboard
  - **Installation Type**: Application binary + frontend + system service
  - **Service Type**:
    - Linux: systemd service (`/etc/systemd/system/neurondesktop.service`)
    - macOS: launchd service (plist file)
  - **Auto-start**: Service is enabled but not started by default (requires configuration)

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

## Package Dependencies

### NeuronDB Extension Package

**Strict PostgreSQL Version Requirements:**
- **DEB**: Requires `postgresql-16 (>= 16.0) | postgresql-17 (>= 17.0) | postgresql-18 (>= 18.0)`
- **RPM**: Requires `postgresql16-server >= 16.0 | postgresql17-server >= 17.0 | postgresql18-server >= 18.0`

**Required Shared Libraries:**
- `libc6` (>= 2.17) - Standard C library
- `libcurl4` (>= 7.16.2) - HTTP client library for ML model runtime
- `libssl3` (>= 3.0.0) | `libssl1.1` (>= 1.1.0) - OpenSSL for encryption
- `zlib1g` (>= 1:1.2.3.4) - Compression library

**Important Notes:**
- The extension is built for specific PostgreSQL versions (16, 17, or 18)
- The package installs extension files for all PostgreSQL versions found during build
- At least one supported PostgreSQL version must be installed
- Extension files are version-specific and must match the PostgreSQL server version

### Application Packages (NeuronAgent, NeuronMCP, NeuronDesktop)

**PostgreSQL Client Requirements:**
- **DEB**: Requires `postgresql-client-16 | postgresql-client-17 | postgresql-client-18 | postgresql-client (>= 16)`
- **RPM**: Requires `postgresql16 >= 16.0 | postgresql17 >= 17.0 | postgresql18 >= 18.0 | postgresql >= 16`

**Required Shared Libraries:**
- `libc6` (>= 2.17) - Standard C library
- `ca-certificates` - SSL certificate bundle for HTTPS connections

**Important Notes:**
- Applications connect to PostgreSQL via client libraries
- They require PostgreSQL 16, 17, or 18 client libraries
- The client version should match or be compatible with the server version

## Prerequisites

### For DEB Packages

```bash
# Ubuntu/Debian
sudo apt-get install -y dpkg-dev fakeroot build-essential

# For NeuronDB (build-time)
sudo apt-get install -y postgresql-server-dev-16 postgresql-server-dev-17 postgresql-server-dev-18

# For NeuronDB (runtime - at least one required)
sudo apt-get install -y postgresql-16 | postgresql-17 | postgresql-18

# For Go services (NeuronAgent, NeuronMCP, NeuronDesktop)
sudo apt-get install -y golang-go  # Or install Go 1.23+ from golang.org

# For NeuronDesktop (also requires Node.js)
sudo apt-get install -y nodejs npm  # Node.js 18+ required
```

### For RPM Packages

```bash
# RHEL/CentOS/Rocky
sudo dnf install -y rpm-build gcc make

# For NeuronDB (build-time)
sudo dnf install -y postgresql16-devel postgresql17-devel postgresql18-devel

# For NeuronDB (runtime - at least one required)
sudo dnf install -y postgresql16-server | postgresql17-server | postgresql18-server

# For Go services (NeuronAgent, NeuronMCP, NeuronDesktop)
sudo dnf install -y golang  # Or install Go 1.23+ from golang.org

# For NeuronDesktop (also requires Node.js)
sudo dnf install -y nodejs npm  # Node.js 18+ required
```

### For macOS Packages (.pkg)

```bash
# macOS requires Xcode Command Line Tools
xcode-select --install

# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# For NeuronDB - install PostgreSQL via Homebrew (at least one required)
brew install postgresql@16 | postgresql@17 | postgresql@18

# For Go services (NeuronAgent, NeuronMCP, NeuronDesktop) - install Go
brew install go  # Or install Go 1.23+ from golang.org

# For NeuronDesktop (also requires Node.js)
brew install node  # Node.js 18+ required
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

**NeuronDesktop:**
```bash
cd packaging/deb/neurondesktop
./build.sh

# Or for RPM
cd packaging/rpm/neurondesktop
./build.sh

# Or for macOS
cd packaging/pkg/neurondesktop
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

### NeuronDesktop Package

- **Binary**: `/usr/bin/neurondesktop-api` (Linux) or `/usr/local/bin/neurondesktop-api` (macOS)
- **Frontend**: `/var/www/neurondesktop/` (production build)
- **Configuration**: `/etc/neurondesktop/config.yaml`
- **Systemd service**: `/etc/systemd/system/neurondesktop.service` (Linux only)
- **Launchd service**: `/Library/LaunchDaemons/com.neurondb.neurondesktop.plist` (macOS only)
- **User**: `neurondesktop` (created automatically)
- **Directories**: `/var/lib/neurondesktop`, `/var/log/neurondesktop`, `/var/www/neurondesktop`

## Installation

### DEB Packages (Ubuntu/Debian)

**Important**: Before installing, ensure you have at least one PostgreSQL version (16, 17, or 18) installed:

```bash
# Install PostgreSQL (at least one version required)
sudo apt-get install -y postgresql-16 | postgresql-17 | postgresql-18

# Install all packages
sudo dpkg -i neurondb_1.0.0.beta_amd64.deb
sudo dpkg -i neuronagent_1.0.0.beta_amd64.deb
sudo dpkg -i neuronmcp_1.0.0.beta_amd64.deb
sudo dpkg -i neurondesktop_1.0.0.beta_amd64.deb

# Fix any missing dependencies
sudo apt-get install -f
```

**Note**: The `neurondb` package requires at least one PostgreSQL server version (16, 17, or 18) to be installed. The package will install extension files for all PostgreSQL versions found on the system.

### RPM Packages (RHEL/CentOS/Rocky)

**Important**: Before installing, ensure you have at least one PostgreSQL version (16, 17, or 18) installed:

```bash
# Install PostgreSQL (at least one version required)
sudo dnf install -y postgresql16-server | postgresql17-server | postgresql18-server

# Install all packages
sudo rpm -ivh neurondb-1.0.0.beta-1.el*.rpm
sudo rpm -ivh neuronagent-1.0.0.beta-1.el*.rpm
sudo rpm -ivh neuronmcp-1.0.0.beta-1.el*.rpm
sudo rpm -ivh neurondesktop-1.0.0.beta-1.el*.rpm
```

**Note**: The `neurondb` package requires at least one PostgreSQL server version (16, 17, or 18) to be installed. The package will install extension files for all PostgreSQL versions found on the system.

### macOS Packages (.pkg)

**Important**: Before installing, ensure you have at least one PostgreSQL version (16, 17, or 18) installed via Homebrew:

```bash
# Install PostgreSQL (at least one version required)
brew install postgresql@16 | postgresql@17 | postgresql@18

# Install individual packages
sudo installer -pkg neurondb-1.0.0.beta-arm64.pkg -target /
sudo installer -pkg neuronagent-1.0.0.beta-arm64.pkg -target /
sudo installer -pkg neuronmcp-1.0.0.beta-arm64.pkg -target /
sudo installer -pkg neurondesktop-1.0.0.beta-arm64.pkg -target /

# Or use the installer GUI by double-clicking the .pkg file
```

**Note**: The `neurondb` package requires at least one PostgreSQL version (16, 17, or 18) to be installed. The package will install extension files for all PostgreSQL versions found on the system.

## Post-Installation

### NeuronDB (PostgreSQL Extension)

**Important**: NeuronDB is a PostgreSQL extension, not a service. It only needs to be placed in the correct PostgreSQL directories (which the package does automatically). The extension must be enabled per database.

1. The package installs files to PostgreSQL extension directories for each PostgreSQL version (16, 17, 18) found on the system.

2. Connect to your PostgreSQL database:
   ```bash
   psql -d your_database
   ```

3. Create the extension in your database:
   ```sql
   CREATE EXTENSION neurondb;
   ```

4. Verify installation:
   ```sql
   SELECT neurondb.version();
   ```

**Note**: The extension is version-specific. If you have multiple PostgreSQL versions installed, the package installs the extension for all of them. Use the appropriate `pg_config` path when building, or install packages built for your specific PostgreSQL version.

### NeuronAgent (Application Service)

**Important**: NeuronAgent is an application that runs as a system service. It must be configured and started as a service on your OS.

1. Configure database connection:
   ```bash
   # Linux
   sudo nano /etc/neuronagent/config.yaml
   
   # macOS
   sudo nano /etc/neuronagent/config.yaml
   ```
   Edit the database connection parameters in the config file.

2. Start the service:
   ```bash
   # Linux (systemd)
   sudo systemctl start neuronagent
   sudo systemctl enable neuronagent  # Enable auto-start on boot
   
   # macOS (launchd)
   sudo launchctl load /Library/LaunchDaemons/com.neurondb.neuronagent.plist
   sudo launchctl start com.neurondb.neuronagent
   ```

3. Check status:
   ```bash
   # Linux
   sudo systemctl status neuronagent
   curl http://localhost:8080/health
   
   # macOS
   sudo launchctl list | grep neuronagent
   curl http://localhost:8080/health
   ```

### NeuronMCP (Application)

**Important**: NeuronMCP is an application that can run as a service or be invoked directly by MCP clients via stdio.

**Option 1: Use with MCP clients (stdio mode - recommended)**

1. Configure database connection:
   ```bash
   sudo nano /etc/neuronmcp/mcp-config.json
   ```

2. Configure your MCP client (e.g., Claude Desktop) to use:
   ```
   /usr/bin/neurondb-mcp  # Linux
   /usr/local/bin/neurondb-mcp  # macOS
   ```

**Option 2: Run as a background service (optional)**

1. Configure database connection (same as above)

2. Start as service:
   ```bash
   # Linux (systemd)
   sudo systemctl start neuronmcp
   sudo systemctl enable neuronmcp
   
   # macOS (launchd)
   sudo launchctl load /Library/LaunchDaemons/com.neurondb.neuronmcp.plist
   sudo launchctl start com.neurondb.neuronmcp
   ```

### NeuronDesktop (Application Service)

**Important**: NeuronDesktop is an application that runs as a system service providing a web interface.

1. Configure database and API connections:
   ```bash
   sudo nano /etc/neurondesktop/config.yaml
   ```

2. Start the service:
   ```bash
   # Linux (systemd)
   sudo systemctl start neurondesktop
   sudo systemctl enable neurondesktop
   
   # macOS (launchd)
   sudo launchctl load /Library/LaunchDaemons/com.neurondb.neurondesktop.plist
   sudo launchctl start com.neurondb.neurondesktop
   ```

3. Access the web interface:
   ```
   http://localhost:3000
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

