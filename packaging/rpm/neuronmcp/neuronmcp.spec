%define name neuronmcp
%define version 1.0.0.beta
%define release 1

Summary: NeuronMCP - Model Context Protocol Server for NeuronDB
Name: %{name}
Version: %{version}
Release: %{release}%{?dist}
License: AGPL-3.0
Group: Applications/Databases
Source0: %{name}-%{version}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-root-%(%{__id_u} -n)
URL: https://www.neurondb.ai
Vendor: neurondb
Packager: neurondb <admin@neurondb.com>

Requires: postgresql >= 16, ca-certificates
BuildRequires: golang >= 1.23

%description
NeuronMCP implements the Model Context Protocol using JSON-RPC 2.0 over stdio.
It provides tools and resources for MCP clients to interact with NeuronDB,
including vector operations, ML model training, and database schema management.

Features:
  - Full MCP protocol implementation with stdio transport
  - Vector operations (search, embedding generation, indexing)
  - ML tools for training and prediction
  - Resources for schema, models, indexes, config, workers, stats
  - Middleware for validation, logging, timeout, error handling
  - JSON config files with environment variable overrides
  - Compatible with MCP clients like Claude Desktop

%prep
%setup -q

%build
cd NeuronMCP
go mod download
CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo \
    -ldflags="-w -s" \
    -o neurondb-mcp \
    ./cmd/neurondb-mcp

%install
rm -rf %{buildroot}
mkdir -p %{buildroot}

# Install binary
install -D -m 755 NeuronMCP/neurondb-mcp %{buildroot}%{_bindir}/neurondb-mcp

# Install configuration
mkdir -p %{buildroot}%{_sysconfdir}/neuronmcp
if [ -f NeuronMCP/mcp-config.json.example ]; then
    install -m 644 NeuronMCP/mcp-config.json.example %{buildroot}%{_sysconfdir}/neuronmcp/mcp-config.json.example
elif [ -f NeuronMCP/mcp-config.json ]; then
    install -m 644 NeuronMCP/mcp-config.json %{buildroot}%{_sysconfdir}/neuronmcp/mcp-config.json.example
fi

# Create directories
mkdir -p %{buildroot}/var/lib/neuronmcp
mkdir -p %{buildroot}/var/log/neuronmcp

%pre
# Create user and group
getent group neuronmcp >/dev/null || groupadd -r neuronmcp
getent passwd neuronmcp >/dev/null || \
    useradd -r -g neuronmcp -d /var/lib/neuronmcp -s /sbin/nologin \
    -c "NeuronMCP service user" neuronmcp

%post
# Set permissions
chown -R neuronmcp:neuronmcp /var/lib/neuronmcp
chown -R neuronmcp:neuronmcp /var/log/neuronmcp
chmod 755 /etc/neuronmcp
chmod 755 /var/lib/neuronmcp
chmod 755 /var/log/neuronmcp

# Copy example config if config doesn't exist
if [ ! -f /etc/neuronmcp/mcp-config.json ]; then
    if [ -f /etc/neuronmcp/mcp-config.json.example ]; then
        cp /etc/neuronmcp/mcp-config.json.example /etc/neuronmcp/mcp-config.json
        chown neuronmcp:neuronmcp /etc/neuronmcp/mcp-config.json
        chmod 640 /etc/neuronmcp/mcp-config.json
    fi
fi

%files
%defattr(-,root,root,-)
%{_bindir}/neurondb-mcp
%config(noreplace) %{_sysconfdir}/neuronmcp/mcp-config.json.example
%dir /var/lib/neuronmcp
%dir /var/log/neuronmcp

%changelog
* $(date +"%a %b %d %Y") neurondb <admin@neurondb.com> - 1.0.0.beta-1
- Initial beta release of NeuronMCP



