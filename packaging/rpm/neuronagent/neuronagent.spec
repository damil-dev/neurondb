%define name neuronagent
%define version 1.0.0.beta
%define release 1

Summary: NeuronAgent - AI Agent Runtime System
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

Requires: postgresql16 >= 16.0 | postgresql17 >= 17.0 | postgresql18 >= 18.0 | postgresql >= 16, ca-certificates
BuildRequires: golang >= 1.23

%description
NeuronAgent is a production-ready AI agent runtime system that provides
REST API and WebSocket endpoints for building applications with long-term
memory and tool execution.

Features:
  - Agent state machine for autonomous task execution
  - Long-term memory with HNSW vector search
  - Tool registry (SQL, HTTP, code execution, shell commands)
  - REST API for agent, session, and message management
  - WebSocket support for streaming responses
  - API key authentication with rate limiting
  - Background job queue with worker pool
  - Direct integration with NeuronDB embedding and LLM functions

%prep
%setup -q

%build
cd NeuronAgent
go mod download
CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo \
    -ldflags="-w -s" \
    -o neuronagent \
    ./cmd/agent-server

%install
rm -rf %{buildroot}
mkdir -p %{buildroot}

# Install binary
install -D -m 755 NeuronAgent/neuronagent %{buildroot}%{_bindir}/neuronagent

# Install configuration
mkdir -p %{buildroot}%{_sysconfdir}/neuronagent
if [ -f NeuronAgent/config.yaml ]; then
    install -m 644 NeuronAgent/config.yaml %{buildroot}%{_sysconfdir}/neuronagent/config.yaml.example
elif [ -f NeuronAgent/configs/config.yaml.example ]; then
    install -m 644 NeuronAgent/configs/config.yaml.example %{buildroot}%{_sysconfdir}/neuronagent/config.yaml.example
fi

# Install migrations
if [ -d NeuronAgent/migrations ]; then
    mkdir -p %{buildroot}%{_datadir}/neuronagent/migrations
    cp -r NeuronAgent/migrations/* %{buildroot}%{_datadir}/neuronagent/migrations/
fi

# Install systemd service
mkdir -p %{buildroot}%{_unitdir}
cat > %{buildroot}%{_unitdir}/neuronagent.service << 'EOF'
[Unit]
Description=NeuronAgent - AI Agent Runtime System
Documentation=https://www.neurondb.ai/docs/neuronagent
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=simple
User=neuronagent
Group=neuronagent
WorkingDirectory=/var/lib/neuronagent
ExecStart=%{_bindir}/neuronagent
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=neuronagent
Environment="CONFIG_PATH=%{_sysconfdir}/neuronagent/config.yaml"
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/neuronagent /var/log/neuronagent

[Install]
WantedBy=multi-user.target
EOF

# Create directories
mkdir -p %{buildroot}/var/lib/neuronagent
mkdir -p %{buildroot}/var/log/neuronagent

%pre
# Create user and group
getent group neuronagent >/dev/null || groupadd -r neuronagent
getent passwd neuronagent >/dev/null || \
    useradd -r -g neuronagent -d /var/lib/neuronagent -s /sbin/nologin \
    -c "NeuronAgent service user" neuronagent

%post
# Set permissions
chown -R neuronagent:neuronagent /var/lib/neuronagent
chown -R neuronagent:neuronagent /var/log/neuronagent
chmod 755 /etc/neuronagent
chmod 755 /var/lib/neuronagent
chmod 755 /var/log/neuronagent

# Copy example config if config doesn't exist
if [ ! -f /etc/neuronagent/config.yaml ]; then
    if [ -f /etc/neuronagent/config.yaml.example ]; then
        cp /etc/neuronagent/config.yaml.example /etc/neuronagent/config.yaml
        chown neuronagent:neuronagent /etc/neuronagent/config.yaml
        chmod 640 /etc/neuronagent/config.yaml
    fi
fi

# Reload systemd and enable service
systemctl daemon-reload
systemctl enable neuronagent.service

%preun
if [ $1 -eq 0 ]; then
    systemctl stop neuronagent.service || true
    systemctl disable neuronagent.service || true
fi

%files
%defattr(-,root,root,-)
%{_bindir}/neuronagent
%config(noreplace) %{_sysconfdir}/neuronagent/config.yaml.example
%{_datadir}/neuronagent/migrations
%{_unitdir}/neuronagent.service
%dir /var/lib/neuronagent
%dir /var/log/neuronagent

%changelog
* $(date +"%a %b %d %Y") neurondb <admin@neurondb.com> - 1.0.0.beta-1
- Initial beta release of NeuronAgent



