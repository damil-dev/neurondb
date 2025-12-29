%define name neurondesktop
%define version 1.0.0.beta
%define release 1

Summary: NeuronDesktop - Unified Web Interface for NeuronDB Ecosystem
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
BuildRequires: golang >= 1.23, nodejs >= 18, npm

%description
NeuronDesktop is a comprehensive web application that provides a unified
interface for managing and interacting with MCP servers, NeuronDB, and
NeuronAgent.

Features:
  - Unified dashboard for all NeuronDB ecosystem components
  - Real-time communication via WebSocket
  - Secure API key-based authentication
  - Professional UI with modern, responsive design
  - Comprehensive logging and metrics
  - MCP server integration and testing
  - Agent management through the UI

%prep
%setup -q

%build
cd NeuronDesktop/api
go mod download
CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo \
    -ldflags="-w -s" \
    -o neurondesktop-api \
    ./cmd/server

cd ../frontend
npm install
npm run build

%install
rm -rf %{buildroot}
mkdir -p %{buildroot}

# Install binary
install -D -m 755 NeuronDesktop/api/neurondesktop-api %{buildroot}%{_bindir}/neurondesktop-api

# Install frontend
mkdir -p %{buildroot}/var/www/neurondesktop
if [ -d NeuronDesktop/frontend/.next ]; then
    cp -r NeuronDesktop/frontend/.next %{buildroot}/var/www/neurondesktop/
fi
if [ -d NeuronDesktop/frontend/public ]; then
    cp -r NeuronDesktop/frontend/public %{buildroot}/var/www/neurondesktop/ || true
fi

# Install migrations
if [ -d NeuronDesktop/api/migrations ]; then
    mkdir -p %{buildroot}%{_datadir}/neurondesktop/migrations
    cp -r NeuronDesktop/api/migrations/* %{buildroot}%{_datadir}/neurondesktop/migrations/
fi

# Install systemd service
mkdir -p %{buildroot}%{_unitdir}
cat > %{buildroot}%{_unitdir}/neurondesktop.service << 'EOF'
[Unit]
Description=NeuronDesktop - Unified Web Interface for NeuronDB Ecosystem
Documentation=https://www.neurondb.ai/docs/neurondesktop
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=simple
User=neurondesktop
Group=neurondesktop
WorkingDirectory=/var/lib/neurondesktop
ExecStart=%{_bindir}/neurondesktop-api
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=neurondesktop
Environment="CONFIG_PATH=%{_sysconfdir}/neurondesktop/config.yaml"
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/neurondesktop /var/log/neurondesktop /var/www/neurondesktop

[Install]
WantedBy=multi-user.target
EOF

# Create directories
mkdir -p %{buildroot}/var/lib/neurondesktop
mkdir -p %{buildroot}/var/log/neurondesktop

%pre
# Create user and group
getent group neurondesktop >/dev/null || groupadd -r neurondesktop
getent passwd neurondesktop >/dev/null || \
    useradd -r -g neurondesktop -d /var/lib/neurondesktop -s /sbin/nologin \
    -c "NeuronDesktop service user" neurondesktop

%post
# Set permissions
chown -R neurondesktop:neurondesktop /var/lib/neurondesktop
chown -R neurondesktop:neurondesktop /var/log/neurondesktop
chown -R neurondesktop:neurondesktop /var/www/neurondesktop
chmod 755 /etc/neurondesktop
chmod 755 /var/lib/neurondesktop
chmod 755 /var/log/neurondesktop
chmod 755 /var/www/neurondesktop

# Create config if it doesn't exist
if [ ! -f /etc/neurondesktop/config.yaml ]; then
    cat > /etc/neurondesktop/config.yaml << 'EOF'
# NeuronDesktop Configuration
server:
  host: "0.0.0.0"
  port: 8081
  frontend_dir: "/var/www/neurondesktop"

database:
  host: "localhost"
  port: 5432
  name: "neurondesk"
  user: "neurondesk"
  password: ""

neuronagent:
  endpoint: "http://localhost:8080"
  api_key: ""

neurondb:
  host: "localhost"
  port: 5432
  database: "neurondb"
  user: "neurondb"
  password: ""
EOF
    chown neurondesktop:neurondesktop /etc/neurondesktop/config.yaml
    chmod 640 /etc/neurondesktop/config.yaml
fi

# Reload systemd and enable service
systemctl daemon-reload
systemctl enable neurondesktop.service

%preun
if [ $1 -eq 0 ]; then
    systemctl stop neurondesktop.service || true
    systemctl disable neurondesktop.service || true
fi

%files
%defattr(-,root,root,-)
%{_bindir}/neurondesktop-api
%config(noreplace) %{_sysconfdir}/neurondesktop/config.yaml
%{_datadir}/neurondesktop/migrations
%{_unitdir}/neurondesktop.service
%dir /var/lib/neurondesktop
%dir /var/log/neurondesktop
%dir /var/www/neurondesktop

%changelog
* $(date +"%a %b %d %Y") neurondb <admin@neurondb.com> - 1.0.0.beta-1
- Initial beta release of NeuronDesktop

