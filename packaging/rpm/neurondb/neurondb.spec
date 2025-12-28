%define name neurondb
%define version 1.0.0.beta
%define release 1

Summary: NeuronDB PostgreSQL Extension - Advanced AI Database
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

Requires: postgresql-server >= 16, libcurl, openssl-libs
BuildRequires: postgresql-devel >= 16, gcc, make, libcurl-devel, openssl-devel, zlib-devel

%description
NeuronDB extends PostgreSQL with vector search, machine learning algorithms,
and hybrid search capabilities. It provides:

  - Vector types and operations with HNSW and IVF indexing
  - 50+ ML functions across 19+ algorithms (classification, regression, clustering)
  - Embedding generation (text, image, multimodal)
  - Hybrid search combining vector and full-text
  - RAG pipeline with LLM integration
  - GPU acceleration (CUDA, ROCm, Metal)
  - Background workers for async operations

Supports PostgreSQL 16, 17, and 18.

%prep
%setup -q

%build
# Build for each PostgreSQL version found
PG_VERSIONS=$(ls -1 /usr/pgsql-*/bin/pg_config 2>/dev/null | sed 's|/usr/pgsql-\([0-9]*\)/bin/pg_config|\1|' || \
             ls -1 /usr/lib/pgsql-*/bin/pg_config 2>/dev/null | sed 's|/usr/lib/pgsql-\([0-9]*\)/bin/pg_config|\1|' || \
             echo "16 17 18")

for PG_VERSION in $PG_VERSIONS; do
    if [ -f "/usr/pgsql-${PG_VERSION}/bin/pg_config" ]; then
        PG_CONFIG="/usr/pgsql-${PG_VERSION}/bin/pg_config"
    elif [ -f "/usr/lib/pgsql-${PG_VERSION}/bin/pg_config" ]; then
        PG_CONFIG="/usr/lib/pgsql-${PG_VERSION}/bin/pg_config"
    else
        continue
    fi
    
    echo "Building for PostgreSQL $PG_VERSION..."
    make clean 2>/dev/null || true
    make PG_CONFIG="$PG_CONFIG"
    
    # Store build artifacts
    mkdir -p build-pg${PG_VERSION}
    cp neurondb.so build-pg${PG_VERSION}/ 2>/dev/null || true
    cp neurondb--1.0.sql build-pg${PG_VERSION}/ 2>/dev/null || true
    cp neurondb.control build-pg${PG_VERSION}/ 2>/dev/null || true
    
    make clean 2>/dev/null || true
done

%install
rm -rf %{buildroot}
mkdir -p %{buildroot}

# Install for each PostgreSQL version
PG_VERSIONS=$(ls -1 /usr/pgsql-*/bin/pg_config 2>/dev/null | sed 's|/usr/pgsql-\([0-9]*\)/bin/pg_config|\1|' || \
             ls -1 /usr/lib/pgsql-*/bin/pg_config 2>/dev/null | sed 's|/usr/lib/pgsql-\([0-9]*\)/bin/pg_config|\1|' || \
             echo "16 17 18")

for PG_VERSION in $PG_VERSIONS; do
    if [ -f "/usr/pgsql-${PG_VERSION}/bin/pg_config" ]; then
        PG_CONFIG="/usr/pgsql-${PG_VERSION}/bin/pg_config"
        PG_LIBDIR=$($PG_CONFIG --pkglibdir)
        PG_SHAREDIR=$($PG_CONFIG --sharedir)
    elif [ -f "/usr/lib/pgsql-${PG_VERSION}/bin/pg_config" ]; then
        PG_CONFIG="/usr/lib/pgsql-${PG_VERSION}/bin/pg_config"
        PG_LIBDIR=$($PG_CONFIG --pkglibdir)
        PG_SHAREDIR=$($PG_CONFIG --sharedir)
    else
        continue
    fi
    
    if [ -f "build-pg${PG_VERSION}/neurondb.so" ]; then
        mkdir -p %{buildroot}$PG_LIBDIR
        mkdir -p %{buildroot}$PG_SHAREDIR/extension
        cp build-pg${PG_VERSION}/neurondb.so %{buildroot}$PG_LIBDIR/
        cp build-pg${PG_VERSION}/neurondb--1.0.sql %{buildroot}$PG_SHAREDIR/extension/ 2>/dev/null || true
        cp build-pg${PG_VERSION}/neurondb.control %{buildroot}$PG_SHAREDIR/extension/ 2>/dev/null || true
    fi
done

%post
echo "NeuronDB extension installed successfully."
echo "To use NeuronDB, connect to your database and run:"
echo "  CREATE EXTENSION neurondb;"

%preun
echo "Removing NeuronDB extension..."

%files
%defattr(-,root,root,-)
/usr/pgsql-*/lib/neurondb.so
/usr/pgsql-*/share/extension/neurondb*
/usr/lib/pgsql-*/lib/neurondb.so
/usr/lib/pgsql-*/share/extension/neurondb*

%changelog
* $(date +"%a %b %d %Y") neurondb <admin@neurondb.com> - 1.0.0.beta-1
- Initial beta release of NeuronDB PostgreSQL extension



