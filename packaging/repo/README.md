# NeuronDB Package Repository

This directory contains scripts and configuration for hosting DEB and RPM packages on GitHub Pages.

## Repository Structure

```
repo/
├── deb/
│   ├── dists/
│   │   └── stable/
│   │       ├── main/
│   │       │   └── binary-amd64/
│   │       │       └── Packages.gz
│   │       ├── InRelease
│   │       ├── Release
│   │       └── Release.gpg
│   └── pool/
│       └── main/
│           ├── n/neurondb/
│           ├── n/neuronagent/
│           └── n/neuronmcp/
├── rpm/
│   ├── el9/
│   │   └── x86_64/
│   │       ├── repodata/
│   │       └── *.rpm
│   └── el8/
│       └── x86_64/
│           ├── repodata/
│           └── *.rpm
└── keys/
    ├── neurondb.gpg          # DEB signing key (public)
    └── RPM-GPG-KEY-neurondb  # RPM signing key (public)
```

## Quick Start for Users

### DEB Repository (Ubuntu/Debian)

#### 1. Add GPG Key

```bash
# Create keyrings directory
sudo mkdir -p /etc/apt/keyrings

# Download and add GPG key
curl -fsSL https://USERNAME.github.io/neurondb/repo/keys/neurondb.gpg | sudo tee /etc/apt/keyrings/neurondb.gpg >/dev/null

# Set proper permissions
sudo chmod 644 /etc/apt/keyrings/neurondb.gpg
```

#### 2. Add Repository

```bash
echo "deb [signed-by=/etc/apt/keyrings/neurondb.gpg] https://USERNAME.github.io/neurondb/repo/deb stable main" | sudo tee /etc/apt/sources.list.d/neurondb.list
```

**Note:** Replace `USERNAME` with your GitHub username or organization name.

#### 3. Update and Install

```bash
sudo apt-get update
sudo apt-get install neurondb neuronagent neuronmcp
```

### RPM Repository (RHEL/CentOS/Rocky)

#### 1. Add Repository

For **EL9** (Rocky 9, RHEL 9, CentOS Stream 9):

```bash
sudo tee /etc/yum.repos.d/neurondb.repo <<EOF
[neurondb]
name=NeuronDB RPM Repository
baseurl=https://USERNAME.github.io/neurondb/repo/rpm/el9/x86_64
enabled=1
gpgcheck=1
gpgkey=https://USERNAME.github.io/neurondb/repo/keys/RPM-GPG-KEY-neurondb
EOF
```

For **EL8** (Rocky 8, RHEL 8, CentOS Stream 8):

```bash
sudo tee /etc/yum.repos.d/neurondb.repo <<EOF
[neurondb]
name=NeuronDB RPM Repository
baseurl=https://USERNAME.github.io/neurondb/repo/rpm/el8/x86_64
enabled=1
gpgcheck=1
gpgkey=https://USERNAME.github.io/neurondb/repo/keys/RPM-GPG-KEY-neurondb
EOF
```

**Note:** Replace `USERNAME` with your GitHub username or organization name.

#### 2. Install

```bash
sudo dnf install neurondb neuronagent neuronmcp
```

Or with `yum` (older systems):

```bash
sudo yum install neurondb neuronagent neuronmcp
```

## Developer Guide

### Prerequisites

#### For DEB Repository Generation

```bash
# Ubuntu/Debian
sudo apt-get install -y dpkg-dev apt-utils gnupg2
```

#### For RPM Repository Generation

```bash
# RHEL/CentOS/Rocky
sudo dnf install -y createrepo_c gnupg2
```

### Setting Up GPG Keys

1. **Generate or import GPG key:**

```bash
cd packaging/repo
./setup-gpg.sh "Your Name" "your.email@example.com"
```

This will:
- Generate a new GPG key (or use existing)
- Export public keys to `repo/keys/`
- Display instructions for GitHub Secrets setup

2. **Export private key for GitHub Secrets:**

```bash
# Get your key ID from setup-gpg.sh output
gpg --armor --export-secret-keys YOUR_KEY_ID | base64 -w 0
```

3. **Add GitHub Secrets:**

Go to your repository settings → Secrets and variables → Actions, and add:

- `GPG_PRIVATE_KEY`: The base64-encoded output from above
- `GPG_KEY_ID`: Your GPG key ID (e.g., `ABC123DEF456`)
- `GPG_PASSPHRASE`: (Optional) Your GPG key passphrase if set

### Building and Publishing Packages

#### Manual Publishing

1. **Build packages and generate repository:**

```bash
cd packaging/repo
./publish.sh [VERSION] [GPG_KEY_ID]
```

Example:
```bash
./publish.sh 1.0.0 ABC123DEF456
```

This will:
- Build all DEB packages (places them in `repo/deb/pool/`)
- Build all RPM packages (places them in `repo/rpm/el*/x86_64/`)
- Generate DEB repository metadata
- Generate RPM repository metadata
- Sign everything with GPG (if key provided)

2. **Commit and push to GitHub:**

```bash
git add repo/
git commit -m "Publish packages version 1.0.0"
git push origin main
```

3. **Enable GitHub Pages:**

- Go to repository Settings → Pages
- Source: Deploy from a branch
- Branch: `main` (or your default branch)
- Folder: `/repo`
- Save

#### Automated Publishing (GitHub Actions)

The workflow (`.github/workflows/publish-packages.yml`) automatically:

- Triggers on manual workflow dispatch
- Builds packages
- Generates repository metadata
- Signs with GPG (if secrets configured)
- Commits and pushes to the main branch's `repo/` folder

**To publish manually via workflow:**

1. Go to Actions tab
2. Select "Publish Packages to Repository"
3. Click "Run workflow"
4. Optionally specify version
5. Run

### Individual Scripts

#### Generate DEB Repository Only

```bash
cd packaging/repo
./generate-deb-repo.sh [GPG_KEY_ID]
```

#### Generate RPM Repository Only

```bash
cd packaging/repo
./generate-rpm-repo.sh [GPG_KEY_ID]
```

#### Clean Repository

```bash
cd packaging/repo
./clean.sh
```

This removes all packages and metadata while preserving directory structure.

## Repository URL Structure

After enabling GitHub Pages, your repository will be available at:

```
https://USERNAME.github.io/REPOSITORY_NAME/repo/
```

For example:
- `https://github.com/yourorg/neurondb` → `https://yourorg.github.io/neurondb/repo/`

**Note:** Since we're serving from the `/repo` folder, the base URL includes `/repo/` in the path.

## Package Components

The repository includes three packages:

1. **neurondb**: PostgreSQL extension for vector search and ML
2. **neuronagent**: AI agent runtime system with REST API
3. **neuronmcp**: Model Context Protocol server

## Troubleshooting

### GPG Key Issues

**Error: "GPG key not found"**

- Ensure GPG key is imported: `gpg --list-secret-keys`
- Check that `GPG_KEY_ID` secret matches your key ID
- Verify key ID format: `gpg --list-secret-keys --keyid-format LONG`

**Error: "Bad signature"**

- Ensure public key is in `repo/keys/`
- Verify GPG key is exported correctly
- Check that users have added the key correctly

### Repository Not Updating

- Clear package manager cache: `apt-get clean` or `dnf clean all`
- Verify GitHub Pages is enabled and pointing to `/repo` folder on main branch
- Check that `repo/` directory is committed and pushed to main branch

### Missing Packages

- Ensure packages are built: `cd packaging/deb && ./build-all-deb.sh`
- Verify packages are in correct repo structure
- Check that repository metadata is regenerated after adding packages

## Security Notes

- Always use GPG signing for production repositories
- Keep GPG private keys secure (use GitHub Secrets)
- Regularly rotate GPG keys
- Verify package integrity before installation

## Support

For issues and questions:
- Documentation: https://www.neurondb.ai/docs
- Support: admin@neurondb.com

