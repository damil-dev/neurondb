# Security Policy

## Supported Versions

### Version Support Matrix

| Component | Version | Supported | Security Updates | End of Life |
|-----------|---------|-----------|------------------|-------------|
| **NeuronDB** | 1.0.x | ✅ Yes | ✅ Yes | - |
| **NeuronDB** | 0.x.x | ❌ No | ❌ No | - |
| **NeuronAgent** | 1.0.x | ✅ Yes | ✅ Yes | - |
| **NeuronAgent** | 0.x.x | ❌ No | ❌ No | - |
| **NeuronMCP** | 1.0.x | ✅ Yes | ✅ Yes | - |
| **NeuronMCP** | 0.x.x | ❌ No | ❌ No | - |
| **NeuronDesktop** | 1.0.x | ✅ Yes | ✅ Yes | - |
| **NeuronDesktop** | 0.x.x | ❌ No | ❌ No | - |

### PostgreSQL Version Compatibility

| NeuronDB Version | PostgreSQL 16 | PostgreSQL 17 | PostgreSQL 18 |
|------------------|---------------|---------------|---------------|
| 1.0.x | ✅ Supported | ✅ Supported | ✅ Supported |
| 0.x.x | ❌ No | ❌ No | ❌ No |

### Support Policy

- **Current Release:** Full support with security updates and bug fixes
- **Previous Minor Release:** Security updates only (for 6 months after current release)
- **Older Versions:** No support, upgrade recommended
- **Pre-release/Beta:** No security support, use at own risk

**Note:** We recommend always using the latest stable release for security updates and bug fixes.

### Supported Versions Matrix

| Component | Version | Security Updates Until | EOL Date |
|-----------|---------|----------------------|----------|
| NeuronDB | 1.0.x | Current + 6 months | TBD |
| NeuronDB | 0.x.x | ❌ EOL | 2024-12-31 |
| NeuronAgent | 1.0.x | Current + 6 months | TBD |
| NeuronAgent | 0.x.x | ❌ EOL | 2024-12-31 |
| NeuronMCP | 1.0.x | Current + 6 months | TBD |
| NeuronMCP | 0.x.x | ❌ EOL | 2024-12-31 |
| NeuronDesktop | 1.0.x | Current + 6 months | TBD |
| NeuronDesktop | 0.x.x | ❌ EOL | 2024-12-31 |

**PostgreSQL Version Support:**
- PostgreSQL 16.x: Supported until PostgreSQL 16 EOL
- PostgreSQL 17.x: Supported until PostgreSQL 17 EOL
- PostgreSQL 18.x: Supported until PostgreSQL 18 EOL

See [COMPATIBILITY.md](../COMPATIBILITY.md) for detailed version compatibility information.

## Responsible Disclosure Process

We take the security of NeuronDB seriously. We follow a coordinated responsible disclosure process to ensure vulnerabilities are properly handled and users are protected.

### Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues, discussions, or other public channels.**

Instead, please report them securely via email to:

**security@neurondb.ai** (preferred)  
**support@neurondb.ai** (alternative)

### What to Include in Your Report

Please provide as much information as possible:

- **Type of issue** (e.g., buffer overflow, SQL injection, cross-site scripting, authentication bypass, etc.)
- **Affected components** (NeuronDB extension, NeuronAgent, NeuronMCP, NeuronDesktop)
- **Severity assessment** (Critical, High, Medium, Low) based on CVSS scoring if applicable
- **Full paths of source file(s)** related to the manifestation of the issue
- **The location of the affected source code** (tag/branch/commit or direct URL)
- **Version information** (which version(s) are affected)
- **Any special configuration** required to reproduce the issue
- **Step-by-step instructions** to reproduce the issue
- **Proof-of-concept or exploit code** (if possible, please use base64 encoding or password-protected archive)
- **Impact of the issue**, including how an attacker might exploit the issue
- **Potential mitigations** or workarounds (if known)

This information will help us triage and resolve your report more quickly.

### Responsible Disclosure Timeline

We follow a coordinated disclosure process with explicit SLA commitments:

1. **Initial Report** (Day 0)
   - Vulnerability reported via secure email
   - **SLA:** Confirmation of receipt within **48 hours**

2. **Initial Assessment** (Days 1-7)
   - We analyze the report and verify the vulnerability
   - We determine severity and affected versions
   - We provide initial assessment and acknowledgment
   - **SLA:** Initial assessment within **7 days**

3. **Resolution Development** (Days 8-90)
   - We develop and test security fixes
   - We coordinate with reporter on testing (if appropriate)
   - Regular updates on progress (at least every 30 days)
   - **SLA:** 
     - Critical: Fix target **30 days**
     - High: Fix target **60 days**
     - Medium: Fix target **90 days**
     - Low: Fix target **120 days**

4. **Fix Release** (Target: Based on severity)
   - Security fix released in a patch version
   - Security advisory published
   - **SLA:** Release within target timeframe based on severity

5. **Public Disclosure** (After fix is available)
   - Public announcement (with reporter credit unless anonymous)
   - CVE assignment (if applicable)
   - Release notes updated
   - **SLA:** Public disclosure within **7 days** of fix release

**Note:** Timelines may vary based on vulnerability severity and complexity. Critical vulnerabilities will be prioritized for faster resolution.

### CVE Process

**CVE Assignment:**
- Critical and High severity vulnerabilities will receive CVE assignments
- CVEs are requested through MITRE or GitHub Security Advisories
- CVE assignment typically occurs within 7 days of fix release

**CVE Publication:**
- CVEs are published in GitHub Security Advisories
- Included in release notes for the patch version
- Listed in [SECURITY.md](SECURITY.md) changelog

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your report within 48 hours
- **Initial Assessment**: We will provide an initial assessment within 7 days
- **Regular Updates**: We will keep you informed of our progress (at least every 30 days)
- **Coordinated Disclosure**: We will work with you to coordinate public disclosure timing
- **Credit**: We will credit you in the security advisory (unless you prefer to remain anonymous)

### Our Commitment

- We take all security reports seriously
- We treat reporters with respect and professionalism
- We keep reporters informed throughout the process
- We work to resolve issues as quickly as possible
- We provide clear timelines and expectations
- We credit researchers appropriately (unless they prefer anonymity)

### Disclosure Policy

We follow responsible disclosure practices:

- **Do not publicly disclose** the vulnerability until we have released a fix and published a security advisory
- **Do not exploit** the vulnerability beyond what is necessary to demonstrate it
- **Do not access or modify** data that does not belong to you
- We will work with you to coordinate disclosure timing
- We will credit you in the security advisory (unless you prefer to remain anonymous)
- We will assign CVEs for significant vulnerabilities (if applicable)

### Security Researcher Guidelines

We appreciate security research and welcome responsible disclosure:

1. **Scope**: Only test against systems you own or have explicit permission to test
2. **Reporting**: Report vulnerabilities through our secure email channels
3. **Timeline**: Allow us time to fix the issue before public disclosure
4. **Privacy**: Do not access or exfiltrate user data
5. **Impact**: Minimize any potential disruption to services
6. **Legal**: Ensure your testing is legal and authorized

### Bug Bounty Program

We do not currently operate a formal bug bounty program. However, we appreciate security research and may recognize significant contributions on a case-by-case basis. Please contact us at security@neurondb.ai for more information.

## Security Best Practices

When using NeuronDB in production:

1. **Access Control**: Use PostgreSQL role-based access control (RBAC) to limit access
2. **Encryption**: Enable SSL/TLS for all database connections
3. **Updates**: Keep PostgreSQL and NeuronDB up to date with the latest security patches
4. **Auditing**: Enable query logging for sensitive operations
5. **Input Validation**: Validate all user inputs before vector operations
6. **Network Security**: Restrict network access to the database server
7. **Credentials**: Use secure credential management for HTTP/LLM integrations

## Known Security Considerations

- Vector encryption features are for demonstration purposes; use PostgreSQL's native encryption for production
- HTTP/LLM integration requires secure credential management
- Shared memory buffers should be sized appropriately to prevent DoS attacks
- Always run NeuronDB with the minimum required privileges

## Security Updates

Security updates will be announced through:

- GitHub Security Advisories
- Release notes
- Email notifications (if you subscribe to security updates)
- Security advisories on our website

## Dependency Scanning

We maintain security through automated dependency scanning in our CI/CD pipeline.

### Automated Scanning

All dependencies are automatically scanned for known vulnerabilities:

- **Dependencies**: Go modules, Python packages, Node.js packages, C/C++ libraries
- **Frequency**: On every commit and PR
- **Tools**: 
  - GitHub Dependabot (for Go, Python, Node.js)
  - Trivy (container image scanning)
  - Snyk (additional dependency scanning)
  - OSV Scanner (Open Source Vulnerabilities)

### Scanning Process

1. **Continuous Monitoring**: Dependencies are scanned on every build
2. **Vulnerability Detection**: Known CVEs are matched against dependency versions
3. **Automated Alerts**: Security team is notified of high/critical vulnerabilities
4. **Remediation**: Vulnerable dependencies are updated or replaced
5. **Verification**: Security tests verify fixes don't introduce regressions

### Dependency Update Policy

- **Critical/High Severity**: Updated within 7 days or included in next patch release (whichever is sooner)
- **Medium Severity**: Updated within 30 days or included in next minor release
- **Low Severity**: Included in next scheduled release

### Manual Scanning

You can also scan dependencies locally:

```bash
# Go dependencies
go list -json -m all | nancy sleuth

# Python dependencies (NeuronAgent examples)
pip-audit

# Node.js dependencies (NeuronMCP)
npm audit

# Container images
trivy image neurondb:latest
```

## Signed Releases

All NeuronDB releases are cryptographically signed to ensure authenticity and integrity.

### Signing Methods

We use multiple signing methods for maximum compatibility:

#### Sigstore (Recommended)

All releases are signed using Sigstore, providing:
- **Cosign signatures**: Cryptographically verifiable signatures
- **Transparency logs**: Public, auditable signature log
- **Fulcio certificates**: Short-lived certificates for signing
- **Rekor**: Immutable signature transparency log

**Verification:**
```bash
# Verify Docker image signature
cosign verify --certificate-identity=signer@neurondb.ai \
  --certificate-oidc-issuer=https://github.com/login/oauth \
  neurondb:1.0.0

# Verify release artifact signature
cosign verify-blob --certificate-identity=signer@neurondb.ai \
  --certificate-oidc-issuer=https://github.com/login/oauth \
  --signature neurondb-1.0.0.sig \
  neurondb-1.0.0.tar.gz
```

#### GPG Signatures

Legacy releases also include GPG signatures for compatibility.

**Public Key:**
- Key ID: `[TO BE ADDED]`
- Fingerprint: `[TO BE ADDED]`
- Download: `https://neurondb.ai/security/gpg-key.asc`

**Verification:**
```bash
# Import public key
gpg --import gpg-key.asc

# Verify signature
gpg --verify neurondb-1.0.0.tar.gz.asc neurondb-1.0.0.tar.gz
```

### What is Signed

- **Source code releases**: All tarballs and zip archives
- **Docker images**: All container images in public registries
- **Binary releases**: All pre-compiled binaries
- **Package repositories**: All DEB and RPM packages
- **Checksums**: SHA256 checksum files

### Release Signing Process

1. **Build**: Releases are built in secure, isolated environments
2. **Sign**: Artifacts are signed using Sigstore and GPG
3. **Publish**: Signed artifacts are published with checksums
4. **Verify**: Signatures are automatically verified in CI before publication
5. **Transparency**: All signatures are logged to public transparency logs

### Verifying Releases

Always verify signatures before installing or using releases:

```bash
# Download release
wget https://github.com/neurondb/NeurondB/releases/download/v1.0.0/neurondb-1.0.0.tar.gz
wget https://github.com/neurondb/NeurondB/releases/download/v1.0.0/neurondb-1.0.0.tar.gz.sig
wget https://github.com/neurondb/NeurondB/releases/download/v1.0.0/neurondb-1.0.0.tar.gz.sigstore

# Verify with Cosign (recommended)
cosign verify-blob --certificate-identity=signer@neurondb.ai \
  --certificate-oidc-issuer=https://github.com/login/oauth \
  --signature neurondb-1.0.0.tar.gz.sigstore \
  --bundle neurondb-1.0.0.tar.gz.bundle \
  neurondb-1.0.0.tar.gz

# Or verify with GPG
gpg --verify neurondb-1.0.0.tar.gz.sig neurondb-1.0.0.tar.gz
```

### Package Repository Signing

APT and YUM repositories are also signed:

**APT (Debian/Ubuntu):**
```bash
# Import GPG key
curl -fsSL https://neurondb.ai/repo/deb/KEY.gpg | sudo gpg --dearmor -o /usr/share/keyrings/neurondb.gpg

# Verify repository signature is valid
apt update
```

**YUM (RHEL/CentOS/Rocky):**
```bash
# Import GPG key
sudo rpm --import https://neurondb.ai/repo/rpm/RPM-GPG-KEY-neurondb

# Verify package signatures
rpm --checksig neurondb-1.0.0-1.x86_64.rpm
```

### Trust and Verification

- **Transparency**: All signatures are logged to public transparency logs (Sigstore Rekor)
- **Auditability**: Signing certificates and keys are publicly auditable
- **Non-repudiation**: Signatures provide cryptographic proof of origin
- **Integrity**: Signatures ensure artifacts have not been tampered with

## Additional Security Resources

- [Security Advisories](https://github.com/neurondb/NeurondB/security/advisories) - Published security advisories
- [Security Configuration Guide](https://www.neurondb.ai/docs/security) - Production security configuration
- [Reporting Issues](mailto:security@neurondb.ai) - Report security vulnerabilities

Thank you for helping keep NeuronDB and our users safe!

