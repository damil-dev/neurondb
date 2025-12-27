# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of NeuronDB seriously. If you believe you have found a
security vulnerability, please report it to us as described below.

### Please do NOT report security vulnerabilities through public GitHub issues.

Instead, please report them via email to:

**support@neurondb.ai**

### What to Include

Please include the following information in your report:

- **Type of issue** (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- **Full paths of source file(s) related to the manifestation of the issue**
- **The location of the affected source code** (tag/branch/commit or direct URL)
- **Any special configuration required to reproduce the issue**
- **Step-by-step instructions to reproduce the issue**
- **Proof-of-concept or exploit code** (if possible)
- **Impact of the issue**, including how an attacker might exploit the issue

This information will help us triage your report more quickly.

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your report within 48 hours
- **Initial Assessment**: We will provide an initial assessment within 7 days
- **Updates**: We will keep you informed of our progress toward resolving the issue
- **Resolution**: We will work with you to understand and resolve the issue quickly

### Disclosure Policy

We follow responsible disclosure practices:

- Please do not publicly disclose the vulnerability until we have released a fix
- We will credit you in the security advisory (unless you prefer to remain anonymous)
- We will work with you to coordinate disclosure timing

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

Thank you for helping keep NeuronDB and our users safe!

