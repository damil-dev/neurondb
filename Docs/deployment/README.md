# Deployment Guide

Deployment options and strategies for the NeuronDB ecosystem.

## Deployment Methods

### Docker (Recommended)

The easiest and most consistent deployment method. See [Docker Deployment Guide](docker.md) for detailed instructions.

**Advantages:**
- Consistent environments across development and production
- Easy to scale and manage
- Isolated dependencies
- Quick setup and teardown

### Source Build

Build and install from source for maximum control and customization.

**Advantages:**
- Full control over build configuration
- Optimized for specific hardware
- No container overhead

### Package Installation

Install using platform-specific packages (DEB/RPM).

**Advantages:**
- System integration
- Package management
- Easy updates

## Component Deployment

### NeuronDB

- **Docker**: See [NeuronDB Docker Guide](../../NeuronDB/docker/README.md)
- **Source**: See [NeuronDB Installation Guide](../../NeuronDB/INSTALL.md)
- **Packages**: See [Packaging Documentation](../../Docs/PACKEGE.md)

### NeuronAgent

- **Docker**: See [NeuronAgent Docker Guide](../../NeuronAgent/docker/README.md)
- **Source**: See [NeuronAgent Deployment Guide](../../NeuronAgent/docs/DEPLOYMENT.md)

### NeuronMCP

- **Docker**: See [NeuronMCP Docker Guide](../../NeuronMCP/docker/README.md)
- **Source**: See [NeuronMCP README](../../NeuronMCP/README.md)

### NeuronDesktop

- **Docker**: See [NeuronDesktop README](../../NeuronDesktop/README.md)
- **Source**: See [NeuronDesktop Deployment Guide](../../NeuronDesktop/docs/DEPLOYMENT.md)

## Deployment Strategies

### Single Host Deployment

All components run on the same host. Suitable for:
- Development environments
- Small-scale production
- Testing and evaluation

### Distributed Deployment

Components run on separate hosts. Suitable for:
- Production environments
- High availability
- Scalability requirements

### Container Orchestration

Deploy using Kubernetes or Docker Swarm. Suitable for:
- Large-scale production
- Auto-scaling requirements
- High availability

## Production Checklist

- [ ] Configured environment variables
- [ ] Set up database backups
- [ ] Implemented health checks
- [ ] Configured monitoring
- [ ] Set up log aggregation
- [ ] Configured auto-scaling (if needed)
- [ ] Tested disaster recovery
- [ ] Enabled SSL/TLS
- [ ] Configured firewall rules
- [ ] Set up secrets management

## Configuration

### Environment Variables

Each component requires specific environment variables. See component-specific documentation:

- [NeuronDB Configuration](../../NeuronDB/docs/configuration.md)
- [NeuronAgent Configuration](../../NeuronAgent/README.md#configuration)
- [NeuronMCP Configuration](../../NeuronMCP/README.md#configuration)
- [NeuronDesktop Configuration](../../NeuronDesktop/README.md#configuration)

### Network Configuration

- **Local Development**: Use `localhost` for all connections
- **Docker Network**: Use container names for service discovery
- **Production**: Use DNS names or IP addresses

## Monitoring

### Health Checks

- **NeuronDB**: `SELECT neurondb.version();`
- **NeuronAgent**: `curl http://localhost:8080/health`
- **NeuronMCP**: Check process status
- **NeuronDesktop**: `curl http://localhost:8081/health`

### Metrics

- **NeuronDB**: Built-in monitoring views and Prometheus metrics
- **NeuronAgent**: Health endpoint and metrics API
- **NeuronDesktop**: Metrics endpoint at `/api/v1/metrics`

## Security

### Best Practices

- Change default passwords
- Use secrets management
- Enable SSL/TLS
- Configure firewall rules
- Implement rate limiting
- Use API key authentication
- Enable audit logging

## Troubleshooting

### Common Issues

- **Connection Errors**: Verify database is running and connection parameters are correct
- **Port Conflicts**: Check if required ports are available
- **Build Errors**: Verify all prerequisites are installed
- **Performance Issues**: Check resource utilization and configuration

For detailed troubleshooting, see:
- [NeuronDB Troubleshooting](../../NeuronDB/docs/troubleshooting.md)
- [Official Documentation](https://www.neurondb.ai/docs/troubleshooting)

## Official Documentation

For comprehensive deployment guides:
**🌐 [https://www.neurondb.ai/docs/deployment](https://www.neurondb.ai/docs/deployment)**

