# Observability Setup Guide

## Overview

This guide explains how to set up end-to-end observability for the NeuronDB ecosystem using Prometheus, OpenTelemetry, and Grafana.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Services   │────▶│ Prometheus   │────▶│  Grafana    │
│ (Metrics)   │     │  (Scrape)    │     │ (Dashboards)│
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ OpenTelemetry │
                    │   (Tracing)   │
                    └──────────────┘
```

## Components

### 1. Prometheus Metrics

All services expose Prometheus metrics at `/metrics`:

- **NeuronAgent**: `http://localhost:8080/metrics`
- **NeuronDesktop API**: `http://localhost:8081/metrics`
- **NeuronDB**: Custom metrics via PostgreSQL extension

### 2. OpenTelemetry Tracing

Distributed tracing across all components:
- Request correlation IDs
- Span propagation
- Trace export to Jaeger/OTLP

### 3. Grafana Dashboards

Pre-built dashboards for:
- System health
- Performance metrics
- Error rates
- Database metrics

## Setup

### Step 1: Start Prometheus

```bash
# Using Docker Compose
docker compose -f docker-compose.observability.yml up -d prometheus
```

### Step 2: Start Grafana

```bash
docker compose -f docker-compose.observability.yml up -d grafana
```

### Step 3: Configure Services

Set environment variables:
```bash
export ENABLE_METRICS=true
export METRICS_PORT=9090
export ENABLE_TRACING=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

### Step 4: Import Dashboards

1. Access Grafana: `http://localhost:3001`
2. Import dashboards from `grafana/dashboards/`
3. Configure data sources (Prometheus, Jaeger)

## Metrics

### NeuronAgent Metrics

- `neurondb_agent_requests_total` - Total requests
- `neurondb_agent_request_duration_seconds` - Request duration
- `neurondb_agent_agents_active` - Active agents
- `neurondb_agent_sessions_active` - Active sessions
- `neurondb_agent_tool_calls_total` - Tool calls
- `neurondb_agent_errors_total` - Error count

### NeuronDesktop Metrics

- `neurondesktop_requests_total` - Total requests
- `neurondesktop_request_duration_seconds` - Request duration
- `neurondesktop_profiles_active` - Active profiles
- `neurondesktop_errors_total` - Error count

### NeuronDB Metrics

- `neurondb_queries_total` - Total queries
- `neurondb_query_duration_seconds` - Query duration
- `neurondb_indexes_total` - Total indexes
- `neurondb_vectors_stored` - Vectors stored
- `neurondb_embedding_requests_total` - Embedding requests

## Tracing

### Trace Context Propagation

All services propagate trace context via headers:
- `traceparent` (W3C Trace Context)
- `tracestate` (W3C Trace State)

### Example Trace

```
HTTP Request → NeuronDesktop API
  ├─ SQL Query → NeuronDB
  ├─ Agent Call → NeuronAgent
  │   ├─ Tool Execution
  │   └─ LLM Call
  └─ Response
```

## Dashboards

### System Health Dashboard

- Service uptime
- Health check status
- Error rates
- Request rates

### Performance Dashboard

- Request latency (p50, p95, p99)
- Throughput
- Database query performance
- Vector search performance

### Database Dashboard

- Connection pool stats
- Query performance
- Index health
- Vector operations

## Alerts

### Critical Alerts

- Service down
- High error rate (> 5%)
- High latency (p95 > 1s)
- Database connection failures

### Warning Alerts

- Elevated error rate (> 1%)
- Slow queries (> 500ms)
- High memory usage (> 80%)

## Runbooks

See `runbooks/` directory for operational runbooks:
- `troubleshooting.md` - Common issues
- `performance-tuning.md` - Performance optimization
- `incident-response.md` - Incident handling

## Best Practices

1. **Set appropriate scrape intervals**: 15-30 seconds
2. **Retain metrics**: 30 days for metrics, 7 days for traces
3. **Use labels wisely**: Don't create high cardinality
4. **Monitor cardinality**: Watch for label explosion
5. **Set up alerts**: But avoid alert fatigue

## Troubleshooting

### Metrics Not Appearing

1. Check service is exposing `/metrics`
2. Verify Prometheus can reach service
3. Check Prometheus targets page
4. Review service logs

### Traces Missing

1. Verify OTEL exporter endpoint
2. Check trace context propagation
3. Review OpenTelemetry logs
4. Verify sampling configuration

### High Cardinality

1. Review label usage
2. Use aggregation where possible
3. Consider reducing label dimensions
4. Monitor Prometheus memory usage

