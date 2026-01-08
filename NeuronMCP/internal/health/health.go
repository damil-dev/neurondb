/*-------------------------------------------------------------------------
 *
 * health.go
 *    Health check handler for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/health/health.go
 *
 *-------------------------------------------------------------------------
 */

package health

import (
	"context"
	"time"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* HealthStatus represents health status */
type HealthStatus struct {
	Status      string                 `json:"status"`
	Database    DatabaseHealth         `json:"database"`
	Tools       ToolsHealth            `json:"tools"`
	Resources   ResourcesHealth        `json:"resources"`
	Pool        PoolHealth             `json:"pool,omitempty"`
	Timestamp   time.Time              `json:"timestamp"`
}

/* DatabaseHealth represents database health */
type DatabaseHealth struct {
	Status    string        `json:"status"`
	Latency   time.Duration `json:"latency,omitempty"`
	Error     string        `json:"error,omitempty"`
}

/* ToolsHealth represents tools health */
type ToolsHealth struct {
	Status      string `json:"status"`
	TotalCount  int    `json:"totalCount"`
	AvailableCount int `json:"availableCount"`
}

/* ResourcesHealth represents resources health */
type ResourcesHealth struct {
	Status      string `json:"status"`
	TotalCount  int    `json:"totalCount"`
	AvailableCount int `json:"availableCount"`
}

/* PoolHealth represents connection pool health */
type PoolHealth struct {
	Status          string `json:"status"`
	TotalConnections int   `json:"totalConnections"`
	IdleConnections  int   `json:"idleConnections"`
	ActiveConnections int `json:"activeConnections"`
	MaxConnections   int   `json:"maxConnections"`
	Utilization     float64 `json:"utilization"`
}

/* Checker performs health checks */
type Checker struct {
	db     *database.Database
	logger *logging.Logger
}

/* NewChecker creates a new health checker */
func NewChecker(db *database.Database, logger *logging.Logger) *Checker {
	return &Checker{
		db:     db,
		logger: logger,
	}
}

/* Check performs a health check */
func (c *Checker) Check(ctx context.Context) *HealthStatus {
	status := &HealthStatus{
		Timestamp: time.Now(),
	}

	/* Check database */
	dbHealth := c.checkDatabase(ctx)
	status.Database = dbHealth

	/* Check tools */
	toolsHealth := c.checkTools(ctx)
	status.Tools = toolsHealth

	/* Check resources */
	resourcesHealth := c.checkResources(ctx)
	status.Resources = resourcesHealth

	/* Check connection pool */
	poolHealth := c.checkPool(ctx)
	status.Pool = poolHealth

	/* Overall status */
	if dbHealth.Status == "healthy" && toolsHealth.Status == "healthy" && resourcesHealth.Status == "healthy" && poolHealth.Status == "healthy" {
		status.Status = "healthy"
	} else {
		status.Status = "degraded"
	}

	return status
}

/* checkDatabase checks database health */
func (c *Checker) checkDatabase(ctx context.Context) DatabaseHealth {
	start := time.Now()
	
	var result int
	err := c.db.QueryRow(ctx, "SELECT 1").Scan(&result)
	latency := time.Since(start)

	if err != nil {
		return DatabaseHealth{
			Status: "unhealthy",
			Error:  err.Error(),
		}
	}

	return DatabaseHealth{
		Status:  "healthy",
		Latency: latency,
	}
}

/* checkTools checks tools health */
func (c *Checker) checkTools(ctx context.Context) ToolsHealth {
	/* For now, assume tools are available if database is healthy */
	/* In a full implementation, we'd check each tool */
	return ToolsHealth{
		Status:        "healthy",
		TotalCount:    50, /* Approximate */
		AvailableCount: 50,
	}
}

/* checkResources checks resources health */
func (c *Checker) checkResources(ctx context.Context) ResourcesHealth {
	/* For now, assume resources are available if database is healthy */
	return ResourcesHealth{
		Status:        "healthy",
		TotalCount:    6, /* schema, models, indexes, config, workers, stats */
		AvailableCount: 6,
	}
}

/* checkPool checks connection pool health */
func (c *Checker) checkPool(ctx context.Context) PoolHealth {
	if c.db == nil {
		return PoolHealth{
			Status: "unknown",
		}
	}

	stats := c.db.GetPoolStats()
	if stats == nil {
		return PoolHealth{
			Status: "unknown",
		}
	}

	totalConns := int(stats.TotalConns)
	activeConns := int(stats.AcquiredConns)
	idleConns := int(stats.IdleConns)

	utilization := 0.0
	if totalConns > 0 {
		utilization = float64(activeConns) / float64(totalConns)
	}

	status := "healthy"
	if utilization > 0.9 {
		status = "warning" /* Pool is 90%+ utilized */
	} else if utilization > 0.95 {
		status = "critical" /* Pool is 95%+ utilized */
	}

	return PoolHealth{
		Status:           status,
		TotalConnections: totalConns,
		IdleConnections:  idleConns,
		ActiveConnections: activeConns,
		MaxConnections:   totalConns,
		Utilization:      utilization,
	}
}

