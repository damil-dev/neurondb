/*-------------------------------------------------------------------------
 *
 * middleware_setup.go
 *    Database operations
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/server/middleware_setup.go
 *
 *-------------------------------------------------------------------------
 */

package server

import (
	"github.com/neurondb/NeuronMCP/internal/audit"
	"github.com/neurondb/NeuronMCP/internal/config"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/middleware"
	"github.com/neurondb/NeuronMCP/internal/middleware/builtin"
)

/* setupBuiltInMiddleware registers all built-in middleware */
func setupBuiltInMiddleware(mgr *middleware.Manager, cfgMgr *config.ConfigManager, logger *logging.Logger) {
	loggingCfg := cfgMgr.GetLoggingConfig()
	serverCfg := cfgMgr.GetServerSettings()

	/* Correlation ID middleware (order: -1, runs first) */
	mgr.Register(builtin.NewCorrelationMiddleware(logger))

  /* Audit middleware (order: 0, runs early to capture all requests) */
	auditLogger := audit.NewLogger(logger)
	mgr.Register(builtin.NewAuditMiddleware(auditLogger))

  /* Authentication middleware (order: 0) */
	authConfig := &builtin.AuthConfig{
		Enabled: false, /* Disabled by default */
	}
	mgr.Register(builtin.NewAuthMiddleware(authConfig, logger))

  /* Scoped authentication middleware (order: 1, runs after auth) */
	/* Use default scope checker - can be replaced with custom implementation */
	scopeChecker := builtin.NewDefaultScopeChecker()
	mgr.Register(builtin.NewScopedAuthMiddleware(scopeChecker))

  /* Rate limiting middleware (order: 10) */
	rateLimitConfig := &builtin.RateLimitConfig{
		Enabled:        false, /* Disabled by default */
		RequestsPerMin: 60,
		BurstSize:      10,
		PerUser:        false,
		PerTool:        false,
	}
	mgr.Register(builtin.NewRateLimitMiddleware(rateLimitConfig, logger))

  /* Validation middleware (order: 1) */
	mgr.Register(builtin.NewValidationMiddleware())

  /* Idempotency middleware (order: 18, after validation, before logging) */
	mgr.Register(builtin.NewIdempotencyMiddleware(logger, true))

  /* Logging middleware (order: 2) */
	mgr.Register(builtin.NewLoggingMiddleware(
		logger,
		loggingCfg.EnableRequestLogging != nil && *loggingCfg.EnableRequestLogging,
		loggingCfg.EnableResponseLogging != nil && *loggingCfg.EnableResponseLogging,
	))

  /* Timeout middleware (order: 3) - only if timeout is configured */
	if serverCfg.Timeout != nil {
		mgr.Register(builtin.NewTimeoutMiddleware(serverCfg.GetTimeout(), logger))
	}

  /* Error handling middleware (order: 100) - always last */
	mgr.Register(builtin.NewErrorHandlingMiddleware(
		logger,
		loggingCfg.EnableErrorStack != nil && *loggingCfg.EnableErrorStack,
	))
}

