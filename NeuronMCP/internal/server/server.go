/*-------------------------------------------------------------------------
 *
 * server.go
 *    Database operations
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/server/server.go
 *
 *-------------------------------------------------------------------------
 */

package server

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/batch"
	"github.com/neurondb/NeuronMCP/internal/config"
	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/health"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/middleware"
	"github.com/neurondb/NeuronMCP/internal/progress"
	"github.com/neurondb/NeuronMCP/internal/prompts"
	"github.com/neurondb/NeuronMCP/internal/resources"
	"github.com/neurondb/NeuronMCP/internal/sampling"
	"github.com/neurondb/NeuronMCP/internal/tools"
	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* Server is the main MCP server */
type Server struct {
	mcpServer          *mcp.Server
	db                 *database.Database
	config             *config.ConfigManager
	logger             *logging.Logger
	middleware         *middleware.Manager
	toolRegistry       *tools.ToolRegistry
	resources          *resources.Manager
	prompts            *prompts.Manager
	sampling           *sampling.Manager
	health             *health.Checker
	progress           *progress.Tracker
	batch              *batch.Processor
	capabilitiesManager *CapabilitiesManager
}

/* NewServer creates a new server */
func NewServer() (*Server, error) {
	return NewServerWithConfig("")
}

/* NewServerWithConfig creates a new server with a specific config path */
func NewServerWithConfig(configPath string) (*Server, error) {
	cfgMgr := config.NewConfigManager()
	_, err := cfgMgr.Load(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	logger := logging.NewLogger(cfgMgr.GetLoggingConfig())

	db := database.NewDatabase()
  /* Log database config for debugging */
	dbCfg := cfgMgr.GetDatabaseConfig()
	logger.Info("Database configuration", map[string]interface{}{
		"host":     dbCfg.GetHost(),
		"port":     dbCfg.GetPort(),
		"database": dbCfg.GetDatabase(),
		"user":     dbCfg.GetUser(),
		"has_password": dbCfg.Password != nil && *dbCfg.Password != "",
	})
	
  /* Try to connect, but don't fail server startup if it fails */
  /* The server can start and tools will fail gracefully with proper error messages */
	if err := db.Connect(dbCfg); err != nil {
		logger.Warn("Failed to connect to database at startup", map[string]interface{}{
			"error": err.Error(),
			"host":     dbCfg.GetHost(),
			"port":     dbCfg.GetPort(),
			"database": dbCfg.GetDatabase(),
			"user":     dbCfg.GetUser(),
			"note":  "Server will start but tools may fail. Database connection will be retried on first use.",
		})
   /* Continue anyway - tools will handle connection errors gracefully */
	} else {
		logger.Info("Connected to database", map[string]interface{}{
			"host":     dbCfg.GetHost(),
			"database": dbCfg.GetDatabase(),
			"user":     dbCfg.GetUser(),
		})
	}

	serverSettings := cfgMgr.GetServerSettings()
	
	/* Get max request size from config */
	maxRequestSize := int64(0) /* Default: unlimited */
	if serverSettings.MaxRequestSize != nil && *serverSettings.MaxRequestSize > 0 {
		maxRequestSize = int64(*serverSettings.MaxRequestSize)
	}
	
	mcpServer := mcp.NewServerWithMaxRequestSize(serverSettings.GetName(), serverSettings.GetVersion(), maxRequestSize)

	mwManager := middleware.NewManager(logger)
	setupBuiltInMiddleware(mwManager, cfgMgr, logger)

	toolRegistry := tools.NewToolRegistry(db, logger)
	tools.RegisterAllTools(toolRegistry, db, logger)

	capabilitiesManager := NewCapabilitiesManager(serverSettings.GetName(), serverSettings.GetVersion(), toolRegistry)

	resourcesManager := resources.NewManager(db)
	resources.RegisterAllResources(resourcesManager, db)
	promptsManager := prompts.NewManager(db, logger)
	samplingManager := sampling.NewManager(db, logger)
	samplingManager.SetToolRegistry(toolRegistry) /* Enable tool calling in sampling */
	healthChecker := health.NewChecker(db, logger)
	progressTracker := progress.NewTracker()
	batchProcessor := batch.NewProcessor(db, toolRegistry, logger)

	s := &Server{
		mcpServer:          mcpServer,
		db:                 db,
		config:             cfgMgr,
		logger:             logger,
		middleware:         mwManager,
		toolRegistry:       toolRegistry,
		resources:          resourcesManager,
		prompts:            promptsManager,
		sampling:           samplingManager,
		health:             healthChecker,
		progress:           progressTracker,
		batch:              batchProcessor,
		capabilitiesManager: capabilitiesManager,
	}

	s.setupHandlers()
	s.setupExperimentalHandlers()

	return s, nil
}

func (s *Server) setupHandlers() {
	s.setupToolHandlers()
	s.setupResourceHandlers()
	s.setupPromptHandlers()
	s.setupSamplingHandlers()
	s.setupHealthHandlers()
	s.setupProgressHandlers()
	s.setupBatchHandlers()
	
  /* Set capabilities using capabilities manager */
	if s.capabilitiesManager != nil {
		caps := s.capabilitiesManager.GetServerCapabilities()
		s.mcpServer.SetCapabilities(caps)
	} else {
		/* Fallback to empty capabilities */
		s.mcpServer.SetCapabilities(mcp.ServerCapabilities{
			Tools: mcp.ToolsCapability{
				ListChanged: false,
			},
			Resources: mcp.ResourcesCapability{
				Subscribe:   false,
				ListChanged: false,
			},
			Prompts:   make(map[string]interface{}),
			Sampling:  make(map[string]interface{}),
		})
	}
}

/* Start starts the server */
func (s *Server) Start(ctx context.Context) error {
	s.logger.Info("Starting Neurondb MCP server", nil)
  /* Run the MCP server - this will block until context is cancelled or EOF */
	err := s.mcpServer.Run(ctx)
	if err != nil && err != context.Canceled {
		s.logger.Warn("MCP server stopped", map[string]interface{}{
			"error": err.Error(),
		})
	}
	return err
}

/* Stop stops the server gracefully */
func (s *Server) Stop() error {
	s.logger.Info("Stopping Neurondb MCP server", nil)
	
	/* Wait for in-flight requests to complete */
	/* Note: In a production environment, you might want to:
	 * - Set a timeout for graceful shutdown
	 * - Track in-flight requests
	 * - Wait for them to complete
	 * - Force shutdown after timeout
	 */
	
	/* Close database connections */
	if s.db != nil {
		s.db.Close()
	}
	
	/* Close any other resources */
	/* TODO: Close HTTP server if running */
	/* TODO: Close SSE connections */
	/* TODO: Cleanup goroutines */
	
	s.logger.Info("NeuronMCP server stopped", nil)
	return nil
}

