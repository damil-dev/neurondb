package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gorilla/mux"
	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/neurondb/NeuronDesktop/api/internal/auth"
	"github.com/neurondb/NeuronDesktop/api/internal/config"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/handlers"
	"github.com/neurondb/NeuronDesktop/api/internal/logging"
	"github.com/neurondb/NeuronDesktop/api/internal/middleware"
)

func main() {
	// Load configuration
	cfg := config.Load()
	
	// Initialize logger
	logger := logging.NewLogger(cfg.Logging.Level, cfg.Logging.Format, cfg.Logging.Output)
	logger.Info("Starting NeuronDesktop API server", nil)
	
	// Connect to database
	database, err := sql.Open("pgx", cfg.Database.DSN())
	if err != nil {
		logger.Error("Failed to open database", err, nil)
		os.Exit(1)
	}
	defer database.Close()
	
	// Configure connection pool
	database.SetMaxOpenConns(cfg.Database.MaxOpenConns)
	database.SetMaxIdleConns(cfg.Database.MaxIdleConns)
	database.SetConnMaxLifetime(cfg.Database.ConnMaxLifetime)
	
	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	if err := database.PingContext(ctx); err != nil {
		logger.Error("Failed to ping database", err, nil)
		os.Exit(1)
	}
	
	logger.Info("Connected to database", map[string]interface{}{
		"host": cfg.Database.Host,
		"port": cfg.Database.Port,
		"name": cfg.Database.Name,
	})
	
	// Initialize components
	queries := db.NewQueries(database)
	keyManager := auth.NewAPIKeyManager(queries)
	
	// Ensure default profile exists on startup
	initCtx, initCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer initCancel()
	
	defaultProfile, err := queries.GetDefaultProfile(initCtx)
	if err != nil || defaultProfile == nil {
		// Create default profile if it doesn't exist
		logger.Info("Creating default profile", nil)
		defaultProfile := &db.Profile{
			UserID:      "nbduser",
			Name:        "Default",
			NeuronDBDSN: "postgresql://nbduser@localhost:5432/neurondb",
			MCPConfig: map[string]interface{}{
				"command": "/Users/pgedge/pge/neurondb/NeuronMCP/bin/neurondb-mcp",
				"args":    []string{},
				"env": map[string]interface{}{
					"NEURONDB_HOST":     "localhost",
					"NEURONDB_PORT":     "5432",
					"NEURONDB_DATABASE": "neurondb",
					"NEURONDB_USER":     "nbduser",
				},
			},
			IsDefault: true,
		}
		
		if err := queries.CreateProfile(initCtx, defaultProfile); err != nil {
			logger.Error("Failed to create default profile", err, nil)
		} else {
			if err := queries.SetDefaultProfile(initCtx, defaultProfile.ID); err != nil {
				logger.Error("Failed to set default profile", err, nil)
			} else {
				logger.Info("Default profile created successfully", map[string]interface{}{
					"profile_id": defaultProfile.ID,
					"name":       defaultProfile.Name,
				})
				defaultProfile, _ = queries.GetDefaultProfile(initCtx)
			}
		}
	} else {
		logger.Info("Default profile already exists", map[string]interface{}{
			"profile_id": defaultProfile.ID,
			"name":       defaultProfile.Name,
		})
	}
	
	// Verify connections on startup
	if defaultProfile != nil {
		logger.Info("Verifying connections for default profile", map[string]interface{}{
			"profile_id": defaultProfile.ID,
		})
		
		// Verify PostgreSQL (NeuronDB) connection
		logger.Info("Verifying PostgreSQL (NeuronDB) connection", map[string]interface{}{
			"dsn": defaultProfile.NeuronDBDSN,
		})
		neurondbConn, err := sql.Open("pgx", defaultProfile.NeuronDBDSN)
		if err == nil {
			testCtx, testCancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer testCancel()
			if err := neurondbConn.PingContext(testCtx); err == nil {
				logger.Info("✓ PostgreSQL (NeuronDB) connection verified", map[string]interface{}{
					"dsn": defaultProfile.NeuronDBDSN,
				})
				// Test NeuronDB extension (optional - may not be installed)
				var version string
				if err := neurondbConn.QueryRowContext(testCtx, "SELECT neurondb.version()").Scan(&version); err == nil {
					logger.Info("✓ NeuronDB extension verified", map[string]interface{}{
						"version": version,
					})
				} else {
					logger.Info("⚠ NeuronDB extension not found (database may not have extension installed)", nil)
				}
			} else {
				logger.Info("⚠ PostgreSQL (NeuronDB) connection failed", map[string]interface{}{
					"error": err.Error(),
				})
			}
			neurondbConn.Close()
		} else {
			logger.Info("⚠ Failed to open PostgreSQL (NeuronDB) connection", map[string]interface{}{
				"error": err.Error(),
			})
		}
		
		// Verify MCP server connection
		if defaultProfile.MCPConfig != nil {
			logger.Info("Verifying MCP server connection", map[string]interface{}{
				"command": defaultProfile.MCPConfig["command"],
			})
			mcpManager := handlers.NewMCPManager(queries)
			client, err := mcpManager.GetClient(initCtx, defaultProfile.ID)
			if err == nil {
				// Try to list tools to verify connection
				tools, err := client.ListTools()
				if err == nil {
					logger.Info("✓ MCP server connection verified", map[string]interface{}{
						"tools_count": len(tools.Tools),
						"command":     defaultProfile.MCPConfig["command"],
					})
				} else {
					logger.Info("⚠ MCP server connected but tools listing failed", map[string]interface{}{
						"error": err.Error(),
					})
				}
			} else {
				logger.Info("⚠ MCP server connection failed", map[string]interface{}{
					"error": err.Error(),
				})
			}
		}
	}
	
	// Initialize handlers
	mcpManager := handlers.NewMCPManager(queries)
	mcpHandlers := handlers.NewMCPHandlers(mcpManager)
	neurondbHandlers := handlers.NewNeuronDBHandlers(queries)
	agentHandlers := handlers.NewAgentHandlers(queries)
	profileHandlers := handlers.NewProfileHandlers(queries)
	metricsHandlers := handlers.NewMetricsHandlers()
	
	// Initialize rate limiter
	rateLimiter := middleware.NewRateLimiter(100, 1*time.Minute)
	
	// Setup router
	router := mux.NewRouter()
	
	// Apply middleware
	router.Use(middleware.RecoveryMiddleware(logger))
	router.Use(middleware.LoggingMiddleware(logger))
	
	// Health check (no auth)
	router.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "ok",
			"service": "neurondesk-api",
			"timestamp": time.Now().UTC().Format(time.RFC3339),
		})
	}).Methods("GET")
	
	// API routes (with auth and rate limiting)
	apiRouter := router.PathPrefix("/api/v1").Subrouter()
	apiRouter.Use(auth.Middleware(keyManager))
	apiRouter.Use(middleware.RateLimitMiddleware(rateLimiter))
	
	// API Key routes
	apiKeyHandlers := handlers.NewAPIKeyHandlers(keyManager, queries)
	apiRouter.HandleFunc("/api-keys", apiKeyHandlers.GenerateAPIKey).Methods("POST")
	apiRouter.HandleFunc("/api-keys", apiKeyHandlers.ListAPIKeys).Methods("GET")
	apiRouter.HandleFunc("/api-keys/{id}", apiKeyHandlers.DeleteAPIKey).Methods("DELETE")
	
	// Profile routes
	apiRouter.HandleFunc("/profiles", profileHandlers.ListProfiles).Methods("GET")
	apiRouter.HandleFunc("/profiles", profileHandlers.CreateProfile).Methods("POST")
	apiRouter.HandleFunc("/profiles/{id}", profileHandlers.GetProfile).Methods("GET")
	apiRouter.HandleFunc("/profiles/{id}", profileHandlers.UpdateProfile).Methods("PUT")
	apiRouter.HandleFunc("/profiles/{id}", profileHandlers.DeleteProfile).Methods("DELETE")
	
	// MCP routes
	apiRouter.HandleFunc("/mcp/connections", mcpHandlers.ListConnections).Methods("GET")
	apiRouter.HandleFunc("/mcp/test", mcpHandlers.TestMCPConfig).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/tools", mcpHandlers.ListTools).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/tools/call", mcpHandlers.CallTool).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/ws", mcpHandlers.MCPWebSocket).Methods("GET")
	
	// NeuronDB routes
	apiRouter.HandleFunc("/profiles/{profile_id}/neurondb/collections", neurondbHandlers.ListCollections).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/neurondb/search", neurondbHandlers.Search).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/neurondb/sql", neurondbHandlers.ExecuteSQL).Methods("POST")
	
	// Agent routes
	apiRouter.HandleFunc("/agent/test", agentHandlers.TestAgentConfig).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents", agentHandlers.ListAgents).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents", agentHandlers.CreateAgent).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents/{agent_id}", agentHandlers.GetAgent).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/models", agentHandlers.ListModels).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/sessions", agentHandlers.CreateSession).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/sessions/{session_id}/messages", agentHandlers.SendMessage).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/ws", agentHandlers.AgentWebSocket).Methods("GET")
	
	// Model configuration routes
	modelConfigHandlers := handlers.NewModelConfigHandlers(queries)
	apiRouter.HandleFunc("/profiles/{profile_id}/models", modelConfigHandlers.ListModelConfigs).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/models", modelConfigHandlers.CreateModelConfig).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/models/{id}", modelConfigHandlers.UpdateModelConfig).Methods("PUT")
	apiRouter.HandleFunc("/profiles/{profile_id}/models/{id}", modelConfigHandlers.DeleteModelConfig).Methods("DELETE")
	apiRouter.HandleFunc("/profiles/{profile_id}/models/default", modelConfigHandlers.GetDefaultModelConfig).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/models/{id}/set-default", modelConfigHandlers.SetDefaultModelConfig).Methods("POST")
	
	// Metrics endpoints (no rate limit)
	apiRouter.HandleFunc("/metrics", metricsHandlers.GetMetrics).Methods("GET")
	apiRouter.HandleFunc("/metrics/reset", metricsHandlers.ResetMetrics).Methods("POST")
	
	// CORS middleware
	router.Use(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			origin := r.Header.Get("Origin")
			allowed := false
			for _, allowedOrigin := range cfg.CORS.AllowedOrigins {
				if allowedOrigin == "*" || allowedOrigin == origin {
					allowed = true
					break
				}
			}
			if allowed {
				w.Header().Set("Access-Control-Allow-Origin", origin)
			}
			
			w.Header().Set("Access-Control-Allow-Methods", joinStrings(cfg.CORS.AllowedMethods, ", "))
			w.Header().Set("Access-Control-Allow-Headers", joinStrings(cfg.CORS.AllowedHeaders, ", "))
			w.Header().Set("Access-Control-Allow-Credentials", "true")
			
			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}
			
			next.ServeHTTP(w, r)
		})
	})
	
	// Create HTTP server
	addr := cfg.Server.Host + ":" + cfg.Server.Port
	srv := &http.Server{
		Addr:         addr,
		Handler:      router,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
	}
	
	// Start server in goroutine
	go func() {
		logger.Info("Server starting", map[string]interface{}{
			"address": addr,
		})
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Error("Server failed", err, nil)
			os.Exit(1)
		}
	}()
	
	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	
	logger.Info("Shutting down server", nil)
	
	// Graceful shutdown
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()
	
	if err := srv.Shutdown(shutdownCtx); err != nil {
		logger.Error("Server shutdown failed", err, nil)
	}
	
	logger.Info("Server stopped", nil)
}

func joinStrings(strs []string, sep string) string {
	if len(strs) == 0 {
		return ""
	}
	result := strs[0]
	for i := 1; i < len(strs); i++ {
		result += sep + strs[i]
	}
	return result
}

