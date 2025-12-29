package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gorilla/mux"
	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/neurondb/NeuronDesktop/api/internal/auth"
	"github.com/neurondb/NeuronDesktop/api/internal/auth/oidc"
	"github.com/neurondb/NeuronDesktop/api/internal/config"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/handlers"
	"github.com/neurondb/NeuronDesktop/api/internal/initialization"
	"github.com/neurondb/NeuronDesktop/api/internal/logging"
	"github.com/neurondb/NeuronDesktop/api/internal/middleware"
	"github.com/neurondb/NeuronDesktop/api/internal/session"
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
	keyManager := auth.NewAPIKeyManager(queries) // Keep for backwards compatibility if needed
	
	// Validate JWT secret if JWT mode is enabled
	if cfg.Auth.Mode == "jwt" || cfg.Auth.Mode == "hybrid" {
		if cfg.Auth.JWTSecret == "" {
			logger.Error("JWT_SECRET is required when using JWT authentication", fmt.Errorf("JWT_SECRET environment variable not set"), nil)
			os.Exit(1)
		}
	}
	
	// Initialize session manager
	sessionMgr := session.NewManager(
		database,
		cfg.Session.AccessTTL,
		cfg.Session.RefreshTTL,
		cfg.Session.CookieDomain,
		cfg.Session.CookieSecure,
		cfg.Session.CookieSameSite,
	)
	
	// Initialize OIDC provider if configured
	var oidcProvider *oidc.Provider
	var oidcHandlers *handlers.OIDCHandlers
	if cfg.Auth.Mode == "oidc" || cfg.Auth.Mode == "hybrid" {
		if cfg.Auth.OIDC.IssuerURL != "" {
			var err error
			oidcProvider, err = oidc.NewProvider(
				ctx,
				cfg.Auth.OIDC.IssuerURL,
				cfg.Auth.OIDC.ClientID,
				cfg.Auth.OIDC.ClientSecret,
				cfg.Auth.OIDC.RedirectURL,
				cfg.Auth.OIDC.Scopes,
			)
			if err != nil {
				logger.Error("Failed to initialize OIDC provider", err, nil)
				// Continue without OIDC if it fails
			} else {
				oidcHandlers = handlers.NewOIDCHandlers(oidcProvider, sessionMgr, queries, cfg.Auth.OIDC.IssuerURL)
				logger.Info("OIDC provider initialized", map[string]interface{}{
					"issuer": cfg.Auth.OIDC.IssuerURL,
				})
			}
		}
	}
	
	// Bootstrap application (admin user, default profile, schema, connections)
	initCtx, initCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer initCancel()
	
	bootstrap := initialization.NewBootstrap(queries, logger)
	if err := bootstrap.Initialize(initCtx); err != nil {
		logger.Error("Failed to bootstrap application", err, nil)
		// Continue anyway - some features may still work
	}
	
	// Initialize handlers
	authHandlers := handlers.NewAuthHandlers(queries)
	mcpManager := handlers.NewMCPManager(queries)
	mcpHandlers := handlers.NewMCPHandlers(mcpManager)
	neurondbHandlers := handlers.NewNeuronDBHandlers(queries, cfg.Security.EnableSQLConsole)
	agentHandlers := handlers.NewAgentHandlers(queries)
	profileHandlers := handlers.NewProfileHandlers(queries)
	metricsHandlers := handlers.NewMetricsHandlers()
	factoryHandlers := handlers.NewFactoryHandlers(queries)
	systemMetricsHandlers := handlers.NewSystemMetricsHandlers(logger)
	
	// Initialize rate limiter (increased for development - 10000 requests per minute)
	rateLimiter := middleware.NewRateLimiter(10000, 1*time.Minute)
	
	// Setup router
	router := mux.NewRouter()
	
	// Apply middleware (order matters)
	router.Use(middleware.RequestIDMiddleware())
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
	
	// System metrics endpoints (no auth, public)
	systemMetricsRouter := router.PathPrefix("/api/v1/system-metrics").Subrouter()
	systemMetricsRouter.HandleFunc("", systemMetricsHandlers.GetSystemMetrics).Methods("GET")
	systemMetricsRouter.HandleFunc("/ws", systemMetricsHandlers.SystemMetricsWebSocket).Methods("GET")
	
	// Database test handler (no auth required)
	databaseTestHandlers := handlers.NewDatabaseTestHandlers()
	
	// Auth routes (no auth required)
	authRouter := router.PathPrefix("/api/v1/auth").Subrouter()
	// Handle OPTIONS for CORS preflight on auth routes
	if cfg.Auth.EnableLocalAuth {
		authRouter.HandleFunc("/register", func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}
			authHandlers.Register(w, r)
		}).Methods("POST", "OPTIONS")
		authRouter.HandleFunc("/login", func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}
			authHandlers.Login(w, r)
		}).Methods("POST", "OPTIONS")
	}
	
	// OIDC routes
	if oidcHandlers != nil {
		authRouter.HandleFunc("/oidc/start", oidcHandlers.StartOIDCFlow).Methods("GET")
		authRouter.HandleFunc("/oidc/callback", oidcHandlers.OIDCCallback).Methods("GET")
		authRouter.HandleFunc("/refresh", oidcHandlers.RefreshToken).Methods("POST")
	}
	
	// Logout route (requires auth)
	authRouter.HandleFunc("/logout", func(w http.ResponseWriter, r *http.Request) {
		if oidcHandlers != nil {
			oidcHandlers.Logout(w, r)
		} else {
			// Fallback for JWT logout (just clear token on client)
			handlers.WriteSuccess(w, map[string]interface{}{"message": "logged out"}, http.StatusOK)
		}
	}).Methods("POST")
	
	// Database test route (no auth required)
	// Handle OPTIONS for CORS preflight
	router.HandleFunc("/api/v1/database/test", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		databaseTestHandlers.TestConnection(w, r)
	}).Methods("POST", "OPTIONS")
	
	// API routes (with auth and rate limiting)
	apiRouter := router.PathPrefix("/api/v1").Subrouter()
	
	// Use session middleware if OIDC is enabled, otherwise JWT
	if cfg.Auth.Mode == "oidc" && oidcHandlers != nil {
		apiRouter.Use(sessionMgr.SessionMiddleware())
	} else if cfg.Auth.Mode == "hybrid" {
		// Hybrid mode: try session first, fallback to JWT
		apiRouter.Use(func(next http.Handler) http.Handler {
			return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// Try session first
				if sessionMgr.GetAccessTokenFromRequest(r) != "" {
					sessionMgr.SessionMiddleware()(next).ServeHTTP(w, r)
					return
				}
				// Fallback to JWT
				auth.JWTMiddleware()(next).ServeHTTP(w, r)
			})
		})
	} else {
		// JWT only mode
		apiRouter.Use(auth.JWTMiddleware())
	}
	
	apiRouter.Use(middleware.RateLimitMiddleware(rateLimiter))
	
	// Current user endpoint
	apiRouter.HandleFunc("/auth/me", authHandlers.GetCurrentUser).Methods("GET")
	
	// API Key routes
	apiKeyHandlers := handlers.NewAPIKeyHandlers(keyManager, queries)
	apiRouter.HandleFunc("/api-keys", apiKeyHandlers.GenerateAPIKey).Methods("POST")
	apiRouter.HandleFunc("/api-keys", apiKeyHandlers.ListAPIKeys).Methods("GET")
	apiRouter.HandleFunc("/api-keys/{id}", apiKeyHandlers.DeleteAPIKey).Methods("DELETE")
	
	// Profile routes (with OPTIONS support for CORS)
	apiRouter.HandleFunc("/profiles", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.Method == "GET" {
			profileHandlers.ListProfiles(w, r)
		} else if r.Method == "POST" {
			profileHandlers.CreateProfile(w, r)
		}
	}).Methods("GET", "POST", "OPTIONS")
	apiRouter.HandleFunc("/profiles/{id}", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.Method == "GET" {
			profileHandlers.GetProfile(w, r)
		} else if r.Method == "PUT" {
			profileHandlers.UpdateProfile(w, r)
		} else if r.Method == "DELETE" {
			profileHandlers.DeleteProfile(w, r)
		}
	}).Methods("GET", "PUT", "DELETE", "OPTIONS")
	
	// MCP routes
	apiRouter.HandleFunc("/mcp/connections", mcpHandlers.ListConnections).Methods("GET")
	apiRouter.HandleFunc("/mcp/test", mcpHandlers.TestMCPConfig).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/tools", mcpHandlers.ListTools).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/tools/call", mcpHandlers.CallTool).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/ws", mcpHandlers.MCPWebSocket).Methods("GET")
	
	// MCP Chat Thread routes
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/threads", mcpHandlers.ListThreads).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/threads", mcpHandlers.CreateThread).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/threads/{thread_id}", mcpHandlers.GetThread).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/threads/{thread_id}", mcpHandlers.UpdateThread).Methods("PUT")
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/threads/{thread_id}", mcpHandlers.DeleteThread).Methods("DELETE")
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/threads/{thread_id}/messages", mcpHandlers.AddMessage).Methods("POST")
	
	// NeuronDB routes
	apiRouter.HandleFunc("/profiles/{profile_id}/neurondb/collections", neurondbHandlers.ListCollections).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/neurondb/search", neurondbHandlers.Search).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/neurondb/sql", neurondbHandlers.ExecuteSQL).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/neurondb/sql/execute", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		neurondbHandlers.ExecuteSQLFull(w, r)
	}).Methods("POST", "OPTIONS")
	
	// Agent routes
	apiRouter.HandleFunc("/agent/test", agentHandlers.TestAgentConfig).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents", agentHandlers.ListAgents).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents", agentHandlers.CreateAgent).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents/{agent_id}", agentHandlers.GetAgent).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/models", agentHandlers.ListModels).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/sessions", agentHandlers.CreateSession).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents/{agent_id}/sessions", agentHandlers.ListSessions).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/sessions/{session_id}", agentHandlers.GetSession).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/sessions/{session_id}/messages", agentHandlers.SendMessage).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/sessions/{session_id}/messages", agentHandlers.GetMessages).Methods("GET")
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
	
	// Factory endpoints
	apiRouter.HandleFunc("/factory/status", factoryHandlers.GetFactoryStatus).Methods("GET")
	apiRouter.HandleFunc("/factory/setup-state", factoryHandlers.GetSetupState).Methods("GET")
	apiRouter.HandleFunc("/factory/setup-state", factoryHandlers.SetSetupState).Methods("POST")
	
	// CORS handler wrapper
	//
	// Important: we wrap the router at the HTTP handler level (instead of router.Use),
	// so CORS headers and OPTIONS preflight responses work even when gorilla/mux would
	// otherwise return 404 for method-mismatches (e.g. OPTIONS on a GET-only route).
	corsHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check if this is a WebSocket upgrade request
		// WebSocket upgrades need direct access to the underlying connection (Hijacker interface)
		// so we bypass the CORS wrapper for WebSocket requests
		if r.Header.Get("Upgrade") == "websocket" {
			router.ServeHTTP(w, r)
			return
		}

		origin := r.Header.Get("Origin")
		allowed := false
		allowAll := false

		// Check if origin is allowed
		for _, allowedOrigin := range cfg.CORS.AllowedOrigins {
			if allowedOrigin == "*" {
				allowAll = true
				allowed = true
				break
			} else if allowedOrigin == origin {
				allowed = true
				break
			}
		}

		// Set CORS headers
		if allowed {
			// If "*" is allowed, use the actual origin (required when credentials are allowed)
			if allowAll && origin != "" {
				w.Header().Set("Access-Control-Allow-Origin", origin)
			} else if allowAll {
				// If "*" is allowed but no origin header, allow all (but can't use credentials)
				w.Header().Set("Access-Control-Allow-Origin", "*")
				// Don't set credentials when using wildcard
			} else if origin != "" {
				w.Header().Set("Access-Control-Allow-Origin", origin)
			}
		}

		// Only set credentials if we're using a specific origin (not "*")
		if allowed && (!allowAll || origin != "") {
			w.Header().Set("Access-Control-Allow-Credentials", "true")
		}

		w.Header().Set("Access-Control-Allow-Methods", joinStrings(cfg.CORS.AllowedMethods, ", "))
		w.Header().Set("Access-Control-Allow-Headers", joinStrings(cfg.CORS.AllowedHeaders, ", "))

		// Preflight
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		router.ServeHTTP(w, r)
	})
	
	// Create HTTP server
	addr := cfg.Server.Host + ":" + cfg.Server.Port
	srv := &http.Server{
		Addr:         addr,
		Handler:      corsHandler,
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

