package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/auth"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/middleware"
)

/* SetupTestServer creates a test HTTP server with all routes configured */
/* This is in the handlers package to avoid import cycles */
func SetupTestServer(queries *db.Queries) *httptest.Server {
	/* Set JWT secret for testing */
	os.Setenv("JWT_SECRET", "test-secret-key-for-testing-only")

	router := mux.NewRouter()

	/* Apply middleware */
	router.Use(middleware.RequestIDMiddleware())
	router.Use(middleware.RecoveryMiddleware(nil))
	// LoggingMiddleware requires logger and queries - skip for test server
	// router.Use(middleware.LoggingMiddleware(nil, nil))

	/* Health check (no auth) */
	router.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "ok",
		})
	}).Methods("GET")

	/* Auth routes */
	authHandlers := NewAuthHandlers(queries)
	authRouter := router.PathPrefix("/api/v1/auth").Subrouter()
	authRouter.HandleFunc("/register", authHandlers.Register).Methods("POST")
	authRouter.HandleFunc("/login", authHandlers.Login).Methods("POST")
	authRouter.HandleFunc("/logout", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]interface{}{"message": "logged out"})
	}).Methods("POST")

	/* API routes (with auth) */
	apiRouter := router.PathPrefix("/api/v1").Subrouter()
	apiRouter.Use(auth.JWTMiddleware())
	apiRouter.Use(middleware.RateLimitMiddleware(middleware.NewRateLimiter(10000, 1)))

	/* Current user endpoint */
	apiRouter.HandleFunc("/auth/me", authHandlers.GetCurrentUser).Methods("GET")

	/* Profile routes */
	profileHandlers := NewProfileHandlers(queries)
	apiRouter.HandleFunc("/profiles", profileHandlers.ListProfiles).Methods("GET")
	apiRouter.HandleFunc("/profiles", profileHandlers.CreateProfile).Methods("POST")
	apiRouter.HandleFunc("/profiles/{id}", profileHandlers.GetProfile).Methods("GET")
	apiRouter.HandleFunc("/profiles/{id}", profileHandlers.UpdateProfile).Methods("PUT")
	apiRouter.HandleFunc("/profiles/{id}", profileHandlers.DeleteProfile).Methods("DELETE")

	/* MCP routes */
	mcpManager := NewMCPManager(queries)
	mcpHandlers := NewMCPHandlers(mcpManager)
	apiRouter.HandleFunc("/mcp/connections", mcpHandlers.ListConnections).Methods("GET")
	apiRouter.HandleFunc("/mcp/test", mcpHandlers.TestMCPConfig).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/tools", mcpHandlers.ListTools).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/mcp/tools/call", mcpHandlers.CallTool).Methods("POST")

	/* NeuronDB routes */
	neurondbHandlers := NewNeuronDBHandlers(queries, false)
	apiRouter.HandleFunc("/profiles/{profile_id}/neurondb/collections", neurondbHandlers.ListCollections).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/neurondb/search", neurondbHandlers.Search).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/neurondb/sql", neurondbHandlers.ExecuteSQL).Methods("POST")

	/* Agent routes */
	agentHandlers := NewAgentHandlers(queries)
	apiRouter.HandleFunc("/agent/test", agentHandlers.TestAgentConfig).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents", agentHandlers.ListAgents).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/agents", agentHandlers.CreateAgent).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/models", agentHandlers.ListModels).Methods("GET")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/sessions", agentHandlers.CreateSession).Methods("POST")
	apiRouter.HandleFunc("/profiles/{profile_id}/agent/sessions/{session_id}/messages", agentHandlers.SendMessage).Methods("POST")

	/* Metrics routes */
	metricsHandlers := NewMetricsHandlers()
	apiRouter.HandleFunc("/metrics", metricsHandlers.GetMetrics).Methods("GET")
	apiRouter.HandleFunc("/metrics/reset", metricsHandlers.ResetMetrics).Methods("POST")

	/* Factory routes */
	factoryHandlers := NewFactoryHandlers(queries)
	apiRouter.HandleFunc("/factory/status", factoryHandlers.GetFactoryStatus).Methods("GET")
	apiRouter.HandleFunc("/factory/setup-state", factoryHandlers.GetSetupState).Methods("GET")
	apiRouter.HandleFunc("/factory/setup-state", factoryHandlers.SetSetupState).Methods("POST")

	return httptest.NewServer(router)
}
