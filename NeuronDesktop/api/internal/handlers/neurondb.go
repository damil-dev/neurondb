package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/neurondb"
	"github.com/neurondb/NeuronDesktop/api/internal/utils"
)

// NeuronDBHandlers handles NeuronDB-related endpoints
type NeuronDBHandlers struct {
	queries          *db.Queries
	clients          map[string]*neurondb.Client
	mu               sync.RWMutex
	enableSQLConsole bool
}

// NewNeuronDBHandlers creates new NeuronDB handlers
func NewNeuronDBHandlers(queries *db.Queries, enableSQLConsole bool) *NeuronDBHandlers {
	return &NeuronDBHandlers{
		queries:          queries,
		clients:          make(map[string]*neurondb.Client),
		enableSQLConsole: enableSQLConsole,
	}
}

// getClient gets or creates a NeuronDB client for a profile
func (h *NeuronDBHandlers) getClient(ctx context.Context, profileID string) (*neurondb.Client, error) {
	h.mu.RLock()
	client, ok := h.clients[profileID]
	h.mu.RUnlock()

	if ok {
		return client, nil
	}

	// Get profile
	profile, err := h.queries.GetProfile(ctx, profileID)
	if err != nil {
		return nil, fmt.Errorf("failed to get profile: %w", err)
	}

	// Create client
	client, err = neurondb.NewClient(profile.NeuronDBDSN)
	if err != nil {
		return nil, fmt.Errorf("failed to create NeuronDB client: %w", err)
	}

	h.mu.Lock()
	h.clients[profileID] = client
	h.mu.Unlock()

	return client, nil
}

// ListCollections lists collections
func (h *NeuronDBHandlers) ListCollections(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	client, err := h.getClient(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	collections, err := client.ListCollections(r.Context())
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, collections, http.StatusOK)
}

// Search performs a vector search
func (h *NeuronDBHandlers) Search(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	var req neurondb.SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	// Validate search request
	if req.Limit == 0 {
		req.Limit = 10
	}
	if errors := utils.ValidateSearchRequest(req.Collection, req.Limit, req.DistanceType); len(errors) > 0 {
		WriteValidationErrors(w, r, errors)
		return
	}

	client, err := h.getClient(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	results, err := client.Search(r.Context(), req)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, results, http.StatusOK)
}

// ExecuteSQL executes a SQL query (SELECT only for safety)
func (h *NeuronDBHandlers) ExecuteSQL(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	var req struct {
		Query string `json:"query"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	// Validate SQL (only SELECT allowed)
	if err := utils.ValidateSQL(req.Query); err != nil {
		WriteError(w, r, http.StatusBadRequest, err, nil)
		return
	}

	client, err := h.getClient(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	results, err := client.ExecuteSQL(r.Context(), req.Query)
	if err != nil {
		WriteError(w, r, http.StatusBadRequest, err, nil)
		return
	}

	WriteSuccess(w, results, http.StatusOK)
}

// ExecuteSQLFull executes any SQL query (CREATE, INSERT, UPDATE, DELETE, etc.)
// Use with caution - this allows full database manipulation
// Requires: 1) SQL console enabled via config, 2) Admin user
func (h *NeuronDBHandlers) ExecuteSQLFull(w http.ResponseWriter, r *http.Request) {
	// Check if SQL console is enabled
	if !h.enableSQLConsole {
		WriteError(w, r, http.StatusForbidden, fmt.Errorf("SQL console is disabled"), nil)
		return
	}

	// Check if user is admin (from context set by auth middleware)
	userID, ok := r.Context().Value("user_id").(string)
	if !ok {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("unauthorized"), nil)
		return
	}

	// Get user to check admin status
	user, err := h.queries.GetUserByID(r.Context(), userID)
	if err != nil || !user.IsAdmin {
		WriteError(w, r, http.StatusForbidden, fmt.Errorf("admin access required"), nil)
		return
	}

	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	var req struct {
		Query string `json:"query"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	if strings.TrimSpace(req.Query) == "" {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("query cannot be empty"), nil)
		return
	}

	client, err := h.getClient(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	results, err := client.ExecuteSQLFull(r.Context(), req.Query)
	if err != nil {
		WriteError(w, r, http.StatusBadRequest, err, nil)
		return
	}

	WriteSuccess(w, results, http.StatusOK)
}
