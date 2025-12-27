package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/auth"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
)

// APIKeyHandlers handles API key management endpoints
type APIKeyHandlers struct {
	keyManager *auth.APIKeyManager
	queries    *db.Queries
}

// NewAPIKeyHandlers creates new API key handlers
func NewAPIKeyHandlers(keyManager *auth.APIKeyManager, queries *db.Queries) *APIKeyHandlers {
	return &APIKeyHandlers{
		keyManager: keyManager,
		queries:    queries,
	}
}

// GenerateAPIKey generates a new API key
func (h *APIKeyHandlers) GenerateAPIKey(w http.ResponseWriter, r *http.Request) {
	var req struct {
		UserID    string `json:"user_id,omitempty"`
		RateLimit int    `json:"rate_limit,omitempty"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, http.StatusBadRequest, err, nil)
		return
	}
	
	if req.RateLimit == 0 {
		req.RateLimit = 100
	}
	
	key, apiKey, err := h.keyManager.GenerateAPIKey(r.Context(), req.UserID, req.RateLimit)
	if err != nil {
		WriteError(w, http.StatusInternalServerError, err, nil)
		return
	}
	
	// Return the full key (only shown once)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"id":         apiKey.ID,
		"key":        key, // Full key - only shown once
		"key_prefix": apiKey.KeyPrefix,
		"user_id":    apiKey.UserID,
		"rate_limit": apiKey.RateLimit,
		"created_at": apiKey.CreatedAt,
	})
}

// ListAPIKeys lists all API keys
func (h *APIKeyHandlers) ListAPIKeys(w http.ResponseWriter, r *http.Request) {
	// Query all API keys from database using queries helper
	keys, err := h.queries.GetAllAPIKeys(r.Context())
	if err != nil {
		WriteError(w, http.StatusInternalServerError, err, nil)
		return
	}
	
	// Don't return full keys, only prefixes
	var response []map[string]interface{}
	for _, key := range keys {
		response = append(response, map[string]interface{}{
			"id":         key.ID,
			"key_prefix": key.KeyPrefix,
			"user_id":    key.UserID,
			"rate_limit": key.RateLimit,
			"created_at": key.CreatedAt,
			"last_used_at": key.LastUsedAt,
		})
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// DeleteAPIKey deletes an API key
func (h *APIKeyHandlers) DeleteAPIKey(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	keyID := vars["id"]
	
	if err := h.keyManager.DeleteAPIKey(r.Context(), keyID); err != nil {
		WriteError(w, http.StatusInternalServerError, err, nil)
		return
	}
	
	w.WriteHeader(http.StatusNoContent)
}

