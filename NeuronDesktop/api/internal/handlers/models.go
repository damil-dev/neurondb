package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
)

/* ModelHandlers handles model and API key management */
type ModelHandlers struct {
	queries *db.Queries
}

/* NewModelHandlers creates new model handlers */
func NewModelHandlers(queries *db.Queries) *ModelHandlers {
	return &ModelHandlers{queries: queries}
}

/* Model represents an LLM model configuration */
type Model struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Provider    string                 `json:"provider"`
	ModelType   string                 `json:"model_type"`
	APIKeySet   bool                   `json:"api_key_set"`
	Config      map[string]interface{} `json:"config,omitempty"`
	CreatedAt   string                 `json:"created_at,omitempty"`
	UpdatedAt   string                 `json:"updated_at,omitempty"`
}

/* ListModels lists all models for a profile */
func (h *ModelHandlers) ListModels(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	// Get profile to access NeuronDB connection
	profile, err := h.queries.GetProfile(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusNotFound, fmt.Errorf("profile not found"), nil)
		return
	}

	// Query models from NeuronDB (using MCP schema)
	query := `
		SELECT 
			id,
			name,
			provider,
			model_type,
			CASE WHEN api_key_encrypted IS NOT NULL THEN true ELSE false END as api_key_set,
			config,
			created_at,
			updated_at
		FROM neurondb.llm_models
		WHERE enabled = true
		ORDER BY name
	`

	rows, err := h.queries.GetDB().QueryContext(r.Context(), query)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to query models: %w", err), nil)
		return
	}
	defer rows.Close()

	var models []Model
	for rows.Next() {
		var m Model
		var configJSON []byte
		err := rows.Scan(
			&m.ID, &m.Name, &m.Provider, &m.ModelType,
			&m.APIKeySet, &configJSON, &m.CreatedAt, &m.UpdatedAt,
		)
		if err != nil {
			continue
		}

		if len(configJSON) > 0 {
			json.Unmarshal(configJSON, &m.Config)
		}

		models = append(models, m)
	}

	WriteSuccess(w, models, http.StatusOK)
}

/* AddModel adds a new model */
func (h *ModelHandlers) AddModel(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	var req struct {
		Name      string                 `json:"name"`
		Provider  string                 `json:"provider"`
		ModelType string                 `json:"model_type"`
		Config    map[string]interface{} `json:"config,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	// Validate required fields
	if req.Name == "" || req.Provider == "" {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("name and provider are required"), nil)
		return
	}

	// Insert model into NeuronDB
	configJSON, _ := json.Marshal(req.Config)
	query := `
		INSERT INTO neurondb.llm_models (name, provider, model_type, config, enabled)
		VALUES ($1, $2, $3, $4::jsonb, true)
		ON CONFLICT (name) DO UPDATE
		SET provider = EXCLUDED.provider,
		    model_type = EXCLUDED.model_type,
		    config = EXCLUDED.config,
		    updated_at = NOW()
		RETURNING id, name, provider, model_type, 
		          CASE WHEN api_key_encrypted IS NOT NULL THEN true ELSE false END as api_key_set,
		          config, created_at, updated_at
	`

	var m Model
	var configJSONOut []byte
	err := h.queries.GetDB().QueryRowContext(r.Context(), query,
		req.Name, req.Provider, req.ModelType, configJSON,
	).Scan(
		&m.ID, &m.Name, &m.Provider, &m.ModelType,
		&m.APIKeySet, &configJSONOut, &m.CreatedAt, &m.UpdatedAt,
	)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to add model: %w", err), nil)
		return
	}

	if len(configJSONOut) > 0 {
		json.Unmarshal(configJSONOut, &m.Config)
	}

	WriteSuccess(w, m, http.StatusCreated)
}

/* SetModelKey sets API key for a model */
func (h *ModelHandlers) SetModelKey(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	modelName := vars["model_name"]

	var req struct {
		APIKey string `json:"api_key"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	if req.APIKey == "" {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("api_key is required"), nil)
		return
	}

	// Use NeuronDB function to set encrypted API key
	query := `SELECT neurondb_set_model_key($1, $2)`
	_, err := h.queries.GetDB().ExecContext(r.Context(), query, modelName, req.APIKey)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to set API key: %w", err), nil)
		return
	}

	WriteSuccess(w, map[string]interface{}{"message": "API key set successfully"}, http.StatusOK)
}

/* DeleteModel deletes a model */
func (h *ModelHandlers) DeleteModel(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	modelID := vars["model_id"]

	// Soft delete by setting enabled = false
	query := `UPDATE neurondb.llm_models SET enabled = false WHERE id = $1`
	result, err := h.queries.GetDB().ExecContext(r.Context(), query, modelID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to delete model: %w", err), nil)
		return
	}

	rowsAffected, _ := result.RowsAffected()
	if rowsAffected == 0 {
		WriteError(w, r, http.StatusNotFound, fmt.Errorf("model not found"), nil)
		return
	}

	WriteSuccess(w, map[string]interface{}{"message": "Model deleted successfully"}, http.StatusOK)
}

/* GetModelInfo gets detailed information about a model */
func (h *ModelHandlers) GetModelInfo(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	modelID := vars["model_id"]

	query := `
		SELECT 
			id, name, provider, model_type,
			CASE WHEN api_key_encrypted IS NOT NULL THEN true ELSE false END as api_key_set,
			config, created_at, updated_at
		FROM neurondb.llm_models
		WHERE id = $1 AND enabled = true
	`

	var m Model
	var configJSON []byte
	err := h.queries.GetDB().QueryRowContext(r.Context(), query, modelID).Scan(
		&m.ID, &m.Name, &m.Provider, &m.ModelType,
		&m.APIKeySet, &configJSON, &m.CreatedAt, &m.UpdatedAt,
	)
	if err != nil {
		WriteError(w, r, http.StatusNotFound, fmt.Errorf("model not found"), nil)
		return
	}

	if len(configJSON) > 0 {
		json.Unmarshal(configJSON, &m.Config)
	}

	WriteSuccess(w, m, http.StatusOK)
}
