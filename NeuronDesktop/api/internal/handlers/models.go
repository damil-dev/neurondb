package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
)

// ModelConfigHandlers handles model configuration endpoints
type ModelConfigHandlers struct {
	queries *db.Queries
}

// NewModelConfigHandlers creates new model config handlers
func NewModelConfigHandlers(queries *db.Queries) *ModelConfigHandlers {
	return &ModelConfigHandlers{
		queries: queries,
	}
}

// ListModelConfigs lists all model configurations for a profile
func (h *ModelConfigHandlers) ListModelConfigs(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	includeAPIKey := r.URL.Query().Get("include_api_key") == "true"

	configs, err := h.queries.ListModelConfigs(r.Context(), profileID, includeAPIKey)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, configs, http.StatusOK)
}

// CreateModelConfig creates a new model configuration
func (h *ModelConfigHandlers) CreateModelConfig(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	var config db.ModelConfig
	if err := json.NewDecoder(r.Body).Decode(&config); err != nil {
		WriteError(w, r, http.StatusBadRequest, err, nil)
		return
	}

	config.ProfileID = profileID
	if config.Metadata == nil {
		config.Metadata = make(map[string]interface{})
	}

	// If this is set as default, unset others
	if config.IsDefault {
		// First create the config
		if err := h.queries.CreateModelConfig(r.Context(), &config); err != nil {
			WriteError(w, r, http.StatusInternalServerError, err, nil)
			return
		}
		// Then set it as default (which unsets others)
		if err := h.queries.SetDefaultModelConfig(r.Context(), profileID, config.ID); err != nil {
			WriteError(w, r, http.StatusInternalServerError, err, nil)
			return
		}
	} else {
		if err := h.queries.CreateModelConfig(r.Context(), &config); err != nil {
			WriteError(w, r, http.StatusInternalServerError, err, nil)
			return
		}
	}

	WriteSuccess(w, config, http.StatusCreated)
}

// UpdateModelConfig updates a model configuration
func (h *ModelConfigHandlers) UpdateModelConfig(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	configID := vars["id"]

	var config db.ModelConfig
	if err := json.NewDecoder(r.Body).Decode(&config); err != nil {
		WriteError(w, r, http.StatusBadRequest, err, nil)
		return
	}

	// Get existing config to preserve profile_id
	existing, err := h.queries.GetModelConfig(r.Context(), configID, false)
	if err != nil {
		WriteError(w, r, http.StatusNotFound, err, nil)
		return
	}

	config.ID = configID
	config.ProfileID = existing.ProfileID
	if config.Metadata == nil {
		config.Metadata = existing.Metadata
	}

	// If setting as default, unset others
	if config.IsDefault {
		if err := h.queries.UpdateModelConfig(r.Context(), &config); err != nil {
			WriteError(w, r, http.StatusInternalServerError, err, nil)
			return
		}
		if err := h.queries.SetDefaultModelConfig(r.Context(), config.ProfileID, config.ID); err != nil {
			WriteError(w, r, http.StatusInternalServerError, err, nil)
			return
		}
	} else {
		if err := h.queries.UpdateModelConfig(r.Context(), &config); err != nil {
			WriteError(w, r, http.StatusInternalServerError, err, nil)
			return
		}
	}

	WriteSuccess(w, config, http.StatusOK)
}

// DeleteModelConfig deletes a model configuration
func (h *ModelConfigHandlers) DeleteModelConfig(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	configID := vars["id"]

	if err := h.queries.DeleteModelConfig(r.Context(), configID); err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// GetDefaultModelConfig gets the default model for a profile
func (h *ModelConfigHandlers) GetDefaultModelConfig(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	config, err := h.queries.GetDefaultModelConfig(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusNotFound, err, nil)
		return
	}

	WriteSuccess(w, config, http.StatusOK)
}

// SetDefaultModelConfig sets a model as default
func (h *ModelConfigHandlers) SetDefaultModelConfig(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	configID := vars["id"]

	if err := h.queries.SetDefaultModelConfig(r.Context(), profileID, configID); err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}



