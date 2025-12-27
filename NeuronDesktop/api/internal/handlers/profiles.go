package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/auth"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/utils"
)

// ProfileHandlers handles profile-related endpoints
type ProfileHandlers struct {
	queries *db.Queries
}

// NewProfileHandlers creates new profile handlers
func NewProfileHandlers(queries *db.Queries) *ProfileHandlers {
	return &ProfileHandlers{queries: queries}
}

// ListProfiles lists profiles for the current user
func (h *ProfileHandlers) ListProfiles(w http.ResponseWriter, r *http.Request) {
	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}
	
	profiles, err := h.queries.ListProfiles(r.Context(), userID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// If no profiles exist, create a default profile
	if len(profiles) == 0 {
		// Auto-detect NeuronMCP binary and create default config
		mcpConfig := utils.GetDefaultMCPConfig()
		
		// Get default NeuronDB DSN from environment
		neurondbDSN := utils.GetDefaultNeuronDBDSN()
		
		// Get agent endpoint from environment if available
		agentEndpoint := os.Getenv("NEURONAGENT_ENDPOINT")
		agentAPIKey := os.Getenv("NEURONAGENT_API_KEY")
		
		defaultProfile := &db.Profile{
			UserID:        userID,
			Name:          "Default",
			NeuronDBDSN:   neurondbDSN,
			MCPConfig:     mcpConfig,
			AgentEndpoint: agentEndpoint,
			AgentAPIKey:   agentAPIKey,
			IsDefault:     true,
		}
		
		if err := h.queries.CreateProfile(r.Context(), defaultProfile); err != nil {
			// Log error but don't fail the request
			fmt.Printf("Failed to create default profile: %v\n", err)
		} else {
			// Set as default
			if err := h.queries.SetDefaultProfile(r.Context(), defaultProfile.ID); err != nil {
				fmt.Printf("Failed to set default profile: %v\n", err)
			}
			// Reload profiles
			profiles, _ = h.queries.ListProfiles(r.Context(), userID)
		}
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(profiles)
}

// GetProfile gets a single profile
func (h *ProfileHandlers) GetProfile(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["id"]
	
	profile, err := h.queries.GetProfile(r.Context(), profileID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(profile)
}

// CreateProfile creates a new profile
func (h *ProfileHandlers) CreateProfile(w http.ResponseWriter, r *http.Request) {
	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok {
		WriteError(w, http.StatusUnauthorized, fmt.Errorf("unauthorized"), nil)
		return
	}
	
	var profile db.Profile
	if err := json.NewDecoder(r.Body).Decode(&profile); err != nil {
		WriteError(w, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}
	
	// Validate profile
	if errors := utils.ValidateProfile(profile.Name, profile.NeuronDBDSN, profile.MCPConfig); len(errors) > 0 {
		WriteValidationErrors(w, errors)
		return
	}
	
	profile.UserID = userID
	
	if err := h.queries.CreateProfile(r.Context(), &profile); err != nil {
		WriteError(w, http.StatusInternalServerError, err, nil)
		return
	}
	
	WriteSuccess(w, profile, http.StatusCreated)
}

// UpdateProfile updates a profile
func (h *ProfileHandlers) UpdateProfile(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["id"]
	
	var profile db.Profile
	if err := json.NewDecoder(r.Body).Decode(&profile); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}
	
	profile.ID = profileID
	
	if err := h.queries.UpdateProfile(r.Context(), &profile); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(profile)
}

// DeleteProfile deletes a profile
func (h *ProfileHandlers) DeleteProfile(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["id"]
	
	if err := h.queries.DeleteProfile(r.Context(), profileID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	w.WriteHeader(http.StatusNoContent)
}

