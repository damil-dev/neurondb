package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/auth"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/utils"
	"github.com/neurondb/NeuronDesktop/api/internal/validation"
	"golang.org/x/crypto/bcrypt"
)

// ProfileHandlers handles profile-related endpoints
type ProfileHandlers struct {
	queries *db.Queries
}

func isAdmin(ctx context.Context) bool {
	return auth.GetIsAdminFromContext(ctx)
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

	var (
		profiles []db.Profile
		err      error
	)
	if isAdmin(r.Context()) {
		profiles, err = h.queries.ListAllProfiles(r.Context())
	} else {
		profiles, err = h.queries.ListProfiles(r.Context(), userID)
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// If no profiles exist, create a default profile
	if len(profiles) == 0 && !isAdmin(r.Context()) {
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

		// Initialize database schema for the default profile's database
		if err := utils.InitSchema(r.Context(), neurondbDSN); err != nil {
			// Log error but don't fail - schema might already be initialized
			fmt.Printf("Warning: Failed to initialize schema for default profile: %v\n", err)
		}

		if err := h.queries.CreateProfile(r.Context(), defaultProfile); err != nil {
			// Log error but don't fail the request
			fmt.Printf("Failed to create default profile: %v\n", err)
		} else {
			// Set as default
			if err := h.queries.SetDefaultProfile(r.Context(), defaultProfile.ID); err != nil {
				fmt.Printf("Failed to set default profile: %v\n", err)
			}
			// Automatically create default model configurations
			if err := utils.CreateDefaultModelsForProfile(r.Context(), h.queries, defaultProfile.ID); err != nil {
				// Log error but don't fail - user can add models manually
				fmt.Printf("Warning: Failed to create default models for profile %s: %v\n", defaultProfile.ID, err)
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

	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	// Validate profile ID (UUID)
	if err := validation.ValidateUUIDRequired(profileID, "profile_id"); err != nil {
		http.Error(w, fmt.Sprintf("Invalid profile ID: %v", err), http.StatusBadRequest)
		return
	}

	profile, err := h.queries.GetProfile(r.Context(), profileID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	// Non-admin users can only access their own profiles
	if !isAdmin(r.Context()) && profile.UserID != userID {
		http.Error(w, "Access denied", http.StatusForbidden)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(profile)
}

// CreateProfile creates a new profile
// NOTE: Profiles are now automatically created during user signup.
// This endpoint is disabled to prevent manual profile creation.
func (h *ProfileHandlers) CreateProfile(w http.ResponseWriter, r *http.Request) {
	WriteError(w, r, http.StatusForbidden, fmt.Errorf("profile creation is not allowed. Profiles are automatically created during user signup"), nil)
	return
}

// UpdateProfile updates a profile
func (h *ProfileHandlers) UpdateProfile(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["id"]

	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("unauthorized"), nil)
		return
	}

	// Validate profile ID (UUID)
	if err := validation.ValidateUUIDRequired(profileID, "profile_id"); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid profile ID: %w", err), nil)
		return
	}

	admin := isAdmin(r.Context())

	var req struct {
		db.Profile
		ProfilePassword string `json:"profile_password"` // Plain password from request
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	// Verify profile belongs to user
	existingProfile, err := h.queries.GetProfile(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusNotFound, fmt.Errorf("profile not found"), nil)
		return
	}

	if !admin && existingProfile.UserID != userID {
		WriteError(w, r, http.StatusForbidden, fmt.Errorf("access denied"), nil)
		return
	}

	profile := req.Profile
	profile.ID = profileID
	if admin {
		// Don't allow admin updates to reassign ownership accidentally
		profile.UserID = existingProfile.UserID
	} else {
		profile.UserID = userID
	}

	// Handle profile password update
	if req.ProfilePassword != "" {
		if profile.ProfileUsername == "" {
			WriteError(w, r, http.StatusBadRequest, fmt.Errorf("profile_username is required when profile_password is set"), nil)
			return
		}
		passwordHash, err := bcrypt.GenerateFromPassword([]byte(req.ProfilePassword), bcrypt.DefaultCost)
		if err != nil {
			WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to hash password"), nil)
			return
		}
		profile.ProfilePassword = string(passwordHash)
	} else if profile.ProfileUsername != "" && profile.ProfileUsername != existingProfile.ProfileUsername {
		// If username changed but no password provided, require password
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("profile_password is required when profile_username is set"), nil)
		return
	} else if profile.ProfileUsername == "" {
		// If clearing username, also clear password
		profile.ProfilePassword = ""
	}

	// Validate DSN if provided
	if profile.NeuronDBDSN != "" {
		if err := validation.ValidateDSNRequired(profile.NeuronDBDSN, "neurondb_dsn"); err != nil {
			WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid DSN: %w", err), nil)
			return
		}
	}

	// Validate profile
	if errors := utils.ValidateProfile(profile.Name, profile.NeuronDBDSN, profile.MCPConfig); len(errors) > 0 {
		WriteValidationErrors(w, r, errors)
		return
	}

	if err := h.queries.UpdateProfile(r.Context(), &profile); err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, profile, http.StatusOK)
}

// DeleteProfile deletes a profile
func (h *ProfileHandlers) DeleteProfile(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["id"]

	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	// Validate profile ID (UUID)
	if err := validation.ValidateUUIDRequired(profileID, "profile_id"); err != nil {
		http.Error(w, fmt.Sprintf("Invalid profile ID: %v", err), http.StatusBadRequest)
		return
	}

	profile, err := h.queries.GetProfile(r.Context(), profileID)
	if err != nil {
		http.Error(w, "Profile not found", http.StatusNotFound)
		return
	}

	// Non-admin users can only delete their own profiles
	if !isAdmin(r.Context()) && profile.UserID != userID {
		http.Error(w, "Access denied", http.StatusForbidden)
		return
	}

	if err := h.queries.DeleteProfile(r.Context(), profileID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}
