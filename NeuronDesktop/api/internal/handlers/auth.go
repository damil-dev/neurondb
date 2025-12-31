package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"

	"github.com/neurondb/NeuronDesktop/api/internal/auth"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/utils"
	"golang.org/x/crypto/bcrypt"
)

// AuthHandlers handles authentication requests
type AuthHandlers struct {
	queries *db.Queries
}

// NewAuthHandlers creates a new auth handlers instance
func NewAuthHandlers(queries *db.Queries) *AuthHandlers {
	return &AuthHandlers{queries: queries}
}

// RegisterRequest is the request to register a new user
type RegisterRequest struct {
	Username    string `json:"username"`
	Password    string `json:"password"`
	NeuronDBDSN string `json:"neurondb_dsn,omitempty"` // Optional: if not provided, uses default from env
}

// LoginRequest is the request to login
type LoginRequest struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

// AuthResponse is the response for auth operations
type AuthResponse struct {
	Token     string  `json:"token"`
	UserID    string  `json:"user_id"`
	Username  string  `json:"username"`
	IsAdmin   bool    `json:"is_admin"`
	ProfileID *string `json:"profile_id,omitempty"` // Profile ID if login matched a profile
}

// Register registers a new user
func (h *AuthHandlers) Register(w http.ResponseWriter, r *http.Request) {
	var req RegisterRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	// Validate input
	if req.Username == "" || req.Password == "" {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("username and password are required"), nil)
		return
	}

	if len(req.Password) < 6 {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("password must be at least 6 characters"), nil)
		return
	}

	// Check if user already exists
	_, err := h.queries.GetUserByUsername(r.Context(), req.Username)
	if err == nil {
		WriteError(w, r, http.StatusConflict, fmt.Errorf("username already exists"), nil)
		return
	}

	// Hash password
	passwordHash, err := bcrypt.GenerateFromPassword([]byte(req.Password), bcrypt.DefaultCost)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to hash password"), nil)
		return
	}

	// Create user
	user := &db.User{
		Username:     req.Username,
		PasswordHash: string(passwordHash),
	}

	if err := h.queries.CreateUser(r.Context(), user); err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("Failed to create user"), nil)
		return
	}

	// Automatically create a profile for the new user
	// Use the same password for profile credentials
	profilePasswordHash, err := bcrypt.GenerateFromPassword([]byte(req.Password), bcrypt.DefaultCost)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("Failed to hash profile password"), nil)
		return
	}

	// Get database configuration - use provided DSN or default
	neurondbDSN := req.NeuronDBDSN
	if neurondbDSN == "" {
		neurondbDSN = utils.GetDefaultNeuronDBDSN()
	}
	mcpConfig := utils.GetDefaultMCPConfig()

	// Create user profile with same username/password
	userProfile := &db.Profile{
		UserID:          user.ID,
		Name:            req.Username, // Use username as profile name
		ProfileUsername: req.Username, // Profile login uses same username
		ProfilePassword: string(profilePasswordHash),
		NeuronDBDSN:     neurondbDSN,
		MCPConfig:       mcpConfig,
		IsDefault:       true,
	}

	if err := h.queries.CreateProfile(r.Context(), userProfile); err != nil {
		// Log error but don't fail registration - user can still log in
		fmt.Fprintf(os.Stderr, "Warning: Failed to create default profile for user '%s' during registration (user can still log in): %v\n", user.Username, err)
	} else {
		// Create default model configurations for this profile
		if err := utils.CreateDefaultModelsForProfile(r.Context(), h.queries, userProfile.ID); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: Failed to create default model configurations for profile %s (models can be added manually): %v\n", userProfile.ID, err)
		}
	}

	// Generate token
	token, err := auth.GenerateToken(user.ID, user.Username, user.IsAdmin)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("Failed to generate token"), nil)
		return
	}

	// Return response with profile ID if created
	response := AuthResponse{
		Token:    token,
		UserID:   user.ID,
		Username: user.Username,
		IsAdmin:  user.IsAdmin,
	}
	if userProfile.ID != "" {
		response.ProfileID = &userProfile.ID
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// Login logs in a user
func (h *AuthHandlers) Login(w http.ResponseWriter, r *http.Request) {
	var req LoginRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("Invalid request body"), nil)
		return
	}

	// Validate input
	if req.Username == "" || req.Password == "" {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("Username and password are required"), nil)
		return
	}

	// First, try to find a profile with matching credentials
	profile, profileErr := h.queries.GetProfileByUsernameAndPassword(r.Context(), req.Username, req.Password)
	if profileErr == nil && profile != nil {
		// Profile credentials matched - use the profile's user_id
		user, err := h.queries.GetUserByID(r.Context(), profile.UserID)
		if err != nil {
			// If profile user doesn't exist, create a token for the profile's user_id
			// This allows profile-based login even if user account doesn't exist
			token, err := auth.GenerateToken(profile.UserID, req.Username, false)
			if err != nil {
				WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("Failed to generate token"), nil)
				return
			}
			response := AuthResponse{
				Token:     token,
				UserID:    profile.UserID,
				Username:  req.Username,
				IsAdmin:   false,
				ProfileID: &profile.ID,
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(response)
			return
		}

		// Generate token for the user
		token, err := auth.GenerateToken(user.ID, user.Username, user.IsAdmin)
		if err != nil {
			WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("Failed to generate token"), nil)
			return
		}

		response := AuthResponse{
			Token:     token,
			UserID:    user.ID,
			Username:  user.Username,
			IsAdmin:   user.IsAdmin,
			ProfileID: &profile.ID,
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
		return
	}

	// If no profile match, try regular user login
	user, err := h.queries.GetUserByUsername(r.Context(), req.Username)
	if err != nil {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("Invalid username or password"), nil)
		return
	}

	// Verify password
	err = bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(req.Password))
	if err != nil {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("Invalid username or password"), nil)
		return
	}

	// Generate token
	token, err := auth.GenerateToken(user.ID, user.Username, user.IsAdmin)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("Failed to generate token"), nil)
		return
	}

	// Return response
	response := AuthResponse{
		Token:    token,
		UserID:   user.ID,
		Username: user.Username,
		IsAdmin:  user.IsAdmin,
	}

	// Always include profile_id so the UI can land on the correct profile without selection
	if p, err := h.queries.GetDefaultProfileForUser(r.Context(), user.ID); err == nil && p != nil {
		response.ProfileID = &p.ID
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// GetCurrentUser returns the current authenticated user
func (h *AuthHandlers) GetCurrentUser(w http.ResponseWriter, r *http.Request) {
	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("Unauthorized"), nil)
		return
	}

	user, err := h.queries.GetUserByID(r.Context(), userID)
	if err != nil {
		WriteError(w, r, http.StatusNotFound, fmt.Errorf("User not found"), nil)
		return
	}

	response := map[string]interface{}{
		"user_id":  user.ID,
		"username": user.Username,
		"is_admin": user.IsAdmin,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}
