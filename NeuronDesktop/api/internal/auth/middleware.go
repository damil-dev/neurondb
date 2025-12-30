package auth

import (
	"context"
	"net/http"

	"github.com/neurondb/NeuronDesktop/api/internal/db"
)

// Middleware provides API key authentication middleware
func Middleware(keyManager *APIKeyManager) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Skip auth for health check
			if r.URL.Path == "/health" || r.URL.Path == "/api/v1/health" {
				next.ServeHTTP(w, r)
				return
			}

			authHeader := r.Header.Get("Authorization")
			if authHeader == "" {
				http.Error(w, "Missing authorization header", http.StatusUnauthorized)
				return
			}

			key, err := ExtractAPIKey(authHeader)
			if err != nil {
				http.Error(w, err.Error(), http.StatusUnauthorized)
				return
			}

			apiKey, err := keyManager.ValidateAPIKey(r.Context(), key)
			if err != nil {
				http.Error(w, "Invalid API key", http.StatusUnauthorized)
				return
			}

			// Add API key info to context
			ctx := context.WithValue(r.Context(), "api_key", apiKey)
			ctx = context.WithValue(ctx, "user_id", apiKey.UserID)

			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

// GetAPIKeyFromContext gets the API key from context
func GetAPIKeyFromContext(ctx context.Context) (*db.APIKey, bool) {
	key, ok := ctx.Value("api_key").(*db.APIKey)
	return key, ok
}

// NOTE: GetUserIDFromContext is now in middleware_jwt.go for JWT authentication
// This file is kept for API key functionality if needed for backwards compatibility
