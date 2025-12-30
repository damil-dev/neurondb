package auth

import (
	"context"
	"net/http"
	"strings"
)

// JWTMiddleware provides JWT authentication middleware
func JWTMiddleware() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Skip auth for OPTIONS requests (CORS preflight)
			if r.Method == "OPTIONS" {
				next.ServeHTTP(w, r)
				return
			}

			// Skip auth for public endpoints
			publicPaths := []string{
				"/health",
				"/api/v1/health",
				"/api/v1/auth/register",
				"/api/v1/auth/login",
			}

			for _, path := range publicPaths {
				if r.URL.Path == path {
					next.ServeHTTP(w, r)
					return
				}
			}

			authHeader := r.Header.Get("Authorization")
			// Browser WebSockets can't set custom headers. For websocket endpoints only, allow token via query param (?token=...).
			if authHeader == "" {
				if strings.HasSuffix(r.URL.Path, "/ws") {
					if token := r.URL.Query().Get("token"); token != "" {
						authHeader = "Bearer " + token
					}
				}
			}
			if authHeader == "" {
				http.Error(w, "Missing authorization header", http.StatusUnauthorized)
				return
			}

			tokenString, err := ExtractToken(authHeader)
			if err != nil {
				http.Error(w, err.Error(), http.StatusUnauthorized)
				return
			}

			claims, err := ValidateToken(tokenString)
			if err != nil {
				http.Error(w, "Invalid token", http.StatusUnauthorized)
				return
			}

			// Add user info to context
			ctx := context.WithValue(r.Context(), "user_id", claims.UserID)
			ctx = context.WithValue(ctx, "username", claims.Username)
			ctx = context.WithValue(ctx, "is_admin", claims.IsAdmin)
			ctx = context.WithValue(ctx, "claims", claims)

			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

// GetUserIDFromContext gets the user ID from context (primary method for JWT auth)
func GetUserIDFromContext(ctx context.Context) (string, bool) {
	userID, ok := ctx.Value("user_id").(string)
	return userID, ok
}

// GetUsernameFromContext gets the username from context (for JWT)
func GetUsernameFromContext(ctx context.Context) (string, bool) {
	username, ok := ctx.Value("username").(string)
	return username, ok
}

// GetIsAdminFromContext gets the admin flag from context (for JWT)
func GetIsAdminFromContext(ctx context.Context) bool {
	isAdmin, ok := ctx.Value("is_admin").(bool)
	return ok && isAdmin
}

// GetClaimsFromContext gets the claims from context
func GetClaimsFromContext(ctx context.Context) (*Claims, bool) {
	claims, ok := ctx.Value("claims").(*Claims)
	return claims, ok
}
