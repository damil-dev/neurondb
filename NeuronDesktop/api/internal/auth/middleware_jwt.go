package auth

import (
	"net/http"
	"strings"
)

/* JWTMiddleware provides JWT authentication middleware */
func JWTMiddleware() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "OPTIONS" {
				next.ServeHTTP(w, r)
				return
			}

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

			ctx := SetUserID(r.Context(), claims.UserID)
			ctx = SetUsername(ctx, claims.Username)
			ctx = SetIsAdmin(ctx, claims.IsAdmin)
			ctx = SetClaims(ctx, claims)

			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

/* NOTE: GetUserIDFromContext, GetUsernameFromContext, GetIsAdminFromContext, and GetClaimsFromContext
 * are now defined in context_keys.go for type-safe context access */
