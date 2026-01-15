package auth

import (
	"context"

	"github.com/neurondb/NeuronDesktop/api/internal/db"
)

/* Context key types for type-safe context values */
type contextKey string

const (
	userIDKey   contextKey = "user_id"
	usernameKey contextKey = "username"
	isAdminKey  contextKey = "is_admin"
	claimsKey   contextKey = "claims"
	apiKeyKey   contextKey = "api_key"
)

/* SetUserID sets user ID in context */
func SetUserID(ctx context.Context, userID string) context.Context {
	return context.WithValue(ctx, userIDKey, userID)
}

/* GetUserIDFromContext gets the user ID from context */
func GetUserIDFromContext(ctx context.Context) (string, bool) {
	userID, ok := ctx.Value(userIDKey).(string)
	return userID, ok
}

/* SetUsername sets username in context */
func SetUsername(ctx context.Context, username string) context.Context {
	return context.WithValue(ctx, usernameKey, username)
}

/* GetUsernameFromContext gets the username from context */
func GetUsernameFromContext(ctx context.Context) (string, bool) {
	username, ok := ctx.Value(usernameKey).(string)
	return username, ok
}

/* SetIsAdmin sets admin flag in context */
func SetIsAdmin(ctx context.Context, isAdmin bool) context.Context {
	return context.WithValue(ctx, isAdminKey, isAdmin)
}

/* GetIsAdminFromContext gets the admin flag from context */
func GetIsAdminFromContext(ctx context.Context) bool {
	isAdmin, ok := ctx.Value(isAdminKey).(bool)
	return ok && isAdmin
}

/* SetClaims sets claims in context */
func SetClaims(ctx context.Context, claims *Claims) context.Context {
	return context.WithValue(ctx, claimsKey, claims)
}

/* GetClaimsFromContext gets the claims from context */
func GetClaimsFromContext(ctx context.Context) (*Claims, bool) {
	claims, ok := ctx.Value(claimsKey).(*Claims)
	return claims, ok
}

/* SetAPIKey sets API key in context */
func SetAPIKey(ctx context.Context, apiKey *db.APIKey) context.Context {
	return context.WithValue(ctx, apiKeyKey, apiKey)
}

/* GetAPIKeyValueFromContext gets the API key value from context (internal) */
func getAPIKeyValueFromContext(ctx context.Context) (*db.APIKey, bool) {
	key, ok := ctx.Value(apiKeyKey).(*db.APIKey)
	return key, ok
}

