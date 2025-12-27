package auth

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"golang.org/x/crypto/bcrypt"
)

// APIKeyManager manages API key authentication
type APIKeyManager struct {
	queries *db.Queries
}

// NewAPIKeyManager creates a new API key manager
func NewAPIKeyManager(queries *db.Queries) *APIKeyManager {
	return &APIKeyManager{queries: queries}
}

// GenerateAPIKey generates a new API key
func (m *APIKeyManager) GenerateAPIKey(ctx context.Context, userID string, rateLimit int) (string, *APIKey, error) {
	// Generate random key (32 bytes = 44 base64 chars)
	keyBytes := make([]byte, 32)
	if _, err := rand.Read(keyBytes); err != nil {
		return "", nil, fmt.Errorf("failed to generate key: %w", err)
	}

	key := base64.URLEncoding.EncodeToString(keyBytes)
	keyPrefix := GetKeyPrefix(key)
	keyHash, err := HashAPIKey(key)
	if err != nil {
		return "", nil, fmt.Errorf("failed to hash key: %w", err)
	}

	apiKey := &db.APIKey{
		ID:        uuid.New().String(),
		KeyHash:   keyHash,
		KeyPrefix: keyPrefix,
		UserID:    userID,
		RateLimit: rateLimit,
	}

	if err := m.queries.CreateAPIKey(ctx, apiKey); err != nil {
		return "", nil, fmt.Errorf("failed to create API key: %w", err)
	}

	return key, apiKey, nil
}

// ValidateAPIKey validates an API key and returns the key record
func (m *APIKeyManager) ValidateAPIKey(ctx context.Context, key string) (*db.APIKey, error) {
	prefix := GetKeyPrefix(key)

	// Find key by prefix
	apiKey, err := m.queries.GetAPIKeyByPrefix(ctx, prefix)
	if err != nil {
		return nil, fmt.Errorf("API key not found")
	}

	// Verify key
	if !VerifyAPIKey(key, apiKey.KeyHash) {
		return nil, fmt.Errorf("invalid API key")
	}

	// Update last used
	_ = m.queries.UpdateAPIKeyLastUsed(ctx, apiKey.ID)

	return apiKey, nil
}

// DeleteAPIKey deletes an API key
func (m *APIKeyManager) DeleteAPIKey(ctx context.Context, id string) error {
	return m.queries.DeleteAPIKey(ctx, id)
}

// APIKey is an alias for db.APIKey for convenience
type APIKey = db.APIKey

// GetKeyPrefix extracts the prefix from an API key
func GetKeyPrefix(key string) string {
	if len(key) < 8 {
		return key
	}
	return key[:8]
}

// HashAPIKey hashes an API key using bcrypt
func HashAPIKey(key string) (string, error) {
	hash, err := bcrypt.GenerateFromPassword([]byte(key), bcrypt.DefaultCost)
	if err != nil {
		return "", fmt.Errorf("failed to hash key: %w", err)
	}
	return string(hash), nil
}

// VerifyAPIKey verifies an API key against its hash
func VerifyAPIKey(key, hash string) bool {
	err := bcrypt.CompareHashAndPassword([]byte(hash), []byte(key))
	return err == nil
}

// ExtractAPIKey extracts the API key from an Authorization header
func ExtractAPIKey(authHeader string) (string, error) {
	if authHeader == "" {
		return "", fmt.Errorf("missing authorization header")
	}

	parts := strings.Split(authHeader, " ")
	if len(parts) != 2 || strings.ToLower(parts[0]) != "bearer" {
		return "", fmt.Errorf("invalid authorization header format")
	}

	return parts[1], nil
}

