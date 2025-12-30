package auth

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

var jwtSecret []byte

func init() {
	secret := os.Getenv("JWT_SECRET")
	if secret == "" {
		// Fail fast if JWT_SECRET is not set when JWT mode is used
		// This will be checked in main.go based on AUTH_MODE
		jwtSecret = nil
	} else {
		jwtSecret = []byte(secret)
	}
}

// GetJWTSecret returns the JWT secret, or error if not configured
func GetJWTSecret() ([]byte, error) {
	if jwtSecret == nil {
		return nil, fmt.Errorf("JWT_SECRET environment variable is required when using JWT authentication")
	}
	return jwtSecret, nil
}

// Claims represents JWT claims
type Claims struct {
	UserID   string `json:"user_id"`
	Username string `json:"username"`
	IsAdmin  bool   `json:"is_admin"`
	jwt.RegisteredClaims
}

// GenerateToken generates a JWT token for a user
func GenerateToken(userID, username string, isAdmin bool) (string, error) {
	secret, err := GetJWTSecret()
	if err != nil {
		return "", err
	}

	expirationTime := time.Now().Add(24 * time.Hour) // Token expires in 24 hours

	claims := &Claims{
		UserID:   userID,
		Username: username,
		IsAdmin:  isAdmin,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(expirationTime),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	tokenString, err := token.SignedString(secret)
	if err != nil {
		return "", err
	}

	return tokenString, nil
}

// ValidateToken validates a JWT token and returns the claims
func ValidateToken(tokenString string) (*Claims, error) {
	secret, err := GetJWTSecret()
	if err != nil {
		return nil, err
	}

	claims := &Claims{}

	token, err := jwt.ParseWithClaims(tokenString, claims, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, errors.New("unexpected signing method")
		}
		return secret, nil
	})

	if err != nil {
		return nil, err
	}

	if !token.Valid {
		return nil, errors.New("invalid token")
	}

	return claims, nil
}

// ExtractToken extracts the JWT token from an Authorization header
func ExtractToken(authHeader string) (string, error) {
	if authHeader == "" {
		return "", errors.New("missing authorization header")
	}

	// Support both "Bearer <token>" and just "<token>"
	parts := strings.Split(authHeader, " ")
	if len(parts) == 2 && strings.ToLower(parts[0]) == "bearer" {
		return parts[1], nil
	}
	if len(parts) == 1 {
		return parts[0], nil
	}

	return "", errors.New("invalid authorization header format")
}
