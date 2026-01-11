#!/bin/bash
# Generate and display API key for NeuronDesktop

set -e

DB_DSN="${DB_DSN:-postgresql://neurondesk:neurondesk@localhost:5432/neurondesk}"
USER_ID="${USER_ID:-nbduser}"
RATE_LIMIT="${RATE_LIMIT:-1000}"

echo "Generating API key for NeuronDesktop..."
echo "Database: $DB_DSN"
echo "User: $USER_ID"
echo ""

cd "$(dirname "$0")/../api"

# Generate API key using Go
go run -exec "cd $(pwd)" << 'EOF'
package main

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/base64"
	"fmt"
	"os"
	"strings"
	"time"

	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/google/uuid"
	"golang.org/x/crypto/bcrypt"
)

func main() {
	dsn := os.Getenv("DB_DSN")
	if dsn == "" {
		dsn = "postgresql://neurondesk:neurondesk@localhost:5432/neurondesk"
	}
	userID := os.Getenv("USER_ID")
	if userID == "" {
		userID = "nbduser"
	}

	db, err := sql.Open("pgx", dsn)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to connect: %v\n", err)
		os.Exit(1)
	}
	defer db.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := db.PingContext(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to ping: %v\n", err)
		os.Exit(1)
	}

	keyBytes := make([]byte, 32)
	rand.Read(keyBytes)
	key := base64.URLEncoding.EncodeToString(keyBytes)
	keyPrefix := key[:8]
	
	keyHash, _ := bcrypt.GenerateFromPassword([]byte(key), bcrypt.DefaultCost)
	keyID := uuid.New().String()

	query := `INSERT INTO api_keys (id, key_hash, key_prefix, user_id, rate_limit, created_at)
		VALUES ($1, $2, $3, $4, $5, NOW()) RETURNING id`
	
	var createdID string
	db.QueryRowContext(ctx, query, keyID, string(keyHash), keyPrefix, userID, 1000).Scan(&createdID)

	fmt.Println("=" + strings.Repeat("=", 70))
	fmt.Println("API Key Generated!")
	fmt.Println("=" + strings.Repeat("=", 70))
	fmt.Printf("Key: %s\n", key)
	fmt.Println("")
	fmt.Println("Set it in browser console:")
	fmt.Printf("localStorage.setItem('neurondesk_api_key', '%s');\n", key)
}
EOF



