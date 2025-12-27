package main

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"database/sql"
	"fmt"
	"os"
	"strings"
	"time"

	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/google/uuid"
	"golang.org/x/crypto/bcrypt"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run generate_api_key.go <database_dsn> [user_id] [rate_limit]")
		fmt.Println("Example: go run generate_api_key.go 'postgresql://neurondesk:neurondesk@localhost:5432/neurondesk' nbduser 100")
		os.Exit(1)
	}

	dsn := os.Args[1]
	userID := "nbduser"
	if len(os.Args) > 2 {
		userID = os.Args[2]
	}
	rateLimit := 100
	if len(os.Args) > 3 {
		fmt.Sscanf(os.Args[3], "%d", &rateLimit)
	}

	// Connect to database
	db, err := sql.Open("pgx", dsn)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to connect to database: %v\n", err)
		os.Exit(1)
	}
	defer db.Close()

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := db.PingContext(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to ping database: %v\n", err)
		os.Exit(1)
	}

	// Generate API key
	keyBytes := make([]byte, 32)
	if _, err := rand.Read(keyBytes); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to generate key: %v\n", err)
		os.Exit(1)
	}

	key := base64.URLEncoding.EncodeToString(keyBytes)
	keyPrefix := key[:8]
	
	// Hash the key
	keyHash, err := bcrypt.GenerateFromPassword([]byte(key), bcrypt.DefaultCost)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to hash key: %v\n", err)
		os.Exit(1)
	}

	keyID := uuid.New().String()

	// Insert into database
	query := `
		INSERT INTO api_keys (id, key_hash, key_prefix, user_id, rate_limit, created_at)
		VALUES ($1, $2, $3, $4, $5, NOW())
		RETURNING id, created_at
	`
	
	var createdID string
	var createdAt time.Time
	err = db.QueryRowContext(ctx, query, keyID, string(keyHash), keyPrefix, userID, rateLimit).Scan(&createdID, &createdAt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to insert API key: %v\n", err)
		os.Exit(1)
	}

	// Output the key
	fmt.Println("=" + strings.Repeat("=", 70))
	fmt.Println("API Key Generated Successfully!")
	fmt.Println("=" + strings.Repeat("=", 70))
	fmt.Printf("Key ID:     %s\n", createdID)
	fmt.Printf("Key Prefix: %s\n", keyPrefix)
	fmt.Printf("User ID:    %s\n", userID)
	fmt.Printf("Rate Limit: %d requests/minute\n", rateLimit)
	fmt.Println("")
	fmt.Println("FULL API KEY (save this - it won't be shown again):")
	fmt.Println(strings.Repeat("-", 72))
	fmt.Println(key)
	fmt.Println(strings.Repeat("-", 72))
	fmt.Println("")
	fmt.Println("To use this key:")
	fmt.Println("1. Copy the full API key above")
	fmt.Println("2. Go to NeuronDesktop Settings page")
	fmt.Println("3. Paste it in the 'API Key' field")
	fmt.Println("4. Click 'Save API Key'")
	fmt.Println("")
	fmt.Println("Or set it directly in browser console:")
	fmt.Printf("  localStorage.setItem('neurondesk_api_key', '%s');\n", key)
	fmt.Println("")
}

