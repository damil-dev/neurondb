//go:build verification
// +build verification

/*-------------------------------------------------------------------------
 *
 * test_connection_verification.go
 *    Comprehensive database connection and retry logic verification
 *
 * Tests NeuronMCP database connection handling, retry logic, and
 * compatibility with NeuronDB.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 *-------------------------------------------------------------------------
 */

package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/neurondb/NeuronMCP/internal/config"
	"github.com/neurondb/NeuronMCP/internal/database"
)

func main() {
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("NEURONMCP DATABASE CONNECTION VERIFICATION")
	fmt.Println(strings.Repeat("=", 80))

	// Test 1: Connection with retry logic
	fmt.Println("\n[Test 1] Testing connection with retry logic...")
	testConnectionRetry()

	// Test 2: Connection pool configuration
	fmt.Println("\n[Test 2] Testing connection pool configuration...")
	testConnectionPool()

	// Test 3: Type registration
	fmt.Println("\n[Test 3] Testing NeuronDB type registration...")
	testTypeRegistration()

	// Test 4: Connection health checks
	fmt.Println("\n[Test 4] Testing connection health checks...")
	testHealthChecks()

	// Test 5: Error handling
	fmt.Println("\n[Test 5] Testing error handling...")
	testErrorHandling()

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("VERIFICATION COMPLETE")
	fmt.Println(strings.Repeat("=", 80))
}

func testConnectionRetry() {
	db := database.NewDatabase()
	cfg := getTestConfig()

	// Test with retry (3 attempts, 2 second delay)
	start := time.Now()
	err := db.ConnectWithRetry(cfg, 3, 2*time.Second)
	duration := time.Since(start)

	if err != nil {
		fmt.Printf("  ❌ FAILED: Connection failed after retries: %v\n", err)
		fmt.Printf("     Duration: %v\n", duration)
		return
	}

	fmt.Printf("  ✅ PASSED: Connection successful\n")
	fmt.Printf("     Duration: %v\n", duration)
	fmt.Printf("     Host: %s:%d\n", cfg.GetHost(), cfg.GetPort())
	fmt.Printf("     Database: %s\n", cfg.GetDatabase())
	fmt.Printf("     User: %s\n", cfg.GetUser())

	// Verify connection is actually working
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := db.TestConnection(ctx); err != nil {
		fmt.Printf("  ❌ FAILED: Connection test failed: %v\n", err)
	} else {
		fmt.Printf("  ✅ PASSED: Connection test successful\n")
	}

	db.Close()
}

func testConnectionPool() {
	db := database.NewDatabase()
	cfg := getTestConfig()

	// Configure pool settings
	minConns := 2
	maxConns := 10
	idleTimeoutMs := int(5 * time.Minute / time.Millisecond)
	poolCfg := &config.PoolConfig{
		Min:             &minConns,
		Max:             &maxConns,
		IdleTimeoutMillis: &idleTimeoutMs,
	}
	cfg.Pool = poolCfg

	err := db.Connect(cfg)
	if err != nil {
		fmt.Printf("  ❌ FAILED: Connection with pool config failed: %v\n", err)
		return
	}

	stats := db.GetPoolStats()
	if stats == nil {
		fmt.Printf("  ❌ FAILED: Could not get pool stats\n")
	} else {
		fmt.Printf("  ✅ PASSED: Pool stats retrieved\n")
		fmt.Printf("     Total connections: %d\n", stats.TotalConns)
		fmt.Printf("     Acquired: %d\n", stats.AcquiredConns)
		fmt.Printf("     Idle: %d\n", stats.IdleConns)
		fmt.Printf("     Constructing: %d\n", stats.ConstructingConns)
	}

	db.Close()
}

func testTypeRegistration() {
	db := database.NewDatabase()
	cfg := getTestConfig()

	err := db.Connect(cfg)
	if err != nil {
		fmt.Printf("  ❌ FAILED: Connection failed: %v\n", err)
		return
	}

	// Test that vector type is registered by querying pg_type
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	query := `
		SELECT oid, typname 
		FROM pg_type 
		WHERE oid IN (17648, 17656)
		ORDER BY oid
	`

	rows, err := db.Query(ctx, query)
	if err != nil {
		fmt.Printf("  ❌ FAILED: Could not query pg_type: %v\n", err)
		db.Close()
		return
	}
	defer rows.Close()

	typeCount := 0
	for rows.Next() {
		var oid int32
		var typname string
		if err := rows.Scan(&oid, &typname); err != nil {
			fmt.Printf("  ⚠️  WARNING: Could not scan type row: %v\n", err)
			continue
		}
		typeCount++
		fmt.Printf("  ✅ Found type: %s (OID: %d)\n", typname, oid)
	}

	if typeCount == 0 {
		fmt.Printf("  ⚠️  WARNING: Vector types not found in pg_type (may be normal if extension not fully loaded)\n")
	} else if typeCount >= 2 {
		fmt.Printf("  ✅ PASSED: Vector types registered (found %d types)\n", typeCount)
	} else {
		fmt.Printf("  ⚠️  WARNING: Only found %d vector type(s), expected 2\n", typeCount)
	}

	db.Close()
}

func testHealthChecks() {
	db := database.NewDatabase()
	cfg := getTestConfig()

	err := db.Connect(cfg)
	if err != nil {
		fmt.Printf("  ❌ FAILED: Connection failed: %v\n", err)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Test connection health
	if err := db.TestConnection(ctx); err != nil {
		fmt.Printf("  ❌ FAILED: Health check failed: %v\n", err)
	} else {
		fmt.Printf("  ✅ PASSED: Health check successful\n")
	}

	// Test IsConnected
	if !db.IsConnected() {
		fmt.Printf("  ❌ FAILED: IsConnected() returned false\n")
	} else {
		fmt.Printf("  ✅ PASSED: IsConnected() returned true\n")
	}

	// Test pool stats
	stats := db.GetPoolStats()
	if stats != nil {
		fmt.Printf("  ✅ PASSED: Pool stats available\n")
		fmt.Printf("     Total: %d, Acquired: %d, Idle: %d\n",
			stats.TotalConns, stats.AcquiredConns, stats.IdleConns)
	} else {
		fmt.Printf("  ⚠️  WARNING: Pool stats not available\n")
	}

	db.Close()

	// Test IsConnected after close
	if db.IsConnected() {
		fmt.Printf("  ⚠️  WARNING: IsConnected() returned true after Close()\n")
	} else {
		fmt.Printf("  ✅ PASSED: IsConnected() correctly returns false after Close()\n")
	}
}

func testErrorHandling() {
	db := database.NewDatabase()

	// Test 1: Query without connection
	ctx := context.Background()
	_, err := db.Query(ctx, "SELECT 1")
	if err == nil {
		fmt.Printf("  ❌ FAILED: Query should fail when not connected\n")
	} else {
		fmt.Printf("  ✅ PASSED: Query correctly fails when not connected\n")
		fmt.Printf("     Error: %v\n", err)
	}

	// Test 2: QueryRow without connection
	row := db.QueryRow(ctx, "SELECT 1")
	var result int
	err = row.Scan(&result)
	if err == nil {
		fmt.Printf("  ❌ FAILED: QueryRow should fail when not connected\n")
	} else {
		fmt.Printf("  ✅ PASSED: QueryRow correctly fails when not connected\n")
	}

	// Test 3: Exec without connection
	_, err = db.Exec(ctx, "SELECT 1")
	if err == nil {
		fmt.Printf("  ❌ FAILED: Exec should fail when not connected\n")
	} else {
		fmt.Printf("  ✅ PASSED: Exec correctly fails when not connected\n")
	}

	// Test 4: Invalid connection string
	invalidHost := "invalid-host-that-does-not-exist"
	invalidPort := 5432
	invalidDB := "test"
	invalidUser := "test"
	invalidPwd := "test"
	invalidCfg := &config.DatabaseConfig{
		Host:     &invalidHost,
		Port:     &invalidPort,
		Database: &invalidDB,
		User:     &invalidUser,
		Password: &invalidPwd,
	}

	err = db.ConnectWithRetry(invalidCfg, 2, 1*time.Second)
	if err == nil {
		fmt.Printf("  ❌ FAILED: Should fail to connect to invalid host\n")
	} else {
		fmt.Printf("  ✅ PASSED: Correctly fails to connect to invalid host\n")
		fmt.Printf("     Error: %v\n", err)
	}
}

func getTestConfig() *config.DatabaseConfig {
	host := getEnv("NEURONDB_HOST", "localhost")
	port := getEnvInt("NEURONDB_PORT", 5432)
	database := getEnv("NEURONDB_DATABASE", "neurondb")
	user := getEnv("NEURONDB_USER", "neurondb")

	cfg := &config.DatabaseConfig{
		Host:     &host,
		Port:     &port,
		Database: &database,
		User:     &user,
	}

	if pwd := os.Getenv("NEURONDB_PASSWORD"); pwd != "" {
		cfg.Password = &pwd
	}

	if connStr := os.Getenv("NEURONDB_CONNECTION_STRING"); connStr != "" {
		cfg.ConnectionString = &connStr
	}

	return cfg
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		var result int
		if _, err := fmt.Sscanf(value, "%d", &result); err == nil {
			return result
		}
	}
	return defaultValue
}

func stringPtr(s string) *string {
	return &s
}

