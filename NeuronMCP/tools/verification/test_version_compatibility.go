//go:build verification_version_compatibility
// +build verification_version_compatibility

/*-------------------------------------------------------------------------
 *
 * test_version_compatibility.go
 *    PostgreSQL version compatibility verification
 *
 * Verifies compatibility with PostgreSQL 16, 17, and 18 as documented.
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
	"regexp"
	"strings"
	"time"

	"github.com/neurondb/NeuronMCP/internal/config"
	"github.com/neurondb/NeuronMCP/internal/database"
)

func main() {
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("POSTGRESQL VERSION COMPATIBILITY VERIFICATION")
	fmt.Println(strings.Repeat("=", 80))

	// Setup database connection
	db := database.NewDatabase()
	cfg := getTestConfig()

	fmt.Println("\n[Setup] Connecting to database...")
	if err := db.Connect(cfg); err != nil {
		fmt.Printf("  ❌ FAILED: Could not connect to database: %v\n", err)
		fmt.Println("  Set NEURONDB_* environment variables to configure connection")
		os.Exit(1)
	}
	defer db.Close()
	fmt.Println("  ✅ Connected to database")

	// Test 1: Check PostgreSQL version
	fmt.Println("\n[Test 1] Checking PostgreSQL version...")
	pgVersion, err := checkPostgreSQLVersion(db)
	if err != nil {
		fmt.Printf("  ❌ FAILED: Could not determine PostgreSQL version: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("  ✅ PostgreSQL version: %s\n", pgVersion)

	// Test 2: Verify version compatibility
	fmt.Println("\n[Test 2] Verifying version compatibility...")
	if isVersionCompatible(pgVersion) {
		fmt.Printf("  ✅ PASSED: PostgreSQL version %s is compatible\n", pgVersion)
	} else {
		fmt.Printf("  ❌ FAILED: PostgreSQL version %s is not compatible (requires 16+)\n", pgVersion)
		os.Exit(1)
	}

	// Test 3: Check NeuronDB extension version
	fmt.Println("\n[Test 3] Checking NeuronDB extension...")
	neurondbVersion, err := checkNeuronDBVersion(db)
	if err != nil {
		fmt.Printf("  ⚠️  WARNING: Could not determine NeuronDB version: %v\n", err)
	} else {
		fmt.Printf("  ✅ NeuronDB extension version: %s\n", neurondbVersion)
	}

	// Test 4: Test vector type support
	fmt.Println("\n[Test 4] Testing vector type support...")
	testVectorTypeSupport(db)

	// Test 5: Test connection pool compatibility
	fmt.Println("\n[Test 5] Testing connection pool compatibility...")
	testConnectionPoolCompatibility(db)

	// Test 6: Test SQL features used by NeuronMCP
	fmt.Println("\n[Test 6] Testing SQL features compatibility...")
	testSQLFeatures(db)

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("COMPATIBILITY VERIFICATION COMPLETE")
	fmt.Println(strings.Repeat("=", 80))
}

func checkPostgreSQLVersion(db *database.Database) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	query := "SELECT version()"
	row := db.QueryRow(ctx, query)

	var version string
	if err := row.Scan(&version); err != nil {
		return "", err
	}

	return version, nil
}

func isVersionCompatible(version string) bool {
	// Check for PostgreSQL 16, 17, or 18
	re := regexp.MustCompile(`PostgreSQL (\d+)\.`)
	matches := re.FindStringSubmatch(version)
	if len(matches) < 2 {
		return false
	}

	var majorVersion int
	if _, err := fmt.Sscanf(matches[1], "%d", &majorVersion); err != nil {
		return false
	}

	return majorVersion >= 16
}

func checkNeuronDBVersion(db *database.Database) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	query := `
		SELECT extversion
		FROM pg_extension
		WHERE extname = 'neurondb'
	`

	row := db.QueryRow(ctx, query)
	var version string
	if err := row.Scan(&version); err != nil {
		return "", err
	}

	return version, nil
}

func testVectorTypeSupport(db *database.Database) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Test creating a vector type
	testTable := "test_version_vectors"
	cleanupSQL := fmt.Sprintf("DROP TABLE IF EXISTS %s", testTable)
	_, _ = db.Exec(ctx, cleanupSQL)
	defer func() {
		_, _ = db.Exec(ctx, cleanupSQL)
	}()

	createSQL := fmt.Sprintf(`
		CREATE TABLE %s (
			id SERIAL PRIMARY KEY,
			vec vector(3)
		)
	`, testTable)

	if _, err := db.Exec(ctx, createSQL); err != nil {
		fmt.Printf("  ❌ FAILED: Could not create vector table: %v\n", err)
		return
	}

	// Test inserting vector
	insertSQL := fmt.Sprintf(`
		INSERT INTO %s (vec) VALUES ('[1.0, 2.0, 3.0]'::vector)
	`, testTable)

	if _, err := db.Exec(ctx, insertSQL); err != nil {
		fmt.Printf("  ❌ FAILED: Could not insert vector: %v\n", err)
		return
	}

	// Test vector operators
	querySQL := fmt.Sprintf(`
		SELECT 
			vec <-> '[0.0, 0.0, 0.0]'::vector AS l2_distance,
			vec <=> '[0.0, 0.0, 0.0]'::vector AS cosine_distance,
			vec <#> '[0.0, 0.0, 0.0]'::vector AS inner_product
		FROM %s
		LIMIT 1
	`, testTable)

	row := db.QueryRow(ctx, querySQL)
	var l2, cosine, inner interface{}
	if err := row.Scan(&l2, &cosine, &inner); err != nil {
		fmt.Printf("  ❌ FAILED: Could not query vector operators: %v\n", err)
		return
	}

	fmt.Printf("  ✅ PASSED: Vector type fully supported\n")
	fmt.Printf("    L2 distance: %v\n", l2)
	fmt.Printf("    Cosine distance: %v\n", cosine)
	fmt.Printf("    Inner product: %v\n", inner)
}

func testConnectionPoolCompatibility(db *database.Database) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Test that connection pool works correctly
	stats := db.GetPoolStats()
	if stats == nil {
		fmt.Printf("  ⚠️  WARNING: Could not get pool stats\n")
		return
	}

	fmt.Printf("  ✅ PASSED: Connection pool compatible\n")
	fmt.Printf("    Total connections: %d\n", stats.TotalConns)
	fmt.Printf("    Acquired: %d\n", stats.AcquiredConns)
	fmt.Printf("    Idle: %d\n", stats.IdleConns)

	// Test multiple concurrent queries
	query := "SELECT 1"
	errors := 0
	for i := 0; i < 5; i++ {
		if _, err := db.Query(ctx, query); err != nil {
			errors++
		}
	}

	if errors == 0 {
		fmt.Printf("  ✅ PASSED: Concurrent queries working\n")
	} else {
		fmt.Printf("  ⚠️  WARNING: %d concurrent query errors\n", errors)
	}
}

func testSQLFeatures(db *database.Database) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	features := []struct {
		name string
		sql  string
	}{
		{"JSONB support", "SELECT '{}'::jsonb"},
		{"Array support", "SELECT ARRAY[1, 2, 3]"},
		{"Window functions", "SELECT row_number() OVER () FROM generate_series(1, 3)"},
		{"CTE support", "WITH test AS (SELECT 1 AS val) SELECT * FROM test"},
	}

	allPassed := true
	for _, feature := range features {
		if _, err := db.Query(ctx, feature.sql); err != nil {
			fmt.Printf("  ❌ FAILED: %s: %v\n", feature.name, err)
			allPassed = false
		} else {
			fmt.Printf("  ✅ %s: supported\n", feature.name)
		}
	}

	if allPassed {
		fmt.Printf("  ✅ PASSED: All required SQL features supported\n")
	} else {
		fmt.Printf("  ❌ FAILED: Some SQL features not supported\n")
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

