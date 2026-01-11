//go:build verification_vector_operations
// +build verification_vector_operations

/*-------------------------------------------------------------------------
 *
 * test_vector_operations.go
 *    Comprehensive vector operations verification
 *
 * Tests NeuronMCP vector search operations with different distance metrics
 * to verify NeuronDB integration.
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
	"github.com/neurondb/NeuronMCP/internal/tools"
)

func main() {
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("NEURONMCP VECTOR OPERATIONS VERIFICATION")
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

	// Test 1: Create test table with vectors
	fmt.Println("\n[Test 1] Creating test table with vectors...")
	if err := createTestTable(db); err != nil {
		fmt.Printf("  ❌ FAILED: Could not create test table: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("  ✅ Test table created")

	// Test 2: Test all distance metrics
	fmt.Println("\n[Test 2] Testing vector search with different distance metrics...")
	testDistanceMetrics(db)

	// Test 3: Test vector similarity operations
	fmt.Println("\n[Test 3] Testing vector similarity operations...")
	testVectorSimilarity(db)

	// Test 4: Test vector arithmetic
	fmt.Println("\n[Test 4] Testing vector arithmetic operations...")
	testVectorArithmetic(db)

	// Test 5: Test vector quantization
	fmt.Println("\n[Test 5] Testing vector quantization...")
	testVectorQuantization(db)

	// Cleanup
	fmt.Println("\n[Cleanup] Removing test table...")
	if err := cleanupTestTable(db); err != nil {
		fmt.Printf("  ⚠️  WARNING: Could not cleanup test table: %v\n", err)
	} else {
		fmt.Println("  ✅ Test table removed")
	}

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("VERIFICATION COMPLETE")
	fmt.Println(strings.Repeat("=", 80))
}

func createTestTable(db *database.Database) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Drop table if exists
	_, _ = db.Exec(ctx, "DROP TABLE IF EXISTS test_vectors")

	// Create table
	createSQL := `
		CREATE TABLE test_vectors (
			id SERIAL PRIMARY KEY,
			name TEXT NOT NULL,
			embedding vector(3) NOT NULL,
			metadata JSONB
		)
	`

	if _, err := db.Exec(ctx, createSQL); err != nil {
		return fmt.Errorf("failed to create table: %w", err)
	}

	// Insert test vectors
	insertSQL := `
		INSERT INTO test_vectors (name, embedding, metadata) VALUES
		('vector1', '[1.0, 0.0, 0.0]'::vector, '{"category": "unit"}'),
		('vector2', '[0.0, 1.0, 0.0]'::vector, '{"category": "unit"}'),
		('vector3', '[0.0, 0.0, 1.0]'::vector, '{"category": "unit"}'),
		('vector4', '[0.5, 0.5, 0.0]'::vector, '{"category": "mixed"}'),
		('vector5', '[0.3, 0.3, 0.4]'::vector, '{"category": "mixed"}')
	`

	if _, err := db.Exec(ctx, insertSQL); err != nil {
		return fmt.Errorf("failed to insert test vectors: %w", err)
	}

	return nil
}

func testDistanceMetrics(db *database.Database) {
	executor := tools.NewQueryExecutor(db)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Test query vector
	queryVector := []interface{}{0.9, 0.1, 0.0}

	// Test all distance metrics
	metrics := []string{"l2", "cosine", "inner_product", "l1", "hamming", "chebyshev", "minkowski"}

	for _, metric := range metrics {
		fmt.Printf("\n  Testing %s distance...\n", metric)

		// For minkowski, we need to handle p parameter differently
		if metric == "minkowski" {
			// Skip minkowski for now as it requires additional parameter
			fmt.Printf("    ⏭️  SKIPPED: Minkowski requires p parameter (test separately)\n")
			continue
		}

		results, err := executor.ExecuteVectorSearch(
			ctx,
			"test_vectors",
			"embedding",
			queryVector,
			metric,
			3,
			[]interface{}{"id", "name"},
		)

		if err != nil {
			fmt.Printf("    ❌ FAILED: %v\n", err)
			continue
		}

		if len(results) == 0 {
			fmt.Printf("    ⚠️  WARNING: No results returned\n")
			continue
		}

		fmt.Printf("    ✅ PASSED: Found %d results\n", len(results))
		for i, result := range results {
			if i >= 3 {
				break
			}
			distance, _ := result["distance"]
			name, _ := result["name"]
			fmt.Printf("      Result %d: %v (distance: %v)\n", i+1, name, distance)
		}
	}
}

func testVectorSimilarity(db *database.Database) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Test vector similarity query
	query := `
		SELECT 
			v1.name AS vec1,
			v2.name AS vec2,
			v1.embedding <=> v2.embedding AS cosine_distance,
			v1.embedding <-> v2.embedding AS l2_distance
		FROM test_vectors v1
		CROSS JOIN test_vectors v2
		WHERE v1.id < v2.id
		ORDER BY cosine_distance
		LIMIT 5
	`

	rows, err := db.Query(ctx, query)
	if err != nil {
		fmt.Printf("  ❌ FAILED: Could not execute similarity query: %v\n", err)
		return
	}
	defer rows.Close()

	count := 0
	for rows.Next() {
		var vec1, vec2 string
		var cosDist, l2Dist interface{}
		if err := rows.Scan(&vec1, &vec2, &cosDist, &l2Dist); err != nil {
			fmt.Printf("  ⚠️  WARNING: Could not scan result: %v\n", err)
			continue
		}
		count++
		if count <= 3 {
			fmt.Printf("  ✅ Similarity: %s <-> %s (cosine: %v, l2: %v)\n", vec1, vec2, cosDist, l2Dist)
		}
	}

	if count > 0 {
		fmt.Printf("  ✅ PASSED: Vector similarity operations working (%d pairs tested)\n", count)
	} else {
		fmt.Printf("  ⚠️  WARNING: No similarity results returned\n")
	}
}

func testVectorArithmetic(db *database.Database) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Test vector addition
	query := `
		SELECT 
			('[1.0, 0.0, 0.0]'::vector + '[0.0, 1.0, 0.0]'::vector)::text AS addition,
			('[1.0, 0.0, 0.0]'::vector - '[0.0, 1.0, 0.0]'::vector)::text AS subtraction,
			('[1.0, 0.0, 0.0]'::vector * 2.0)::text AS multiplication
	`

	row := db.QueryRow(ctx, query)
	var addition, subtraction, multiplication string

	if err := row.Scan(&addition, &subtraction, &multiplication); err != nil {
		fmt.Printf("  ❌ FAILED: Could not execute arithmetic query: %v\n", err)
		return
	}

	fmt.Printf("  ✅ PASSED: Vector arithmetic operations working\n")
	fmt.Printf("    Addition: %s\n", addition)
	fmt.Printf("    Subtraction: %s\n", subtraction)
	fmt.Printf("    Multiplication: %s\n", multiplication)
}

func testVectorQuantization(db *database.Database) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Test if quantization functions exist
	query := `
		SELECT 
			proname 
		FROM pg_proc 
		WHERE proname LIKE '%quantize%' 
		AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb')
		LIMIT 5
	`

	rows, err := db.Query(ctx, query)
	if err != nil {
		fmt.Printf("  ⏭️  SKIPPED: Could not check quantization functions: %v\n", err)
		return
	}
	defer rows.Close()

	funcs := []string{}
	for rows.Next() {
		var funcName string
		if err := rows.Scan(&funcName); err != nil {
			continue
		}
		funcs = append(funcs, funcName)
	}

	if len(funcs) > 0 {
		fmt.Printf("  ✅ PASSED: Found %d quantization functions\n", len(funcs))
		for _, f := range funcs {
			fmt.Printf("    - %s\n", f)
		}
	} else {
		fmt.Printf("  ⚠️  WARNING: No quantization functions found (may not be available)\n")
	}
}

func cleanupTestTable(db *database.Database) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err := db.Exec(ctx, "DROP TABLE IF EXISTS test_vectors")
	return err
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

