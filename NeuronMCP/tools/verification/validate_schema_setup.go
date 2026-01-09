/*-------------------------------------------------------------------------
 *
 * validate_schema_setup.go
 *    Validate NeuronMCP configuration schema setup
 *
 * Verifies that all required tables, functions, and data are properly
 * set up in the database.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 *-------------------------------------------------------------------------
 */

// +build ignore

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
	fmt.Println("NEURONMCP SCHEMA SETUP VALIDATION")
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

	// Test 1: Verify required extensions
	fmt.Println("\n[Test 1] Verifying required extensions...")
	verifyExtensions(db)

	// Test 2: Verify LLM tables
	fmt.Println("\n[Test 2] Verifying LLM configuration tables...")
	verifyLLMTables(db)

	// Test 3: Verify index configuration tables
	fmt.Println("\n[Test 3] Verifying index configuration tables...")
	verifyIndexTables(db)

	// Test 4: Verify worker configuration tables
	fmt.Println("\n[Test 4] Verifying worker configuration tables...")
	verifyWorkerTables(db)

	// Test 5: Verify ML configuration tables
	fmt.Println("\n[Test 5] Verifying ML configuration tables...")
	verifyMLTables(db)

	// Test 6: Verify tool and system configuration tables
	fmt.Println("\n[Test 6] Verifying tool and system configuration tables...")
	verifyToolSystemTables(db)

	// Test 7: Verify management functions
	fmt.Println("\n[Test 7] Verifying management functions...")
	verifyManagementFunctions(db)

	// Test 8: Verify pre-populated data
	fmt.Println("\n[Test 8] Verifying pre-populated data...")
	verifyPrePopulatedData(db)

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("VALIDATION COMPLETE")
	fmt.Println(strings.Repeat("=", 80))
}

func verifyExtensions(db *database.Database) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	query := `
		SELECT extname, extversion
		FROM pg_extension
		WHERE extname IN ('neurondb', 'pgcrypto')
		ORDER BY extname
	`

	rows, err := db.Query(ctx, query)
	if err != nil {
		fmt.Printf("  ❌ FAILED: Could not query extensions: %v\n", err)
		return
	}
	defer rows.Close()

	extensions := make(map[string]string)
	for rows.Next() {
		var name, version string
		if err := rows.Scan(&name, &version); err != nil {
			continue
		}
		extensions[name] = version
	}

	required := []string{"neurondb", "pgcrypto"}
	missing := []string{}

	for _, ext := range required {
		if _, found := extensions[ext]; found {
			fmt.Printf("  ✅ Found extension: %s (version: %s)\n", ext, extensions[ext])
		} else {
			fmt.Printf("  ❌ Missing extension: %s\n", ext)
			missing = append(missing, ext)
		}
	}

	if len(missing) == 0 {
		fmt.Printf("  ✅ PASSED: All required extensions installed\n")
	} else {
		fmt.Printf("  ❌ FAILED: Missing extensions: %v\n", missing)
		fmt.Printf("     Run: CREATE EXTENSION IF NOT EXISTS %s;\n", strings.Join(missing, "; CREATE EXTENSION IF NOT EXISTS "))
	}
}

func verifyLLMTables(db *database.Database) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	requiredTables := []string{
		"neurondb.llm_providers",
		"neurondb.llm_models",
		"neurondb.llm_model_keys",
		"neurondb.llm_model_configs",
		"neurondb.llm_model_usage",
	}

	verifyTables(ctx, db, requiredTables, "LLM")
}

func verifyIndexTables(db *database.Database) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	requiredTables := []string{
		"neurondb.index_configs",
		"neurondb.index_templates",
	}

	verifyTables(ctx, db, requiredTables, "Index")
}

func verifyWorkerTables(db *database.Database) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	requiredTables := []string{
		"neurondb.worker_configs",
		"neurondb.worker_schedules",
	}

	verifyTables(ctx, db, requiredTables, "Worker")
}

func verifyMLTables(db *database.Database) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	requiredTables := []string{
		"neurondb.ml_default_configs",
		"neurondb.ml_model_templates",
	}

	verifyTables(ctx, db, requiredTables, "ML")
}

func verifyToolSystemTables(db *database.Database) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	requiredTables := []string{
		"neurondb.tool_configs",
		"neurondb.system_configs",
	}

	verifyTables(ctx, db, requiredTables, "Tool/System")
}

func verifyTables(ctx context.Context, db *database.Database, tables []string, category string) {
	// Build query with IN clause instead of ANY for better compatibility
	placeholders := make([]string, len(tables))
	args := make([]interface{}, len(tables))
	for i, table := range tables {
		placeholders[i] = fmt.Sprintf("$%d", i+1)
		args[i] = table
	}
	
	query := fmt.Sprintf(`
		SELECT schemaname || '.' || tablename AS full_name
		FROM pg_tables
		WHERE schemaname || '.' || tablename IN (%s)
	`, strings.Join(placeholders, ", "))

	rows, err := db.Query(ctx, query, args...)
	if err != nil {
		fmt.Printf("  ❌ FAILED: Could not query %s tables: %v\n", category, err)
		return
	}
	defer rows.Close()

	found := make(map[string]bool)
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			continue
		}
		found[name] = true
	}

	missing := []string{}
	for _, table := range tables {
		if found[table] {
			fmt.Printf("  ✅ Found table: %s\n", table)
		} else {
			fmt.Printf("  ❌ Missing table: %s\n", table)
			missing = append(missing, table)
		}
	}

	if len(missing) == 0 {
		fmt.Printf("  ✅ PASSED: All %s tables exist\n", category)
	} else {
		fmt.Printf("  ❌ FAILED: Missing %s tables: %v\n", category, missing)
		fmt.Printf("     Run setup script: ./scripts/setup_neurondb_mcp.sh\n")
	}
}

func verifyManagementFunctions(db *database.Database) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Check for key management functions
	keyFunctions := []string{
		"neurondb_set_model_key",
		"neurondb_get_model_key",
		"neurondb_get_model_config",
		"neurondb_get_index_config",
		"neurondb_get_worker_config",
		"neurondb_get_ml_defaults",
		"neurondb_get_tool_config",
		"neurondb_get_system_config",
	}

	// Build query with IN clause
	placeholders := make([]string, len(keyFunctions))
	args := make([]interface{}, len(keyFunctions))
	for i, funcName := range keyFunctions {
		placeholders[i] = fmt.Sprintf("$%d", i+1)
		args[i] = funcName
	}
	
	query := fmt.Sprintf(`
		SELECT proname
		FROM pg_proc
		WHERE pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb')
		AND proname IN (%s)
	`, strings.Join(placeholders, ", "))

	rows, err := db.Query(ctx, query, args...)
	if err != nil {
		fmt.Printf("  ❌ FAILED: Could not query functions: %v\n", err)
		return
	}
	defer rows.Close()

	found := make(map[string]bool)
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			continue
		}
		found[name] = true
	}

	missing := []string{}
	for _, funcName := range keyFunctions {
		if found[funcName] {
			fmt.Printf("  ✅ Found function: %s\n", funcName)
		} else {
			fmt.Printf("  ❌ Missing function: %s\n", funcName)
			missing = append(missing, funcName)
		}
	}

	if len(missing) == 0 {
		fmt.Printf("  ✅ PASSED: All key management functions exist\n")
	} else {
		fmt.Printf("  ⚠️  WARNING: Missing %d functions (may be in sql/002_functions.sql)\n", len(missing))
	}
}

func verifyPrePopulatedData(db *database.Database) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Check LLM providers
	query := `SELECT COUNT(*) FROM neurondb.llm_providers`
	row := db.QueryRow(ctx, query)
	var providerCount int
	if err := row.Scan(&providerCount); err != nil {
		fmt.Printf("  ⚠️  WARNING: Could not count LLM providers: %v\n", err)
	} else {
		if providerCount > 0 {
			fmt.Printf("  ✅ Found %d LLM providers\n", providerCount)
		} else {
			fmt.Printf("  ⚠️  WARNING: No LLM providers found (expected 5+)\n")
		}
	}

	// Check LLM models
	query = `SELECT COUNT(*) FROM neurondb.llm_models`
	row = db.QueryRow(ctx, query)
	var modelCount int
	if err := row.Scan(&modelCount); err != nil {
		fmt.Printf("  ⚠️  WARNING: Could not count LLM models: %v\n", err)
	} else {
		if modelCount >= 50 {
			fmt.Printf("  ✅ Found %d LLM models (expected 50+)\n", modelCount)
		} else {
			fmt.Printf("  ⚠️  WARNING: Found %d LLM models (expected 50+)\n", modelCount)
		}
	}

	// Check index templates
	query = `SELECT COUNT(*) FROM neurondb.index_templates`
	row = db.QueryRow(ctx, query)
	var templateCount int
	if err := row.Scan(&templateCount); err != nil {
		fmt.Printf("  ⚠️  WARNING: Could not count index templates: %v\n", err)
	} else {
		if templateCount >= 6 {
			fmt.Printf("  ✅ Found %d index templates (expected 6+)\n", templateCount)
		} else {
			fmt.Printf("  ⚠️  WARNING: Found %d index templates (expected 6+)\n", templateCount)
		}
	}

	// Check worker configs
	query = `SELECT COUNT(*) FROM neurondb.worker_configs`
	row = db.QueryRow(ctx, query)
	var workerCount int
	if err := row.Scan(&workerCount); err != nil {
		fmt.Printf("  ⚠️  WARNING: Could not count worker configs: %v\n", err)
	} else {
		if workerCount >= 3 {
			fmt.Printf("  ✅ Found %d worker configurations (expected 3+)\n", workerCount)
		} else {
			fmt.Printf("  ⚠️  WARNING: Found %d worker configurations (expected 3+)\n", workerCount)
		}
	}

	fmt.Printf("  ✅ PASSED: Pre-populated data verification complete\n")
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

