/*-------------------------------------------------------------------------
 *
 * test_tool_execution.go
 *    Tool execution flow verification
 *
 * Tests tool execution flow from MCP client through to NeuronDB function calls.
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
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/tools"
)

func main() {
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("TOOL EXECUTION FLOW VERIFICATION")
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

	// Setup logger
	logCfg := &config.LoggingConfig{
		Level:  "info",
		Format: "text",
	}
	logger := logging.NewLogger(logCfg)

	// Setup tool registry
	fmt.Println("\n[Setup] Initializing tool registry...")
	registry := tools.NewToolRegistry(db, logger)
	tools.RegisterAllTools(registry, db, logger)
	fmt.Printf("  ✅ Tool registry initialized with %d tools\n", registry.GetCount())

	// Test 1: Test vector search tool
	fmt.Println("\n[Test 1] Testing vector_search tool execution...")
	testVectorSearchTool(registry, db)

	// Test 2: Test PostgreSQL tool
	fmt.Println("\n[Test 2] Testing postgresql_version tool execution...")
	testPostgreSQLTool(registry)

	// Test 3: Test tool parameter validation
	fmt.Println("\n[Test 3] Testing tool parameter validation...")
	testToolValidation(registry)

	// Test 4: Test tool error handling
	fmt.Println("\n[Test 4] Testing tool error handling...")
	testToolErrorHandling(registry, db)

	// Test 5: Test tool execution with timeout
	fmt.Println("\n[Test 5] Testing tool execution timeout...")
	testToolTimeout(registry, db)

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("TOOL EXECUTION VERIFICATION COMPLETE")
	fmt.Println(strings.Repeat("=", 80))
}

func testVectorSearchTool(registry *tools.ToolRegistry, db *database.Database) {
	// Create test table
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	testTable := "test_tool_vectors"
	cleanupSQL := fmt.Sprintf("DROP TABLE IF EXISTS %s", testTable)
	_, _ = db.Exec(ctx, cleanupSQL)
	defer func() {
		_, _ = db.Exec(ctx, cleanupSQL)
	}()

	createSQL := fmt.Sprintf(`
		CREATE TABLE %s (
			id SERIAL PRIMARY KEY,
			name TEXT,
			embedding vector(3)
		)
	`, testTable)

	if _, err := db.Exec(ctx, createSQL); err != nil {
		fmt.Printf("  ❌ FAILED: Could not create test table: %v\n", err)
		return
	}

	insertSQL := fmt.Sprintf(`
		INSERT INTO %s (name, embedding) VALUES
		('test1', '[1.0, 0.0, 0.0]'::vector),
		('test2', '[0.0, 1.0, 0.0]'::vector),
		('test3', '[0.0, 0.0, 1.0]'::vector)
	`, testTable)

	if _, err := db.Exec(ctx, insertSQL); err != nil {
		fmt.Printf("  ❌ FAILED: Could not insert test data: %v\n", err)
		return
	}

	// Get vector_search tool
	tool := registry.GetTool("vector_search")
	if tool == nil {
		fmt.Printf("  ❌ FAILED: vector_search tool not found\n")
		return
	}

	// Execute tool
	params := map[string]interface{}{
		"table":         testTable,
		"vector_column": "embedding",
		"query_vector":  []interface{}{0.9, 0.1, 0.0},
		"limit":         3,
		"distance_metric": "l2",
	}

	result, err := tool.Execute(ctx, params)
	if err != nil {
		fmt.Printf("  ❌ FAILED: Tool execution error: %v\n", err)
		return
	}

	if !result.Success {
		fmt.Printf("  ❌ FAILED: Tool execution failed: %v\n", result.Error)
		return
	}

	fmt.Printf("  ✅ PASSED: vector_search tool executed successfully\n")
	if data, ok := result.Data.(map[string]interface{}); ok {
		if results, ok := data["results"].([]interface{}); ok {
			fmt.Printf("    Found %d results\n", len(results))
		}
	}
}

func testPostgreSQLTool(registry *tools.ToolRegistry) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	tool := registry.GetTool("postgresql_version")
	if tool == nil {
		fmt.Printf("  ❌ FAILED: postgresql_version tool not found\n")
		return
	}

	params := map[string]interface{}{}
	result, err := tool.Execute(ctx, params)
	if err != nil {
		fmt.Printf("  ❌ FAILED: Tool execution error: %v\n", err)
		return
	}

	if !result.Success {
		fmt.Printf("  ❌ FAILED: Tool execution failed: %v\n", result.Error)
		return
	}

	fmt.Printf("  ✅ PASSED: postgresql_version tool executed successfully\n")
	if data, ok := result.Data.(map[string]interface{}); ok {
		if version, ok := data["version"].(string); ok {
			fmt.Printf("    PostgreSQL version: %s\n", version)
		}
	}
}

func testToolValidation(registry *tools.ToolRegistry) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	tool := registry.GetTool("vector_search")
	if tool == nil {
		fmt.Printf("  ❌ FAILED: vector_search tool not found\n")
		return
	}

	// Test with missing required parameter
	params := map[string]interface{}{
		"table": "test_table",
		// Missing vector_column and query_vector
	}

	result, err := tool.Execute(ctx, params)
	if err != nil {
		fmt.Printf("  ❌ FAILED: Unexpected error: %v\n", err)
		return
	}

	if result.Success {
		fmt.Printf("  ❌ FAILED: Tool should have failed validation\n")
		return
	}

	if result.Error != nil && strings.Contains(result.Error.Message, "required") {
		fmt.Printf("  ✅ PASSED: Tool correctly validates required parameters\n")
		fmt.Printf("    Error: %s\n", result.Error.Message)
	} else {
		fmt.Printf("  ⚠️  WARNING: Validation error format unexpected\n")
	}
}

func testToolErrorHandling(registry *tools.ToolRegistry, db *database.Database) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	tool := registry.GetTool("vector_search")
	if tool == nil {
		fmt.Printf("  ❌ FAILED: vector_search tool not found\n")
		return
	}

	// Test with non-existent table
	params := map[string]interface{}{
		"table":         "non_existent_table_xyz",
		"vector_column": "embedding",
		"query_vector":  []interface{}{0.1, 0.2, 0.3},
		"limit":         10,
	}

	result, err := tool.Execute(ctx, params)
	if err != nil {
		fmt.Printf("  ❌ FAILED: Unexpected error: %v\n", err)
		return
	}

	if result.Success {
		fmt.Printf("  ❌ FAILED: Tool should have failed for non-existent table\n")
		return
	}

	if result.Error != nil {
		fmt.Printf("  ✅ PASSED: Tool correctly handles errors\n")
		fmt.Printf("    Error: %s\n", result.Error.Message)
	} else {
		fmt.Printf("  ⚠️  WARNING: Error not properly returned\n")
	}
}

func testToolTimeout(registry *tools.ToolRegistry, db *database.Database) {
	// Create a context with very short timeout
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
	defer cancel()

	tool := registry.GetTool("postgresql_version")
	if tool == nil {
		fmt.Printf("  ❌ FAILED: postgresql_version tool not found\n")
		return
	}

	params := map[string]interface{}{}
	result, err := tool.Execute(ctx, params)

	// Tool should either timeout or complete quickly
	if err != nil && err == context.DeadlineExceeded {
		fmt.Printf("  ✅ PASSED: Tool correctly respects context timeout\n")
	} else if result != nil && result.Success {
		fmt.Printf("  ✅ PASSED: Tool completed before timeout\n")
	} else {
		fmt.Printf("  ⚠️  WARNING: Timeout behavior unexpected\n")
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

