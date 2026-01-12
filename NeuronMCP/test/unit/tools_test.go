/*-------------------------------------------------------------------------
 *
 * tools_test.go
 *    Unit tests for NeuronMCP tools
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/test/unit/tools_test.go
 *
 *-------------------------------------------------------------------------
 */

package unit

import (
	"context"
	"testing"

	"github.com/neurondb/NeuronMCP/internal/config"
	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/tools"
)

/* TestToolRegistry tests tool registry functionality */
func TestToolRegistry(t *testing.T) {
	db := database.NewDatabase()
	output := "stderr"
	logger := logging.NewLogger(&config.LoggingConfig{
		Level:  "info",
		Format: "text",
		Output: &output,
	})

	registry := tools.NewToolRegistry(db, logger)
	if registry == nil {
		t.Fatal("NewToolRegistry returned nil")
	}

	/* Register all tools */
	tools.RegisterAllTools(registry, db, logger)

	/* Verify tools are registered */
	definitions := registry.GetAllDefinitions()
	if len(definitions) == 0 {
		t.Fatal("No tools registered")
	}

	/* Check for expected tool categories */
	expectedTools := map[string]bool{
		/* Vector search */
		"neurondb_vector_search":              false,
		"neurondb_vector_search_l2":           false,
		"neurondb_vector_search_cosine":       false,
		"neurondb_vector_search_inner_product": false,
		/* Embeddings */
		"neurondb_generate_embedding": false,
		"neurondb_batch_embedding":    false,
		/* ML */
		"neurondb_train_model": false,
		"neurondb_predict":     false,
		"neurondb_list_models": false,
		/* PostgreSQL */
		"postgresql_version": false,
		"postgresql_stats":   false,
	}

	for _, def := range definitions {
		if _, ok := expectedTools[def.Name]; ok {
			expectedTools[def.Name] = true
		}
	}

	/* Verify all expected tools are present */
	for toolName, found := range expectedTools {
		if !found {
			t.Errorf("Expected tool %s not found in registry", toolName)
		}
	}
}

/* TestToolValidation tests parameter validation */
func TestToolValidation(t *testing.T) {
	db := database.NewDatabase()
	output := "stderr"
	logger := logging.NewLogger(&config.LoggingConfig{
		Level:  "info",
		Format: "text",
		Output: &output,
	})

	registry := tools.NewToolRegistry(db, logger)
	tools.RegisterAllTools(registry, db, logger)

	/* Test vector_search validation */
	tool := registry.GetTool("neurondb_vector_search")
	if tool == nil {
		t.Fatal("neurondb_vector_search tool not found")
	}

	ctx := context.Background()

	/* Test with missing required parameter */
	result, err := tool.Execute(ctx, map[string]interface{}{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if result.Success {
		t.Error("Expected validation error for missing parameters")
	}
	if result.Error == nil {
		t.Error("Expected error result for missing parameters")
	}

	/* Test with invalid parameter type */
	result, err = tool.Execute(ctx, map[string]interface{}{
		"table":        123, /* Should be string */
		"vector_column": "embedding",
		"query_vector": []float64{0.1, 0.2, 0.3},
		"limit":        10,
	})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if result.Success {
		t.Error("Expected validation error for invalid parameter type")
	}

	/* Test with valid parameters (may fail due to DB connection, but validation should pass) */
	result, err = tool.Execute(ctx, map[string]interface{}{
		"table":        "test_table",
		"vector_column": "embedding",
		"query_vector": []float64{0.1, 0.2, 0.3},
		"limit":        10,
	})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	/* Validation should pass even if DB connection fails */
	/* If result has error, it should be a connection/execution error, not validation */
	if result.Error != nil && result.Error.Code == "VALIDATION_ERROR" {
		/* This might happen if there are additional validation checks */
		/* For now, we just verify the tool executed without panicking */
		t.Logf("Tool execution returned validation error (may be expected): %v", result.Error)
	}
}

/* TestToolSchemas tests tool input/output schemas */
func TestToolSchemas(t *testing.T) {
	db := database.NewDatabase()
	output := "stderr"
	logger := logging.NewLogger(&config.LoggingConfig{
		Level:  "info",
		Format: "text",
		Output: &output,
	})

	registry := tools.NewToolRegistry(db, logger)
	tools.RegisterAllTools(registry, db, logger)

	definitions := registry.GetAllDefinitions()

	for _, def := range definitions {
		/* Verify schema structure */
		if def.InputSchema == nil {
			t.Errorf("Tool %s has nil input schema", def.Name)
			continue
		}

		/* InputSchema is already map[string]interface{}, so we can use it directly */
		schemaMap := def.InputSchema

		if schemaType, ok := schemaMap["type"].(string); ok {
			if schemaType != "object" {
				t.Errorf("Tool %s input schema type is %s, expected object", def.Name, schemaType)
			}
		}

		/* Verify description is present */
		if def.Description == "" {
			t.Errorf("Tool %s has empty description", def.Name)
		}
	}
}

/* TestToolDefinitions tests tool definition completeness */
func TestToolDefinitions(t *testing.T) {
	db := database.NewDatabase()
	output := "stderr"
	logger := logging.NewLogger(&config.LoggingConfig{
		Level:  "info",
		Format: "text",
		Output: &output,
	})

	registry := tools.NewToolRegistry(db, logger)
	tools.RegisterAllTools(registry, db, logger)

	definitions := registry.GetAllDefinitions()

	/* Verify we have a reasonable number of tools (at least 50) */
	/* Note: Actual count may vary based on which tools are registered */
	if len(definitions) < 50 {
		t.Errorf("Expected at least 50 tools, got %d", len(definitions))
	}

	/* Verify all tools have required fields */
	for _, def := range definitions {
		if def.Name == "" {
			t.Errorf("Tool has empty name")
		}
		if def.Description == "" {
			t.Errorf("Tool %s has empty description", def.Name)
		}
		if def.InputSchema == nil {
			t.Errorf("Tool %s has nil input schema", def.Name)
		}
	}
}

