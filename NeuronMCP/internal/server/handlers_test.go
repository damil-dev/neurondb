package server

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/neurondb/NeuronMCP/internal/config"
	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/tools"
	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

func TestServerSetup(t *testing.T) {
	// Test that server can be created (may fail if no database, but that's OK)
	cfgMgr := config.NewConfigManager()
	cfgMgr.Load("") // Load defaults

	logger := logging.NewLogger(cfgMgr.GetLoggingConfig())
	db := database.NewDatabase()

	toolRegistry := tools.NewToolRegistry(db, logger)
	tools.RegisterAllTools(toolRegistry, db, logger)

	// Verify tools are registered
	definitions := toolRegistry.GetAllDefinitions()
	if len(definitions) == 0 {
		t.Error("No tools registered")
	}

	// Check that we have expected tools
	toolNames := make(map[string]bool)
	for _, def := range definitions {
		toolNames[def.Name] = true
	}

	expectedTools := []string{"vector_search", "train_model", "predict", "cluster_data"}
	for _, expected := range expectedTools {
		if !toolNames[expected] {
			t.Errorf("Missing expected tool: %s", expected)
		}
	}
}

func TestToolRegistry(t *testing.T) {
	cfgMgr := config.NewConfigManager()
	cfgMgr.Load("")

	logger := logging.NewLogger(cfgMgr.GetLoggingConfig())
	db := database.NewDatabase()

	toolRegistry := tools.NewToolRegistry(db, logger)
	tools.RegisterAllTools(toolRegistry, db, logger)

	// Test that nonexistent tool returns nil
	tool := toolRegistry.GetTool("nonexistent_tool")
	if tool != nil {
		t.Error("GetTool() should return nil for nonexistent tool")
	}

	// Test that existing tool is found
	tool = toolRegistry.GetTool("vector_search")
	if tool == nil {
		t.Error("GetTool() should return tool for 'vector_search'")
	}
}

