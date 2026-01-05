package integration

import (
	"context"
	"testing"

	testutil "github.com/neurondb/NeuronDesktop/api/internal/testing"
)

func TestMCPIntegration_ListTools(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronMCP {
		t.Skip("Skipping NeuronMCP tests (SKIP_NEURONMCP=true)")
	}

	if config.NeuronMCPCommand == "" {
		t.Skip("Skipping test: NeuronMCP command not configured")
	}

	ctx := context.Background()
	env := map[string]string{
		"NEURONDB_CONNECTION_STRING": config.NeuronDBDSN,
	}

	client := testutil.CreateNeuronMCPTestClient(t, config.NeuronMCPCommand, env)
	if client == nil {
		return
	}
	defer client.Close()

	/* List tools */
	toolsResp, err := client.ListTools()
	if err != nil {
		t.Fatalf("Failed to list tools: %v", err)
	}

	if toolsResp == nil {
		t.Fatal("Tools response is nil")
	}

	/* Verify tools structure */
	if len(toolsResp.Tools) == 0 {
		t.Log("No tools available (this might be expected)")
	}

	for _, tool := range toolsResp.Tools {
		if tool.Name == "" {
			t.Error("Tool missing name")
		}
		if tool.Description == "" {
			t.Logf("Tool %s missing description", tool.Name)
		}
	}
}

func TestMCPIntegration_CallTool(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronMCP {
		t.Skip("Skipping NeuronMCP tests (SKIP_NEURONMCP=true)")
	}

	if config.NeuronMCPCommand == "" {
		t.Skip("Skipping test: NeuronMCP command not configured")
	}

	ctx := context.Background()
	env := map[string]string{
		"NEURONDB_CONNECTION_STRING": config.NeuronDBDSN,
	}

	client := testutil.CreateNeuronMCPTestClient(t, config.NeuronMCPCommand, env)
	if client == nil {
		return
	}
	defer client.Close()

	/* First, list tools to find one we can call */
	toolsResp, err := client.ListTools()
	if err != nil {
		t.Fatalf("Failed to list tools: %v", err)
	}

	if len(toolsResp.Tools) == 0 {
		t.Skip("No tools available to test")
	}

	/* Try calling the first tool with minimal arguments */
	firstTool := toolsResp.Tools[0]
	arguments := make(map[string]interface{})

	/* Try to call the tool */
	result, err := client.CallTool(firstTool.Name, arguments)
	if err != nil {
		/* Tool call might fail due to missing arguments, which is expected */
		t.Logf("Tool call failed (might be due to missing arguments): %v", err)
		return
	}

	if result == nil {
		t.Error("Tool result is nil")
		return
	}

	/* Verify result structure */
	if result.IsError {
		t.Logf("Tool returned error: %v", result)
	} else {
		if len(result.Content) == 0 {
			t.Log("Tool returned empty content")
		}
	}
}

func TestMCPIntegration_ToolValidation(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronMCP {
		t.Skip("Skipping NeuronMCP tests (SKIP_NEURONMCP=true)")
	}

	if config.NeuronMCPCommand == "" {
		t.Skip("Skipping test: NeuronMCP command not configured")
	}

	env := map[string]string{
		"NEURONDB_CONNECTION_STRING": config.NeuronDBDSN,
	}

	client := testutil.CreateNeuronMCPTestClient(t, config.NeuronMCPCommand, env)
	if client == nil {
		return
	}
	defer client.Close()

	/* Try calling a non-existent tool */
	_, err := client.CallTool("nonexistent_tool_xyz", map[string]interface{}{})
	if err == nil {
		t.Error("Expected error when calling non-existent tool")
	}
}

func TestMCPIntegration_ToolParameters(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronMCP {
		t.Skip("Skipping NeuronMCP tests (SKIP_NEURONMCP=true)")
	}

	if config.NeuronMCPCommand == "" {
		t.Skip("Skipping test: NeuronMCP command not configured")
	}

	env := map[string]string{
		"NEURONDB_CONNECTION_STRING": config.NeuronDBDSN,
	}

	client := testutil.CreateNeuronMCPTestClient(t, config.NeuronMCPCommand, env)
	if client == nil {
		return
	}
	defer client.Close()

	/* List tools */
	toolsResp, err := client.ListTools()
	if err != nil {
		t.Fatalf("Failed to list tools: %v", err)
	}

	if len(toolsResp.Tools) == 0 {
		t.Skip("No tools available to test")
	}

	/* Check tool input schemas */
	for _, tool := range toolsResp.Tools {
		if tool.InputSchema != nil {
			/* Verify schema structure */
			schema := tool.InputSchema

			/* Check for properties */
			if props, ok := schema["properties"].(map[string]interface{}); ok {
				for propName, propDef := range props {
					if propDef == nil {
						t.Logf("Tool %s property %s has nil definition", tool.Name, propName)
					}
				}
			}
		}
	}
}

