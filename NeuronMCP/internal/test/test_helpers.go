/*-------------------------------------------------------------------------
 *
 * test_helpers.go
 *    Test helper utilities
 *
 * Provides common utilities for testing NeuronMCP functionality.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/test/test_helpers.go
 *
 *-------------------------------------------------------------------------
 */

package test

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/server"
	"github.com/neurondb/NeuronMCP/internal/tools"
	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* TestServer wraps a server for testing */
type TestServer struct {
	Server       *server.Server
	ToolRegistry *tools.ToolRegistry
	DB           *database.Database
}

/* NewTestServer creates a new test server */
func NewTestServer() (*TestServer, error) {
	s, err := server.NewServer()
	if err != nil {
		return nil, fmt.Errorf("failed to create test server: %w", err)
	}

	return &TestServer{
		Server: s,
	}, nil
}

/* CallTool calls a tool via the server */
func (ts *TestServer) CallTool(ctx context.Context, toolName string, arguments map[string]interface{}) (*mcp.ToolResult, error) {
	/* This would need to be implemented based on your server's handler interface */
	/* For now, this is a placeholder */
	return nil, fmt.Errorf("CallTool not yet implemented in test helpers")
}

/* ListTools lists all available tools */
func (ts *TestServer) ListTools(ctx context.Context) ([]mcp.ToolDefinition, error) {
	/* This would need to be implemented based on your server's handler interface */
	return nil, fmt.Errorf("ListTools not yet implemented in test helpers")
}

/* ValidateToolOutput validates tool output against its schema */
func ValidateToolOutput(tool tools.Tool, output interface{}) error {
	schema := tool.OutputSchema()
	if schema == nil {
		return nil // No schema to validate against
	}

	valid, errors := tools.ValidateOutput(output, schema)
	if !valid {
		return fmt.Errorf("output validation failed: %v", errors)
	}

	return nil
}

/* AssertToolVersion asserts that a tool has the expected version */
func AssertToolVersion(tool tools.Tool, expectedVersion string) error {
	actualVersion := tool.Version()
	if actualVersion != expectedVersion {
		return fmt.Errorf("tool version mismatch: expected %s, got %s", expectedVersion, actualVersion)
	}
	return nil
}

