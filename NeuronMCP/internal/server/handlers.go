/*-------------------------------------------------------------------------
 *
 * handlers.go
 *    Database operations
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/server/handlers.go
 *
 *-------------------------------------------------------------------------
 */

package server

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/middleware"
	"github.com/neurondb/NeuronMCP/internal/tools"
	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* min returns the minimum of two integers */
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

/* setupToolHandlers sets up tool-related MCP handlers */
func (s *Server) setupToolHandlers() {
  /* List tools handler */
	s.mcpServer.SetHandler("tools/list", s.handleListTools)

  /* Call tool handler */
	s.mcpServer.SetHandler("tools/call", s.handleCallTool)

  /* Search tools handler */
	s.mcpServer.SetHandler("tools/search", s.handleSearchTools)
}

/* handleListTools handles the tools/list request */
func (s *Server) handleListTools(ctx context.Context, params json.RawMessage) (interface{}, error) {
	definitions := s.toolRegistry.GetAllDefinitions()
	filtered := s.filterToolsByFeatures(definitions)
	
	mcpTools := make([]mcp.ToolDefinition, len(filtered))
	for i, def := range filtered {
		mcpTools[i] = mcp.ToolDefinition{
			Name:         def.Name,
			Description:  def.Description,
			InputSchema:  def.InputSchema,
			OutputSchema: def.OutputSchema,
			Version:      def.Version,
			Deprecated:   def.Deprecated,
			Deprecation:  def.Deprecation,
		}
	}
	
	return mcp.ListToolsResponse{Tools: mcpTools}, nil
}

/* handleCallTool handles the tools/call request */
func (s *Server) handleCallTool(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var req mcp.CallToolRequest
	if err := json.Unmarshal(params, &req); err != nil {
		return nil, fmt.Errorf("failed to parse tools/call request: params_length=%d, params_preview='%s', error=%w (invalid JSON format or missing required fields)", len(params), string(params[:min(100, len(params))]), err)
	}

	if req.Name == "" {
		return nil, fmt.Errorf("tool name is required in tools/call request: received empty name, params=%v", req)
	}

	mcpReq := &middleware.MCPRequest{
		Method: "tools/call",
		Params: map[string]interface{}{
			"name":           req.Name,
			"arguments":      req.Arguments,
			"dryRun":         req.DryRun,
			"idempotencyKey": req.IdempotencyKey,
			"requireConfirm": req.RequireConfirm,
		},
	}

	return s.middleware.Execute(ctx, mcpReq, func(ctx context.Context, _ *middleware.MCPRequest) (*middleware.MCPResponse, error) {
		return s.executeTool(ctx, req.Name, req.Arguments, req.DryRun, req.IdempotencyKey, req.RequireConfirm)
	})
}

/* executeTool executes a tool and returns the response */
func (s *Server) executeTool(ctx context.Context, toolName string, arguments map[string]interface{}, dryRun bool, idempotencyKey string, requireConfirm bool) (*middleware.MCPResponse, error) {
	if toolName == "" {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: fmt.Sprintf("Tool name is required and cannot be empty: received empty string, arguments_count=%d", len(arguments))},
			},
			IsError: true,
		}, nil
	}

	tool := s.toolRegistry.GetTool(toolName)
	if tool == nil {
		availableTools := s.toolRegistry.GetAllDefinitions()
		toolNames := make([]string, 0, len(availableTools))
		for _, def := range availableTools {
			toolNames = append(toolNames, def.Name)
		}
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: fmt.Sprintf("Tool not found: tool_name='%s', arguments_count=%d, available_tools_count=%d, available_tools=%v", toolName, len(arguments), len(availableTools), toolNames)},
			},
			IsError: true,
		}, nil
	}

  /* Log tool execution start */
	s.logger.Info("Executing tool", map[string]interface{}{
		"tool_name":       toolName,
		"arguments_count": len(arguments),
		"dry_run":         dryRun,
		"idempotency_key": idempotencyKey,
		"require_confirm": requireConfirm,
	})

	/* Handle dry run mode */
	if dryRun {
		dryRunExecutor := tools.NewDryRunExecutor(tool)
		dryRunResult, err := dryRunExecutor.Execute(ctx, arguments)
		if err != nil {
			return s.formatToolError(dryRunResult), nil
		}
		resultJSON, _ := json.MarshalIndent(dryRunResult.Data, "", "  ")
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: string(resultJSON)},
			},
			Metadata: map[string]interface{}{
				"dryRun": true,
				"tool":   toolName,
			},
		}, nil
	}

	/* Check if confirmation is required */
	if requireConfirm && tools.RequiresConfirmation(toolName) {
		/* Check if confirmation is provided in arguments */
		if confirmed, ok := arguments["confirmed"].(bool); !ok || !confirmed {
			return &middleware.MCPResponse{
				Content: []middleware.ContentBlock{
					{Type: "text", Text: fmt.Sprintf("Confirmation required for tool '%s' - set 'confirmed' parameter to true", toolName)},
				},
				IsError: true,
				Metadata: map[string]interface{}{
					"error_code": "CONFIRMATION_REQUIRED",
					"tool":       toolName,
				},
			}, nil
		}
	}

	/* Handle idempotency - for now, log it (full implementation would check/store in database) */
	if idempotencyKey != "" {
		s.logger.Debug(fmt.Sprintf("Tool execution with idempotency key: %s", idempotencyKey), nil)
		/* TODO: Check if this idempotency key was already used and return cached result */
	}

	result, err := tool.Execute(ctx, arguments)
	if err != nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: fmt.Sprintf("Tool execution error: tool_name='%s', arguments_count=%d, arguments=%v, error=%v", toolName, len(arguments), arguments, err)},
			},
			IsError: true,
		}, nil
	}

	return s.formatToolResult(result)
}

/* formatToolResult formats a tool result as an MCP response */
func (s *Server) formatToolResult(result *tools.ToolResult) (*middleware.MCPResponse, error) {
	if !result.Success {
		return s.formatToolError(result), nil
	}

	/* Validate output against schema if tool has output schema */
	/* Note: This is a placeholder - full validation would check the tool's output schema */
	
	resultJSON, _ := json.MarshalIndent(result.Data, "", "  ")
	return &middleware.MCPResponse{
		Content: []middleware.ContentBlock{
			{Type: "text", Text: string(resultJSON)},
		},
		Metadata: result.Metadata,
	}, nil
}

/* formatToolError formats a tool error as an MCP response */
func (s *Server) formatToolError(result *tools.ToolResult) *middleware.MCPResponse {
	errorText := "Unknown error"
	errorMetadata := make(map[string]interface{})
	
	if result.Error != nil {
		errorText = result.Error.Message
		errorMetadata["message"] = result.Error.Message
		if result.Error.Code != "" {
			errorMetadata["code"] = result.Error.Code
		}
		if result.Error.Details != nil {
			errorMetadata["details"] = result.Error.Details
		}
	}
	
	return &middleware.MCPResponse{
		Content: []middleware.ContentBlock{
			{Type: "text", Text: fmt.Sprintf("Error: %s", errorText)},
		},
		IsError: true,
		Metadata: errorMetadata,
	}
}

/* handleSearchTools handles the tools/search request */
func (s *Server) handleSearchTools(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var req struct {
		Query    string `json:"query,omitempty"`
		Category string `json:"category,omitempty"`
	}
	if err := json.Unmarshal(params, &req); err != nil {
		return nil, fmt.Errorf("failed to parse tools/search request: %w", err)
	}

	definitions := s.toolRegistry.Search(req.Query, req.Category)
	
	mcpTools := make([]mcp.ToolDefinition, len(definitions))
	for i, def := range definitions {
		mcpTools[i] = mcp.ToolDefinition{
			Name:         def.Name,
			Description:  def.Description,
			InputSchema:  def.InputSchema,
			OutputSchema: def.OutputSchema,
			Version:      def.Version,
			Deprecated:   def.Deprecated,
			Deprecation:  def.Deprecation,
		}
	}
	
	return mcp.ListToolsResponse{Tools: mcpTools}, nil
}

