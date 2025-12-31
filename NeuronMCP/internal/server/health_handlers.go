/*-------------------------------------------------------------------------
 *
 * health_handlers.go
 *    Health check handler setup for NeuronMCP
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/server/health_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package server

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/middleware"
)

/* setupHealthHandlers sets up health check MCP handlers */
func (s *Server) setupHealthHandlers() {
  /* Health check handler */
	s.mcpServer.SetHandler("health/check", s.handleHealthCheck)
}

/* handleHealthCheck handles the health/check request */
func (s *Server) handleHealthCheck(ctx context.Context, params json.RawMessage) (interface{}, error) {
	mcpReq := &middleware.MCPRequest{
		Method: "health/check",
		Params: make(map[string]interface{}),
	}

	return s.middleware.Execute(ctx, mcpReq, func(ctx context.Context, _ *middleware.MCPRequest) (*middleware.MCPResponse, error) {
		result, err := s.health.HandleHealthCheck(ctx, params)
		if err != nil {
			return &middleware.MCPResponse{
				Content: []middleware.ContentBlock{
					{Type: "text", Text: fmt.Sprintf("Error: %v", err)},
				},
				IsError: true,
			}, nil
		}

		resultJSON, _ := json.MarshalIndent(result, "", "  ")
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: string(resultJSON)},
			},
		}, nil
	})
}










