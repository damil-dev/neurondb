/*-------------------------------------------------------------------------
 *
 * sampling_handlers.go
 *    Sampling handler setup for NeuronMCP
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/server/sampling_handlers.go
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

/* setupSamplingHandlers sets up sampling-related MCP handlers */
func (s *Server) setupSamplingHandlers() {
  /* Create message handler */
	s.mcpServer.SetHandler("sampling/createMessage", s.handleCreateMessage)
}

/* handleCreateMessage handles the sampling/createMessage request */
func (s *Server) handleCreateMessage(ctx context.Context, params json.RawMessage) (interface{}, error) {
	mcpReq := &middleware.MCPRequest{
		Method: "sampling/createMessage",
		Params: make(map[string]interface{}),
	}

	return s.middleware.Execute(ctx, mcpReq, func(ctx context.Context) (*middleware.MCPResponse, error) {
		result, err := s.sampling.HandleCreateMessage(ctx, params)
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




