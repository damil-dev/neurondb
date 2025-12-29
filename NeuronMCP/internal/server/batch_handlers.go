/*-------------------------------------------------------------------------
 *
 * batch_handlers.go
 *    Batch handler setup for NeuronMCP
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/server/batch_handlers.go
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

/* setupBatchHandlers sets up batch MCP handlers */
func (s *Server) setupBatchHandlers() {
  /* Batch tool calls handler */
	s.mcpServer.SetHandler("tools/call_batch", s.handleCallBatch)
}

/* handleCallBatch handles the tools/call_batch request */
func (s *Server) handleCallBatch(ctx context.Context, params json.RawMessage) (interface{}, error) {
	mcpReq := &middleware.MCPRequest{
		Method: "tools/call_batch",
		Params: make(map[string]interface{}),
	}

	return s.middleware.Execute(ctx, mcpReq, func(ctx context.Context) (*middleware.MCPResponse, error) {
		result, err := s.batch.HandleCallBatch(ctx, params)
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





