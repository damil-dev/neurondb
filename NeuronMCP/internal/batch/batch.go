/*-------------------------------------------------------------------------
 *
 * batch.go
 *    Batch operations for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/batch/batch.go
 *
 *-------------------------------------------------------------------------
 */

package batch

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/tools"
)

/* BatchRequest represents a batch tool call request */
type BatchRequest struct {
	Tools      []ToolCall            `json:"tools"`
	Transaction bool                 `json:"transaction,omitempty"`
}

/* ToolCall represents a single tool call in a batch */
type ToolCall struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments,omitempty"`
}

/* BatchResult represents a batch operation result */
type BatchResult struct {
	Results []ToolResult            `json:"results"`
	Success bool                    `json:"success"`
	Error   string                  `json:"error,omitempty"`
}

/* ToolResult represents a single tool result */
type ToolResult struct {
	Tool    string                 `json:"tool"`
	Success bool                   `json:"success"`
	Data    interface{}            `json:"data,omitempty"`
	Error   string                 `json:"error,omitempty"`
}

/* Processor processes batch operations */
type Processor struct {
	db           *database.Database
	toolRegistry *tools.ToolRegistry
	logger       *logging.Logger
}

/* NewProcessor creates a new batch processor */
func NewProcessor(db *database.Database, toolRegistry *tools.ToolRegistry, logger *logging.Logger) *Processor {
	return &Processor{
		db:           db,
		toolRegistry: toolRegistry,
		logger:       logger,
	}
}

/* ProcessBatch processes a batch of tool calls */
func (p *Processor) ProcessBatch(ctx context.Context, req BatchRequest) (*BatchResult, error) {
	if len(req.Tools) == 0 {
		return &BatchResult{
			Results: []ToolResult{},
			Success: true,
		}, nil
	}

	if len(req.Tools) > 100 {
		return nil, fmt.Errorf("batch size exceeds maximum of 100 tools: received %d tools", len(req.Tools))
	}

	results := make([]ToolResult, 0, len(req.Tools))

	if req.Transaction {
		/* Start transaction */
		tx, err := p.db.Begin(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to start transaction: %w", err)
		}
		
		/* Track if we need to rollback */
		shouldRollback := false
		defer func() {
			if shouldRollback {
				if rollbackErr := tx.Rollback(ctx); rollbackErr != nil {
					if p.logger != nil {
						p.logger.Warn("Failed to rollback transaction", map[string]interface{}{
							"error": rollbackErr.Error(),
						})
					}
				}
			}
		}()

		/* Execute all tools in transaction */
		for i, toolCall := range req.Tools {
			if toolCall.Name == "" {
				shouldRollback = true
				return &BatchResult{
					Results: results,
					Success: false,
					Error:   fmt.Sprintf("Tool at index %d has empty name, transaction rolled back", i),
				}, nil
			}

			result := p.executeTool(ctx, toolCall)
			results = append(results, result)
			
			/* If any tool fails and transaction is enabled, stop and rollback */
			if !result.Success {
				shouldRollback = true
				return &BatchResult{
					Results: results,
					Success: false,
					Error:   fmt.Sprintf("Tool %s failed at index %d, transaction rolled back", toolCall.Name, i),
				}, nil
			}
		}

		/* Commit transaction */
		if err := tx.Commit(ctx); err != nil {
			return nil, fmt.Errorf("failed to commit transaction: %w", err)
		}
	} else {
		/* Execute tools without transaction */
		for i, toolCall := range req.Tools {
			if toolCall.Name == "" {
				results = append(results, ToolResult{
					Tool:    fmt.Sprintf("tool_%d", i),
					Success: false,
					Error:   fmt.Sprintf("Tool at index %d has empty name", i),
				})
				continue
			}

			result := p.executeTool(ctx, toolCall)
			results = append(results, result)
		}
	}

	/* Check if all succeeded */
	allSuccess := true
	for _, result := range results {
		if !result.Success {
			allSuccess = false
			break
		}
	}

	return &BatchResult{
		Results: results,
		Success: allSuccess,
	}, nil
}

/* executeTool executes a single tool */
func (p *Processor) executeTool(ctx context.Context, toolCall ToolCall) ToolResult {
	tool := p.toolRegistry.GetTool(toolCall.Name)
	if tool == nil {
		return ToolResult{
			Tool:    toolCall.Name,
			Success: false,
			Error:   fmt.Sprintf("Tool not found: %s", toolCall.Name),
		}
	}

	result, err := tool.Execute(ctx, toolCall.Arguments)
	if err != nil {
		return ToolResult{
			Tool:    toolCall.Name,
			Success: false,
			Error:   err.Error(),
		}
	}

	if !result.Success {
		errorMsg := "Unknown error"
		if result.Error != nil {
			errorMsg = result.Error.Message
		}
		return ToolResult{
			Tool:    toolCall.Name,
			Success: false,
			Error:   errorMsg,
		}
	}

	return ToolResult{
		Tool:    toolCall.Name,
		Success: true,
		Data:    result.Data,
	}
}

