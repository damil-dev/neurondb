/*-------------------------------------------------------------------------
 *
 * registry.go
 *    Tool implementation for NeuronMCP
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/registry.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/auth"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

/* Registry manages tool registration and execution */
type Registry struct {
	queries     *db.Queries
	db          *db.DB
	handlers    map[string]ToolHandler
	auditLogger *auth.AuditLogger
	mu          sync.RWMutex
}

/* NewRegistry creates a new tool registry */
func NewRegistry(queries *db.Queries, database *db.DB) *Registry {
	registry := &Registry{
		queries:     queries,
		db:          database,
		handlers:    make(map[string]ToolHandler),
		auditLogger: auth.NewAuditLogger(queries),
	}

  /* Register built-in handlers */
	sqlTool := NewSQLTool(queries)
	sqlTool.db = database
	registry.RegisterHandler("sql", sqlTool)
	registry.RegisterHandler("http", NewHTTPTool())
	registry.RegisterHandler("code", NewCodeTool())
	registry.RegisterHandler("shell", NewShellTool())

	return registry
}

/* NewRegistryWithNeuronDB creates a new tool registry with NeuronDB clients */
func NewRegistryWithNeuronDB(queries *db.Queries, database *db.DB, neurondbClient interface{}) *Registry {
	registry := NewRegistry(queries, database)

	/* Register NeuronDB tool handlers if client is provided */
	if client, ok := neurondbClient.(*neurondb.Client); ok {
		if client.ML != nil {
			registry.RegisterHandler("ml", NewMLTool(client.ML))
		}
		if client.Vector != nil {
			registry.RegisterHandler("vector", NewVectorTool(client.Vector))
		}
		if client.RAG != nil {
			registry.RegisterHandler("rag", NewRAGTool(client.RAG))
		}
		if client.Analytics != nil {
			registry.RegisterHandler("analytics", NewAnalyticsTool(client.Analytics))
		}
		if client.HybridSearch != nil {
			registry.RegisterHandler("hybrid_search", NewHybridSearchTool(client.HybridSearch))
		}
		if client.Reranking != nil {
			registry.RegisterHandler("reranking", NewRerankingTool(client.Reranking))
		}
	}

	return registry
}

/* RegisterHandler registers a tool handler for a specific handler type */
func (r *Registry) RegisterHandler(handlerType string, handler ToolHandler) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.handlers[handlerType] = handler
}

/* Get retrieves a tool from the database */
/* Implements agent.ToolRegistry interface */
func (r *Registry) Get(ctx context.Context, name string) (*db.Tool, error) {
	tool, err := r.queries.GetTool(ctx, name)
	if err != nil {
		return nil, fmt.Errorf("tool retrieval failed: tool_name='%s', error=%w", name, err)
	}
	return tool, nil
}

/* Execute executes a tool with the given arguments */
/* Implements agent.ToolRegistry interface */
func (r *Registry) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	return r.ExecuteTool(ctx, tool, args)
}

/* ExecuteTool executes a tool with the given arguments (internal method) */
func (r *Registry) ExecuteTool(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	if !tool.Enabled {
		argKeys := make([]string, 0, len(args))
		for k := range args {
			argKeys = append(argKeys, k)
		}
		return "", fmt.Errorf("tool execution failed: tool_name='%s', handler_type='%s', enabled=false, args_count=%d, arg_keys=[%v]",
			tool.Name, tool.HandlerType, len(args), argKeys)
	}

  /* Validate arguments */
	if err := ValidateArgs(args, tool.ArgSchema); err != nil {
		argKeys := make([]string, 0, len(args))
		for k := range args {
			argKeys = append(argKeys, k)
		}
		return "", fmt.Errorf("tool validation failed: tool_name='%s', handler_type='%s', args_count=%d, arg_keys=[%v], validation_error='%v'",
			tool.Name, tool.HandlerType, len(args), argKeys, err)
	}

  /* Get handler */
	r.mu.RLock()
	handler, exists := r.handlers[tool.HandlerType]
	r.mu.RUnlock()

	if !exists {
		argKeys := make([]string, 0, len(args))
		for k := range args {
			argKeys = append(argKeys, k)
		}
		availableHandlers := make([]string, 0, len(r.handlers))
		for k := range r.handlers {
			availableHandlers = append(availableHandlers, k)
		}
		return "", fmt.Errorf("tool execution failed: tool_name='%s', handler_type='%s', handler_not_found=true, args_count=%d, arg_keys=[%v], available_handlers=[%v]",
			tool.Name, tool.HandlerType, len(args), argKeys, availableHandlers)
	}

  /* Execute tool */
	result, err := handler.Execute(ctx, tool, args)
	
  /* Audit log tool execution */
	agentID, _ := agent.GetAgentIDFromContext(ctx)
	sessionID, _ := agent.GetSessionIDFromContext(ctx)
	
	outputs := make(map[string]interface{})
	if err == nil {
		outputs["result"] = result
		outputs["success"] = true
	} else {
		outputs["error"] = err.Error()
		outputs["success"] = false
	}
	
	/* Log audit trail (async, don't block on errors) */
	go func() {
		bgCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		
		var agentIDPtr, sessionIDPtr *uuid.UUID
		if agentID != uuid.Nil {
			agentIDPtr = &agentID
		}
		if sessionID != uuid.Nil {
			sessionIDPtr = &sessionID
		}
		
		_ = r.auditLogger.LogToolCall(bgCtx, nil, nil, agentIDPtr, sessionIDPtr, tool.Name, args, outputs)
	}()
	
	if err != nil {
		argKeys := make([]string, 0, len(args))
		for k := range args {
			argKeys = append(argKeys, k)
		}
		return "", fmt.Errorf("tool execution failed: tool_name='%s', handler_type='%s', args_count=%d, arg_keys=[%v], error=%w",
			tool.Name, tool.HandlerType, len(args), argKeys, err)
	}
	return result, nil
}

/* ListTools returns all enabled tools */
func (r *Registry) ListTools(ctx context.Context) ([]db.Tool, error) {
	return r.queries.ListTools(ctx)
}

