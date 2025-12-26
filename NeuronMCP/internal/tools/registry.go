/*-------------------------------------------------------------------------
 *
 * registry.go
 *    Tool registry for NeuronMCP
 *
 * Manages tool registration, definitions, and execution for the MCP server.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/registry.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"fmt"
	"strings"
	"sync"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* ToolDefinition represents a tool's definition for MCP */
type ToolDefinition struct {
	Name         string                 `json:"name"`
	Description  string                 `json:"description"`
	InputSchema  map[string]interface{} `json:"inputSchema"`
	OutputSchema map[string]interface{} `json:"outputSchema,omitempty"`
	Version      string                 `json:"version,omitempty"`
	Deprecated   bool                   `json:"deprecated,omitempty"`
	Deprecation  *mcp.DeprecationInfo   `json:"deprecation,omitempty"`
}

/* ToolRegistry manages tool registration and execution */
type ToolRegistry struct {
	tools      map[string]Tool
	definitions map[string]ToolDefinition
	mu         sync.RWMutex
	db         *database.Database
	logger     *logging.Logger
}

/* NewToolRegistry creates a new tool registry */
func NewToolRegistry(db *database.Database, logger *logging.Logger) *ToolRegistry {
	return &ToolRegistry{
		tools:       make(map[string]Tool),
		definitions: make(map[string]ToolDefinition),
		db:          db,
		logger:      logger,
	}
}

/* Register registers a tool */
func (r *ToolRegistry) Register(tool Tool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	definition := ToolDefinition{
		Name:         tool.Name(),
		Description:  tool.Description(),
		InputSchema:  tool.InputSchema(),
		OutputSchema: tool.OutputSchema(),
		Version:      tool.Version(),
		Deprecated:   tool.Deprecated(),
		Deprecation:  tool.Deprecation(),
	}

	r.tools[tool.Name()] = tool
	r.definitions[tool.Name()] = definition
	if r.logger != nil {
		r.logger.Debug(fmt.Sprintf("Registered tool: %s (version: %s)", tool.Name(), tool.Version()), nil)
	}
}

/* RegisterAll registers multiple tools */
func (r *ToolRegistry) RegisterAll(tools []Tool) {
	for _, tool := range tools {
		r.Register(tool)
	}
}

/* GetTool retrieves a tool by name */
func (r *ToolRegistry) GetTool(name string) Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.tools[name]
}

/* GetDefinition retrieves a tool definition by name */
func (r *ToolRegistry) GetDefinition(name string) (ToolDefinition, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	def, exists := r.definitions[name]
	return def, exists
}

/* GetAllDefinitions returns all tool definitions */
func (r *ToolRegistry) GetAllDefinitions() []ToolDefinition {
	r.mu.RLock()
	defer r.mu.RUnlock()

	definitions := make([]ToolDefinition, 0, len(r.definitions))
	for _, def := range r.definitions {
		definitions = append(definitions, def)
	}
	return definitions
}

/* GetAllToolNames returns all registered tool names */
func (r *ToolRegistry) GetAllToolNames() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.tools))
	for name := range r.tools {
		names = append(names, name)
	}
	return names
}

/* HasTool checks if a tool exists */
func (r *ToolRegistry) HasTool(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, exists := r.tools[name]
	return exists
}

/* Unregister removes a tool */
func (r *ToolRegistry) Unregister(name string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	removed := false
	if _, exists := r.tools[name]; exists {
		delete(r.tools, name)
		delete(r.definitions, name)
		removed = true
		if r.logger != nil {
			r.logger.Debug(fmt.Sprintf("Unregistered tool: %s", name), nil)
		}
	}
	return removed
}

/* Clear removes all tools */
func (r *ToolRegistry) Clear() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tools = make(map[string]Tool)
	r.definitions = make(map[string]ToolDefinition)
}

/* GetCount returns the number of registered tools */
func (r *ToolRegistry) GetCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.tools)
}

/* Search searches for tools by name or description */
func (r *ToolRegistry) Search(query string, category string) []ToolDefinition {
	r.mu.RLock()
	defer r.mu.RUnlock()

	results := make([]ToolDefinition, 0)
	queryLower := strings.ToLower(strings.TrimSpace(query))
	categoryLower := strings.ToLower(strings.TrimSpace(category))

	for _, def := range r.definitions {
		/* Search in name */
		nameMatch := query == "" || containsIgnoreCase(def.Name, query) || containsIgnoreCase(def.Name, queryLower)

		/* Search in description */
		descMatch := query == "" || containsIgnoreCase(def.Description, query) || containsIgnoreCase(def.Description, queryLower)

		/* Category filter */
		categoryMatch := true
		if category != "" {
			/* Extract category from tool name prefix */
			categoryMatch = false
			toolNameLower := strings.ToLower(def.Name)
			if strings.HasPrefix(toolNameLower, categoryLower+"_") {
				categoryMatch = true
			}
			/* Also check if category matches common prefixes */
			categories := []string{"vector", "ml", "rag", "analytics", "indexing", "embedding", "hybrid", "rerank"}
			for _, cat := range categories {
				if categoryLower == cat && strings.HasPrefix(toolNameLower, cat+"_") {
					categoryMatch = true
					break
				}
			}
		}

		if (nameMatch || descMatch) && categoryMatch {
			results = append(results, def)
		}
	}

	return results
}

/* containsIgnoreCase checks if a string contains another (case-insensitive) */
func containsIgnoreCase(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || 
		strings.Contains(strings.ToLower(s), strings.ToLower(substr)))
}

