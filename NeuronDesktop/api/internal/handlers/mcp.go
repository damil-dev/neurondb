package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/mcp"
	"github.com/neurondb/NeuronDesktop/api/internal/utils"
)

// MCPHandlers handles MCP-related endpoints
type MCPHandlers struct {
	mcpManager *MCPManager
}

// NewMCPHandlers creates new MCP handlers
func NewMCPHandlers(mcpManager *MCPManager) *MCPHandlers {
	return &MCPHandlers{mcpManager: mcpManager}
}

// ListTools lists tools from the MCP server
func (h *MCPHandlers) ListTools(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	
	client, err := h.mcpManager.GetClient(r.Context(), profileID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	tools, err := client.ListTools()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(tools)
}

// CallTool calls a tool on the MCP server
func (h *MCPHandlers) CallTool(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	
	var req struct {
		Name      string                 `json:"name"`
		Arguments map[string]interface{} `json:"arguments"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}
	
	// Validate tool call
	if errors := utils.ValidateToolCall(req.Name, req.Arguments); len(errors) > 0 {
		WriteValidationErrors(w, errors)
		return
	}
	
	client, err := h.mcpManager.GetClient(r.Context(), profileID)
	if err != nil {
		WriteError(w, http.StatusInternalServerError, err, nil)
		return
	}
	
	result, err := client.CallTool(req.Name, req.Arguments)
	if err != nil {
		WriteError(w, http.StatusInternalServerError, err, nil)
		return
	}
	
	WriteSuccess(w, result, http.StatusOK)
}

// ListConnections lists active MCP connections
func (h *MCPHandlers) ListConnections(w http.ResponseWriter, r *http.Request) {
	connections := h.mcpManager.ListConnections()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(connections)
}

// TestMCPConfig tests an MCP configuration without saving it
func (h *MCPHandlers) TestMCPConfig(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Command string            `json:"command"`
		Args    []string          `json:"args,omitempty"`
		Env     map[string]string `json:"env,omitempty"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}
	
	// Create temporary MCP config
	mcpConfig := mcp.MCPConfig{
		Command: req.Command,
		Args:    req.Args,
		Env:     req.Env,
	}
	
	if mcpConfig.Command == "" {
		mcpConfig.Command = "neurondb-mcp"
	}
	if mcpConfig.Args == nil {
		mcpConfig.Args = []string{}
	}
	if mcpConfig.Env == nil {
		mcpConfig.Env = make(map[string]string)
	}
	
	// Try to create and initialize client
	testClient, err := mcp.NewClient(mcpConfig)
	if err != nil {
		WriteError(w, http.StatusBadRequest, fmt.Errorf("failed to create MCP client: %w", err), nil)
		return
	}
	defer testClient.Close()
	
	// Try to list tools as a test
	_, err = testClient.ListTools()
	if err != nil {
		WriteError(w, http.StatusBadRequest, fmt.Errorf("failed to list tools: %w", err), nil)
		return
	}
	
	WriteSuccess(w, map[string]interface{}{
		"success": true,
		"message": "MCP configuration test passed",
	}, http.StatusOK)
}

// MCPManager manages MCP client connections
type MCPManager struct {
	clients map[string]*mcp.Client
	mu      sync.RWMutex
	queries *db.Queries
}

// NewMCPManager creates a new MCP manager
func NewMCPManager(queries *db.Queries) *MCPManager {
	return &MCPManager{
		clients: make(map[string]*mcp.Client),
		queries: queries,
	}
}

// GetMCPManager returns the MCP manager (for use in websocket handler)
func (h *MCPHandlers) GetMCPManager() *MCPManager {
	return h.mcpManager
}

// GetClient gets or creates an MCP client for a profile
func (m *MCPManager) GetClient(ctx context.Context, profileID string) (*mcp.Client, error) {
	m.mu.RLock()
	client, ok := m.clients[profileID]
	m.mu.RUnlock()
	
	if ok && client.IsAlive() {
		return client, nil
	}
	
	// Get profile
	profile, err := m.queries.GetProfile(ctx, profileID)
	if err != nil {
		return nil, fmt.Errorf("failed to get profile: %w", err)
	}
	
	if profile == nil {
		return nil, fmt.Errorf("profile not found: %s", profileID)
	}
	
	// Parse MCP config
	mcpConfig := mcp.MCPConfig{
		Command: "neurondb-mcp", // Default
		Args:    []string{},
		Env:     make(map[string]string),
	}
	
	if profile.MCPConfig != nil {
		if cmd, ok := profile.MCPConfig["command"].(string); ok {
			mcpConfig.Command = cmd
		}
		if args, ok := profile.MCPConfig["args"].([]interface{}); ok {
			for _, arg := range args {
				if s, ok := arg.(string); ok {
					mcpConfig.Args = append(mcpConfig.Args, s)
				}
			}
		}
		if env, ok := profile.MCPConfig["env"].(map[string]interface{}); ok {
			for k, v := range env {
				if s, ok := v.(string); ok {
					mcpConfig.Env[k] = s
				}
			}
		}
	}
	
	// Create new client
	newClient, err := mcp.NewClient(mcpConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create MCP client: %w", err)
	}
	client = newClient
	
	m.mu.Lock()
	m.clients[profileID] = client
	m.mu.Unlock()
	
	return client, nil
}

// ListConnections lists all active connections
func (m *MCPManager) ListConnections() []map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	var connections []map[string]interface{}
	for profileID, client := range m.clients {
		connections = append(connections, map[string]interface{}{
			"profile_id": profileID,
			"alive":      client.IsAlive(),
		})
	}
	
	return connections
}

