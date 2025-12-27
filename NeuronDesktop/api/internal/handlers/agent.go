package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/agent"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
)

// AgentHandlers handles NeuronAgent proxy endpoints
type AgentHandlers struct {
	queries *db.Queries
	clients map[string]*agent.Client
	mu      sync.RWMutex
}

// GetQueries returns the queries instance (for use in websocket handler)
func (h *AgentHandlers) GetQueries() *db.Queries {
	return h.queries
}

// NewAgentHandlers creates new agent handlers
func NewAgentHandlers(queries *db.Queries) *AgentHandlers {
	return &AgentHandlers{
		queries: queries,
		clients: make(map[string]*agent.Client),
	}
}

// getClient gets or creates an agent client for a profile
func (h *AgentHandlers) getClient(ctx context.Context, profileID string) (*agent.Client, error) {
	h.mu.RLock()
	client, ok := h.clients[profileID]
	h.mu.RUnlock()
	
	if ok {
		return client, nil
	}
	
	// Get profile
	profile, err := h.queries.GetProfile(ctx, profileID)
	if err != nil {
		return nil, fmt.Errorf("failed to get profile: %w", err)
	}
	
	if profile.AgentEndpoint == "" {
		return nil, fmt.Errorf("agent endpoint not configured")
	}
	
	// Create client
	client = agent.NewClient(profile.AgentEndpoint, profile.AgentAPIKey)
	
	h.mu.Lock()
	h.clients[profileID] = client
	h.mu.Unlock()
	
	return client, nil
}

// ListAgents lists agents
func (h *AgentHandlers) ListAgents(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	
	client, err := h.getClient(r.Context(), profileID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	agents, err := client.ListAgents(r.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(agents)
}

// CreateSession creates a session
func (h *AgentHandlers) CreateSession(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	
	var req agent.CreateSessionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}
	
	client, err := h.getClient(r.Context(), profileID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	session, err := client.CreateSession(r.Context(), req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(session)
}

// SendMessage sends a message
func (h *AgentHandlers) SendMessage(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	sessionID := vars["session_id"]
	
	var req agent.SendMessageRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}
	
	client, err := h.getClient(r.Context(), profileID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	message, err := client.SendMessage(r.Context(), sessionID, req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(message)
}

// GetAgent gets a single agent
func (h *AgentHandlers) GetAgent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	agentID := vars["agent_id"]
	
	client, err := h.getClient(r.Context(), profileID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	agent, err := client.GetAgent(r.Context(), agentID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(agent)
}

// CreateAgent creates a new agent
func (h *AgentHandlers) CreateAgent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	
	var req agent.CreateAgentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}
	
	client, err := h.getClient(r.Context(), profileID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	agent, err := client.CreateAgent(r.Context(), req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(agent)
}

// TestAgentConfig tests an Agent configuration without saving it
func (h *AgentHandlers) TestAgentConfig(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Endpoint string `json:"endpoint"`
		APIKey   string `json:"api_key"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}
	
	if req.Endpoint == "" {
		WriteError(w, http.StatusBadRequest, fmt.Errorf("endpoint is required"), nil)
		return
	}
	
	// Create temporary client
	testClient := agent.NewClient(req.Endpoint, req.APIKey)
	
	// Try to list agents as a test
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	
	_, err := testClient.ListAgents(ctx)
	if err != nil {
		WriteError(w, http.StatusBadRequest, fmt.Errorf("failed to connect to agent: %w", err), nil)
		return
	}
	
	WriteSuccess(w, map[string]interface{}{
		"success": true,
		"message": "Agent configuration test passed",
	}, http.StatusOK)
}

// ListModels returns available LLM models
func (h *AgentHandlers) ListModels(w http.ResponseWriter, r *http.Request) {
	// Common LLM models supported by NeuronDB
	models := []map[string]interface{}{
		{
			"name":        "gpt-4",
			"display_name": "GPT-4",
			"provider":    "OpenAI",
			"description": "Most capable GPT-4 model",
		},
		{
			"name":        "gpt-4-turbo",
			"display_name": "GPT-4 Turbo",
			"provider":    "OpenAI",
			"description": "Faster and cheaper GPT-4 variant",
		},
		{
			"name":        "gpt-3.5-turbo",
			"display_name": "GPT-3.5 Turbo",
			"provider":    "OpenAI",
			"description": "Fast and cost-effective model",
		},
		{
			"name":        "claude-3-opus",
			"display_name": "Claude 3 Opus",
			"provider":    "Anthropic",
			"description": "Most capable Claude model",
		},
		{
			"name":        "claude-3-sonnet",
			"display_name": "Claude 3 Sonnet",
			"provider":    "Anthropic",
			"description": "Balanced performance and speed",
		},
		{
			"name":        "claude-3-haiku",
			"display_name": "Claude 3 Haiku",
			"provider":    "Anthropic",
			"description": "Fastest Claude model",
		},
		{
			"name":        "gemini-pro",
			"display_name": "Gemini Pro",
			"provider":    "Google",
			"description": "Google's advanced model",
		},
		{
			"name":        "llama-2-70b",
			"display_name": "Llama 2 70B",
			"provider":    "Meta",
			"description": "Open-source large model",
		},
		{
			"name":        "custom",
			"display_name": "Custom Model",
			"provider":    "Custom",
			"description": "Enter a custom model name",
		},
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"models": models,
	})
}

