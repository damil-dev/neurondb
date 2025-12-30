package testing

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/neurondb/NeuronDesktop/api/internal/agent"
	"github.com/neurondb/NeuronDesktop/api/internal/mcp"
)

// MockMCPClient is a mock MCP client for testing
type MockMCPClient struct {
	Tools      []mcp.ToolDefinition
	CallResult *mcp.ToolResult
	CallError  error
	Connected  bool
}

// NewMockMCPClient creates a new mock MCP client
func NewMockMCPClient() *MockMCPClient {
	return &MockMCPClient{
		Tools: []mcp.ToolDefinition{
			{
				Name:        "vector_search",
				Description: "Perform vector search",
				InputSchema: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"query_vector": map[string]interface{}{
							"type":  "array",
							"items": map[string]interface{}{"type": "number"},
						},
						"table": map[string]interface{}{
							"type": "string",
						},
						"limit": map[string]interface{}{
							"type":    "integer",
							"default": 10,
						},
					},
				},
			},
		},
		Connected: true,
	}
}

// ListTools returns mock tools
func (m *MockMCPClient) ListTools(ctx context.Context) ([]mcp.ToolDefinition, error) {
	if !m.Connected {
		return nil, fmt.Errorf("MCP server not connected")
	}
	return m.Tools, nil
}

// CallTool calls a mock tool
func (m *MockMCPClient) CallTool(ctx context.Context, name string, arguments map[string]interface{}) (*mcp.ToolResult, error) {
	if !m.Connected {
		return nil, fmt.Errorf("MCP server not connected")
	}
	if m.CallError != nil {
		return nil, m.CallError
	}
	if m.CallResult != nil {
		return m.CallResult, nil
	}
	// Default mock response
	return &mcp.ToolResult{
		Content: []mcp.ContentBlock{
			{
				Type: "text",
				Text: fmt.Sprintf("Mock result for tool %s", name),
			},
		},
		IsError: false,
	}, nil
}

// Close closes the mock client
func (m *MockMCPClient) Close() error {
	m.Connected = false
	return nil
}

// IsAlive checks if mock client is alive
func (m *MockMCPClient) IsAlive() bool {
	return m.Connected
}

// Model represents a model in NeuronAgent (simplified for testing)
type Model struct {
	Name     string `json:"name"`
	Type     string `json:"type"`
	Provider string `json:"provider"`
}

// MockAgentClient is a mock Agent client for testing
type MockAgentClient struct {
	Agents      []agent.Agent
	Sessions    []agent.Session
	Messages    []agent.Message
	Models      []Model
	CreateError error
	CallError   error
}

// NewMockAgentClient creates a new mock Agent client
func NewMockAgentClient() *MockAgentClient {
	return &MockAgentClient{
		Agents: []agent.Agent{
			{
				ID:          "agent-1",
				Name:        "Test Agent",
				Description: "A test agent",
				ModelName:   "gpt-4",
			},
		},
		Models: []Model{
			{
				Name:     "gpt-4",
				Type:     "chat",
				Provider: "openai",
			},
			{
				Name:     "gpt-3.5-turbo",
				Type:     "chat",
				Provider: "openai",
			},
		},
	}
}

// ListAgents returns mock agents
func (m *MockAgentClient) ListAgents(ctx context.Context) ([]agent.Agent, error) {
	if m.CallError != nil {
		return nil, m.CallError
	}
	return m.Agents, nil
}

// CreateAgent creates a mock agent
func (m *MockAgentClient) CreateAgent(ctx context.Context, req agent.CreateAgentRequest) (*agent.Agent, error) {
	if m.CreateError != nil {
		return nil, m.CreateError
	}
	newAgent := &agent.Agent{
		ID:          fmt.Sprintf("agent-%d", len(m.Agents)+1),
		Name:        req.Name,
		Description: req.Description,
		ModelName:   req.ModelName,
	}
	m.Agents = append(m.Agents, *newAgent)
	return newAgent, nil
}

// GetAgent returns a mock agent
func (m *MockAgentClient) GetAgent(ctx context.Context, agentID string) (*agent.Agent, error) {
	if m.CallError != nil {
		return nil, m.CallError
	}
	for _, a := range m.Agents {
		if a.ID == agentID {
			return &a, nil
		}
	}
	return nil, fmt.Errorf("agent not found")
}

// ListModels returns mock models
func (m *MockAgentClient) ListModels(ctx context.Context) ([]Model, error) {
	if m.CallError != nil {
		return nil, m.CallError
	}
	return m.Models, nil
}

// CreateSession creates a mock session
func (m *MockAgentClient) CreateSession(ctx context.Context, req agent.CreateSessionRequest) (*agent.Session, error) {
	if m.CreateError != nil {
		return nil, m.CreateError
	}
	newSession := &agent.Session{
		ID:      fmt.Sprintf("session-%d", len(m.Sessions)+1),
		AgentID: req.AgentID,
	}
	m.Sessions = append(m.Sessions, *newSession)
	return newSession, nil
}

// ListSessions returns mock sessions
func (m *MockAgentClient) ListSessions(ctx context.Context, agentID string) ([]agent.Session, error) {
	if m.CallError != nil {
		return nil, m.CallError
	}
	var filtered []agent.Session
	for _, s := range m.Sessions {
		if s.AgentID == agentID {
			filtered = append(filtered, s)
		}
	}
	return filtered, nil
}

// GetSession returns a mock session
func (m *MockAgentClient) GetSession(ctx context.Context, sessionID string) (*agent.Session, error) {
	if m.CallError != nil {
		return nil, m.CallError
	}
	for _, s := range m.Sessions {
		if s.ID == sessionID {
			return &s, nil
		}
	}
	return nil, fmt.Errorf("session not found")
}

// SendMessage sends a mock message
func (m *MockAgentClient) SendMessage(ctx context.Context, sessionID string, req agent.SendMessageRequest) (*agent.Message, error) {
	if m.CallError != nil {
		return nil, m.CallError
	}
	newMessage := &agent.Message{
		ID:        fmt.Sprintf("msg-%d", len(m.Messages)+1),
		SessionID: sessionID,
		Role:      req.Role,
		Content:   req.Content,
	}
	m.Messages = append(m.Messages, *newMessage)
	return newMessage, nil
}

// GetMessages returns mock messages
func (m *MockAgentClient) GetMessages(ctx context.Context, sessionID string) ([]agent.Message, error) {
	if m.CallError != nil {
		return nil, m.CallError
	}
	var filtered []agent.Message
	for _, msg := range m.Messages {
		if msg.SessionID == sessionID {
			filtered = append(filtered, msg)
		}
	}
	return filtered, nil
}

// MockNeuronDBClient is a mock NeuronDB client for testing
type MockNeuronDBClient struct {
	Collections   []map[string]interface{}
	SearchResults []map[string]interface{}
	QueryResults  []map[string]interface{}
	CallError     error
}

// NewMockNeuronDBClient creates a new mock NeuronDB client
func NewMockNeuronDBClient() *MockNeuronDBClient {
	return &MockNeuronDBClient{
		Collections: []map[string]interface{}{
			{
				"name":       "documents",
				"schema":     "public",
				"vector_col": "embedding",
				"row_count":  100,
			},
		},
		SearchResults: []map[string]interface{}{
			{
				"id":       1,
				"score":    0.95,
				"distance": 0.05,
				"data": map[string]interface{}{
					"id":      1,
					"content": "Test document",
				},
			},
		},
	}
}

// ListCollections returns mock collections
func (m *MockNeuronDBClient) ListCollections(ctx context.Context) ([]map[string]interface{}, error) {
	if m.CallError != nil {
		return nil, m.CallError
	}
	return m.Collections, nil
}

// Search performs a mock search
func (m *MockNeuronDBClient) Search(ctx context.Context, collection, schema string, queryVector []float64, limit int, filter map[string]interface{}) ([]map[string]interface{}, error) {
	if m.CallError != nil {
		return nil, m.CallError
	}
	return m.SearchResults, nil
}

// ExecuteSQL executes a mock SQL query
func (m *MockNeuronDBClient) ExecuteSQL(ctx context.Context, query string) ([]map[string]interface{}, error) {
	if m.CallError != nil {
		return nil, m.CallError
	}
	return m.QueryResults, nil
}

// Helper function to create JSON-RPC request for MCP
func CreateMCPRequest(method string, params interface{}) ([]byte, error) {
	request := map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  method,
	}
	if params != nil {
		request["params"] = params
	}
	return json.Marshal(request)
}

// Helper function to parse JSON-RPC response
func ParseMCPResponse(data []byte) (map[string]interface{}, error) {
	var response map[string]interface{}
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, err
	}
	return response, nil
}
