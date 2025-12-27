package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client provides HTTP access to NeuronAgent
type Client struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
}

// NewClient creates a new NeuronAgent client
func NewClient(baseURL, apiKey string) *Client {
	return &Client{
		baseURL: baseURL,
		apiKey:  apiKey,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// Agent represents an agent in NeuronAgent
type Agent struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	Description  string                 `json:"description,omitempty"`
	SystemPrompt string                 `json:"system_prompt,omitempty"`
	ModelName    string                 `json:"model_name,omitempty"`
	EnabledTools []string               `json:"enabled_tools,omitempty"`
	Config       map[string]interface{} `json:"config,omitempty"`
	CreatedAt    string                 `json:"created_at,omitempty"`
}

// Session represents a session in NeuronAgent
type Session struct {
	ID            string                 `json:"id"`
	AgentID       string                 `json:"agent_id"`
	ExternalUserID string                `json:"external_user_id,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt     string                 `json:"created_at,omitempty"`
}

// Message represents a message in a session
type Message struct {
	ID        string                 `json:"id"`
	SessionID string                 `json:"session_id"`
	Role      string                 `json:"role"`
	Content   string                 `json:"content"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt string                 `json:"created_at,omitempty"`
}

// CreateAgentRequest is the request to create an agent
type CreateAgentRequest struct {
	Name         string                 `json:"name"`
	Description  string                 `json:"description,omitempty"`
	SystemPrompt string                 `json:"system_prompt,omitempty"`
	ModelName    string                 `json:"model_name,omitempty"`
	EnabledTools []string               `json:"enabled_tools,omitempty"`
	Config       map[string]interface{} `json:"config,omitempty"`
}

// CreateSessionRequest is the request to create a session
type CreateSessionRequest struct {
	AgentID       string                 `json:"agent_id"`
	ExternalUserID string                `json:"external_user_id,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

// SendMessageRequest is the request to send a message
type SendMessageRequest struct {
	Role    string                 `json:"role"`
	Content string                 `json:"content"`
	Stream  bool                   `json:"stream,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// ListAgents lists all agents
func (c *Client) ListAgents(ctx context.Context) ([]Agent, error) {
	req, err := c.newRequest(ctx, "GET", "/api/v1/agents", nil)
	if err != nil {
		return nil, err
	}
	
	var agents []Agent
	if err := c.doRequest(req, &agents); err != nil {
		return nil, err
	}
	
	return agents, nil
}

// GetAgent gets a single agent
func (c *Client) GetAgent(ctx context.Context, id string) (*Agent, error) {
	req, err := c.newRequest(ctx, "GET", fmt.Sprintf("/api/v1/agents/%s", id), nil)
	if err != nil {
		return nil, err
	}
	
	var agent Agent
	if err := c.doRequest(req, &agent); err != nil {
		return nil, err
	}
	
	return &agent, nil
}

// CreateAgent creates a new agent
func (c *Client) CreateAgent(ctx context.Context, req CreateAgentRequest) (*Agent, error) {
	httpReq, err := c.newRequest(ctx, "POST", "/api/v1/agents", req)
	if err != nil {
		return nil, err
	}
	
	var agent Agent
	if err := c.doRequest(httpReq, &agent); err != nil {
		return nil, err
	}
	
	return &agent, nil
}

// CreateSession creates a new session
func (c *Client) CreateSession(ctx context.Context, req CreateSessionRequest) (*Session, error) {
	httpReq, err := c.newRequest(ctx, "POST", "/api/v1/sessions", req)
	if err != nil {
		return nil, err
	}
	
	var session Session
	if err := c.doRequest(httpReq, &session); err != nil {
		return nil, err
	}
	
	return &session, nil
}

// GetSession gets a session
func (c *Client) GetSession(ctx context.Context, id string) (*Session, error) {
	req, err := c.newRequest(ctx, "GET", fmt.Sprintf("/api/v1/sessions/%s", id), nil)
	if err != nil {
		return nil, err
	}
	
	var session Session
	if err := c.doRequest(req, &session); err != nil {
		return nil, err
	}
	
	return &session, nil
}

// SendMessage sends a message to a session
func (c *Client) SendMessage(ctx context.Context, sessionID string, req SendMessageRequest) (*Message, error) {
	httpReq, err := c.newRequest(ctx, "POST", fmt.Sprintf("/api/v1/sessions/%s/messages", sessionID), req)
	if err != nil {
		return nil, err
	}
	
	var message Message
	if err := c.doRequest(httpReq, &message); err != nil {
		return nil, err
	}
	
	return &message, nil
}

// GetMessages gets messages from a session
func (c *Client) GetMessages(ctx context.Context, sessionID string) ([]Message, error) {
	req, err := c.newRequest(ctx, "GET", fmt.Sprintf("/api/v1/sessions/%s/messages", sessionID), nil)
	if err != nil {
		return nil, err
	}
	
	var messages []Message
	if err := c.doRequest(req, &messages); err != nil {
		return nil, err
	}
	
	return messages, nil
}

// Helper methods

func (c *Client) newRequest(ctx context.Context, method, path string, body interface{}) (*http.Request, error) {
	var bodyReader io.Reader
	if body != nil {
		bodyBytes, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
		bodyReader = bytes.NewReader(bodyBytes)
	}
	
	url := c.baseURL + path
	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	
	return req, nil
}

func (c *Client) doRequest(req *http.Request, result interface{}) error {
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()
	
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response: %w", err)
	}
	
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("API error: %s (status: %d)", string(body), resp.StatusCode)
	}
	
	if result != nil {
		if err := json.Unmarshal(body, result); err != nil {
			return fmt.Errorf("failed to unmarshal response: %w", err)
		}
	}
	
	return nil
}

