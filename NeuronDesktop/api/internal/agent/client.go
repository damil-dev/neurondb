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

/* HTTPError represents an HTTP error response */
type HTTPError struct {
	StatusCode int
	Message    string
	Body       string
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("HTTP %d: %s", e.StatusCode, e.Message)
}

/* Client provides HTTP access to NeuronAgent */
type Client struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
}

/* NewClient creates a new NeuronAgent client */
func NewClient(baseURL, apiKey string) *Client {
	return &Client{
		baseURL: baseURL,
		apiKey:  apiKey,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

/* Agent represents an agent in NeuronAgent */
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

/* Session represents a session in NeuronAgent */
type Session struct {
	ID             string                 `json:"id"`
	AgentID        string                 `json:"agent_id"`
	ExternalUserID string                 `json:"external_user_id,omitempty"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt      string                 `json:"created_at,omitempty"`
}

/* Message represents a message in a session */
type Message struct {
	ID        string                 `json:"id"`
	SessionID string                 `json:"session_id"`
	Role      string                 `json:"role"`
	Content   string                 `json:"content"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt string                 `json:"created_at,omitempty"`
}

/* CreateAgentRequest is the request to create an agent */
type CreateAgentRequest struct {
	Name         string                 `json:"name"`
	Description  string                 `json:"description,omitempty"`
	SystemPrompt string                 `json:"system_prompt,omitempty"`
	ModelName    string                 `json:"model_name,omitempty"`
	EnabledTools []string               `json:"enabled_tools,omitempty"`
	Config       map[string]interface{} `json:"config,omitempty"`
}

/* UpdateAgentRequest is the request to update an agent */
type UpdateAgentRequest struct {
	Name         string                 `json:"name,omitempty"`
	Description  string                 `json:"description,omitempty"`
	SystemPrompt string                 `json:"system_prompt,omitempty"`
	ModelName    string                 `json:"model_name,omitempty"`
	EnabledTools []string               `json:"enabled_tools,omitempty"`
	Config       map[string]interface{} `json:"config,omitempty"`
}

/* CreateSessionRequest is the request to create a session */
type CreateSessionRequest struct {
	AgentID        string                 `json:"agent_id"`
	ExternalUserID string                 `json:"external_user_id,omitempty"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
}

/* SendMessageRequest is the request to send a message */
type SendMessageRequest struct {
	Role     string                 `json:"role"`
	Content  string                 `json:"content"`
	Stream   bool                   `json:"stream,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

/* ListAgents lists all agents */
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

/* GetAgent gets a single agent */
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

/* CreateAgent creates a new agent */
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

/* UpdateAgent updates an existing agent */
func (c *Client) UpdateAgent(ctx context.Context, id string, req UpdateAgentRequest) (*Agent, error) {
	httpReq, err := c.newRequest(ctx, "PUT", fmt.Sprintf("/api/v1/agents/%s", id), req)
	if err != nil {
		return nil, err
	}

	var agent Agent
	if err := c.doRequest(httpReq, &agent); err != nil {
		return nil, err
	}

	return &agent, nil
}

/* DeleteAgent deletes an agent */
func (c *Client) DeleteAgent(ctx context.Context, id string) error {
	req, err := c.newRequest(ctx, "DELETE", fmt.Sprintf("/api/v1/agents/%s", id), nil)
	if err != nil {
		return err
	}

	return c.doRequest(req, nil)
}

/* CreateSession creates a new session */
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

/* GetSession gets a session */
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

/* SendMessage sends a message to a session */
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

/* GetMessages gets messages from a session */
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

/* ListSessions lists sessions for an agent */
func (c *Client) ListSessions(ctx context.Context, agentID string) ([]Session, error) {
	req, err := c.newRequest(ctx, "GET", fmt.Sprintf("/api/v1/agents/%s/sessions", agentID), nil)
	if err != nil {
		return nil, err
	}

	var sessions []Session
	if err := c.doRequest(req, &sessions); err != nil {
		return nil, err
	}

	return sessions, nil
}

/* Model represents a model in NeuronAgent */
type Model struct {
	ID          string                 `json:"id,omitempty"`
	Name        string                 `json:"name"`
	DisplayName string                 `json:"display_name,omitempty"`
	Provider    string                 `json:"provider,omitempty"`
	Description string                 `json:"description,omitempty"`
	Config      map[string]interface{} `json:"config,omitempty"`
}

/* ListModels lists available models from NeuronAgent */
func (c *Client) ListModels(ctx context.Context) ([]Model, error) {
	req, err := c.newRequest(ctx, "GET", "/api/v1/models", nil)
	if err != nil {
		return nil, err
	}

	var response struct {
		Models []Model `json:"models"`
	}
	if err := c.doRequest(req, &response); err != nil {
		/* If endpoint doesn't exist, return empty list (backward compatibility) */
		if httpErr, ok := err.(*HTTPError); ok && httpErr.StatusCode == 404 {
			return []Model{}, nil
		}
		return nil, err
	}

	return response.Models, nil
}

/* Helper methods */

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
		return &HTTPError{
			StatusCode: resp.StatusCode,
			Message:    fmt.Sprintf("API error: %s", string(body)),
			Body:       string(body),
		}
	}

	if result != nil {
		if err := json.Unmarshal(body, result); err != nil {
			return fmt.Errorf("failed to unmarshal response: %w", err)
		}
	}

	return nil
}
