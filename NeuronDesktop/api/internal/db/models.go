package db

import (
	"time"
)

/* Profile represents a connection profile */
type Profile struct {
	ID                string                 `json:"id"`
	Name              string                 `json:"name"`
	UserID            string                 `json:"user_id"`
	ProfileUsername   string                 `json:"profile_username,omitempty"` // Username for this profile
	ProfilePassword   string                 `json:"-"`                          // Password hash (never serialize)
	MCPConfig         map[string]interface{} `json:"mcp_config"`
	NeuronDBDSN       string                 `json:"neurondb_dsn"`
	AgentEndpoint     string                 `json:"agent_endpoint,omitempty"`
	AgentAPIKey       string                 `json:"agent_api_key,omitempty"`
	DefaultCollection string                 `json:"default_collection,omitempty"`
	IsDefault         bool                   `json:"is_default"`
	CreatedAt         time.Time              `json:"created_at"`
	UpdatedAt         time.Time              `json:"updated_at"`
}

/* APIKey represents an API key */
type APIKey struct {
	ID         string     `json:"id"`
	KeyHash    string     `json:"-"`
	KeyPrefix  string     `json:"key_prefix"`
	UserID     string     `json:"user_id"`
	ProfileID  *string    `json:"profile_id,omitempty"`
	RateLimit  int        `json:"rate_limit"`
	LastUsedAt *time.Time `json:"last_used_at,omitempty"`
	CreatedAt  time.Time  `json:"created_at"`
}

/* RequestLog represents a logged request/response */
type RequestLog struct {
	ID           string                 `json:"id"`
	ProfileID    *string                `json:"profile_id,omitempty"`
	Endpoint     string                 `json:"endpoint"`
	Method       string                 `json:"method"`
	RequestBody  map[string]interface{} `json:"request_body,omitempty"`
	ResponseBody map[string]interface{} `json:"response_body,omitempty"`
	StatusCode   int                    `json:"status_code"`
	DurationMS   int                    `json:"duration_ms"`
	CreatedAt    time.Time              `json:"created_at"`
}

/* AuditLog represents an audit log entry */
type AuditLog struct {
	ID           string                 `json:"id"`
	UserID       string                 `json:"user_id"`
	Action       string                 `json:"action"`
	ResourceType string                 `json:"resource_type"`
	ResourceID   *string                `json:"resource_id,omitempty"`
	Details      map[string]interface{} `json:"details,omitempty"`
	IPAddress    *string                `json:"ip_address,omitempty"`
	UserAgent    *string                `json:"user_agent,omitempty"`
	CreatedAt    time.Time              `json:"created_at"`
}

/* ModelConfig represents a model configuration */
type ModelConfig struct {
	ID            string                 `json:"id"`
	ProfileID     string                 `json:"profile_id"`
	ModelProvider string                 `json:"model_provider"`    // 'openai', 'anthropic', 'google', 'ollama', 'custom'
	ModelName     string                 `json:"model_name"`        // 'gpt-4', 'claude-3-opus', 'gemini-pro', 'llama2', etc.
	APIKey        string                 `json:"api_key,omitempty"` // Only returned if explicitly requested
	BaseURL       string                 `json:"base_url,omitempty"`
	IsDefault     bool                   `json:"is_default"`
	IsFree        bool                   `json:"is_free"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt     time.Time              `json:"created_at"`
	UpdatedAt     time.Time              `json:"updated_at"`
}

/* AppSetting represents an application setting */
type AppSetting struct {
	Key       string                 `json:"key"`
	Value     map[string]interface{} `json:"value"`
	UpdatedAt time.Time              `json:"updated_at"`
}

/* User represents a user account */
type User struct {
	ID           string    `json:"id"`
	Username     string    `json:"username"`
	IsAdmin      bool      `json:"is_admin"`
	PasswordHash string    `json:"-"` // Never serialize password hash
	CreatedAt    time.Time `json:"created_at"`
	UpdatedAt    time.Time `json:"updated_at"`
}

/* Organization represents an organization/workspace */
type Organization struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Slug        string                 `json:"slug"`
	Description string                 `json:"description,omitempty"`
	Settings    map[string]interface{} `json:"settings,omitempty"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

/* OrganizationMember represents a user's membership in an organization */
type OrganizationMember struct {
	ID             string    `json:"id"`
	OrganizationID string    `json:"organization_id"`
	UserID         string    `json:"user_id"`
	Role           string    `json:"role"` // 'owner', 'admin', 'member', 'viewer'
	CreatedAt      time.Time `json:"created_at"`
	UpdatedAt      time.Time `json:"updated_at"`
}
