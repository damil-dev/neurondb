/*-------------------------------------------------------------------------
 *
 * sampling.go
 *    Sampling engine for NeuronMCP
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/sampling/sampling.go
 *
 *-------------------------------------------------------------------------
 */

package sampling

import (
	"context"
	"fmt"
	"strings"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* Message represents a chat message */
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

/* SamplingRequest represents a sampling request */
type SamplingRequest struct {
	Messages    []Message              `json:"messages"`
	Model       string                 `json:"model,omitempty"`
	Temperature *float64               `json:"temperature,omitempty"`
	MaxTokens   *int                   `json:"maxTokens,omitempty"`
	TopP        *float64               `json:"topP,omitempty"`
	Stream      bool                   `json:"stream,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

/* SamplingResponse represents a sampling response */
type SamplingResponse struct {
	Content   string                 `json:"content"`
	Model     string                 `json:"model"`
	StopReason string                `json:"stopReason,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

/* Manager manages sampling operations */
type Manager struct {
	db        *database.Database
	logger    *logging.Logger
	llmClient *LLMClient
}

/* NewManager creates a new sampling manager */
func NewManager(db *database.Database, logger *logging.Logger) *Manager {
	return &Manager{
		db:        db,
		logger:    logger,
		llmClient: NewLLMClient(),
	}
}

/* CreateMessage creates a completion from messages */
func (m *Manager) CreateMessage(ctx context.Context, req SamplingRequest) (*SamplingResponse, error) {
	if ctx == nil {
		return nil, fmt.Errorf("context cannot be nil")
	}

	if len(req.Messages) == 0 {
		return nil, fmt.Errorf("messages array cannot be empty")
	}

	/* Validate messages */
	for i, msg := range req.Messages {
		if msg.Role == "" {
			return nil, fmt.Errorf("message at index %d has empty role", i)
		}
		if msg.Content == "" {
			return nil, fmt.Errorf("message at index %d has empty content", i)
		}
	}

	/* Determine model to use */
	modelName := req.Model
	if modelName == "" {
		return nil, fmt.Errorf("model name is required")
	}

	/* Use NeuronDB's built-in LLM function: neurondb.llm() */
	/* This provides 100% compatibility with Claude Desktop MCP protocol */
	/* while using NeuronDB's native AI capabilities */
	
	/* Build prompt from messages (convert chat format to prompt) */
	var promptBuilder strings.Builder
	for _, msg := range req.Messages {
		switch msg.Role {
		case "user":
			promptBuilder.WriteString("User: ")
			promptBuilder.WriteString(msg.Content)
			promptBuilder.WriteString("\n\n")
		case "assistant":
			promptBuilder.WriteString("Assistant: ")
			promptBuilder.WriteString(msg.Content)
			promptBuilder.WriteString("\n\n")
		case "system":
			promptBuilder.WriteString("System: ")
			promptBuilder.WriteString(msg.Content)
			promptBuilder.WriteString("\n\n")
		default:
			promptBuilder.WriteString(fmt.Sprintf("%s: %s\n\n", msg.Role, msg.Content))
		}
	}
	/* Add final prompt for assistant response */
	promptBuilder.WriteString("Assistant: ")
	prompt := promptBuilder.String()

	/* Build LLM parameters */
	temperature := 0.7
	if req.Temperature != nil {
		temperature = *req.Temperature
	}
	maxTokens := 2048
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}
	
	llmParamsJSON := fmt.Sprintf(`{"temperature": %.2f, "max_tokens": %d}`, temperature, maxTokens)
	
	/* Call neurondb.llm() function directly */
	/* neurondb.llm(task, model, input_text, input_array, params, max_length) returns JSONB */
	/* Task 'complete' generates text completion */
	/* Response format: {"text": "...", "model": "...", "task": "complete"} */
	/* Use NULL for model to use the configured default (neurondb.llm_model setting) */
	query := `SELECT (neurondb.llm('complete', NULL, $1, NULL, $2::jsonb, $3)->>'text') AS response`
	
	var completion string
	err := m.db.QueryRow(ctx, query, prompt, llmParamsJSON, maxTokens).Scan(&completion)
	if err != nil {
		return nil, fmt.Errorf("failed to call neurondb.llm(): %w", err)
	}
	
	if completion == "" {
		return nil, fmt.Errorf("empty response from neurondb.llm()")
	}

	return &SamplingResponse{
		Content:    completion,
		Model:      modelName, /* Return the requested model name for compatibility */
		StopReason: "stop",
		Metadata:   req.Metadata,
	}, nil
}

/* ModelConfig represents a model configuration */
type ModelConfig struct {
	Provider string
	BaseURL  string
	APIKey   string
}

/* getModelConfig gets model configuration */
func (m *Manager) getModelConfig(ctx context.Context, modelName string) (*ModelConfig, error) {
	/* Try to get model config from neurondesk database */
	query := `
		SELECT 
			model_provider,
			COALESCE(base_url, '') as base_url,
			COALESCE(api_key, '') as api_key
		FROM model_configs
		WHERE model_name = $1
		LIMIT 1
	`

	var config ModelConfig
	err := m.db.QueryRow(ctx, query, modelName).Scan(
		&config.Provider,
		&config.BaseURL,
		&config.APIKey,
	)

	if err != nil {
		/* Fallback: assume it's Ollama if no config found */
		m.logger.Warn("Model config not found, assuming Ollama", map[string]interface{}{
			"model": modelName,
		})
		return &ModelConfig{
			Provider: "ollama",
			BaseURL:  "http://localhost:11434",
			APIKey:   "",
		}, nil
	}

	/* Set default base URLs if not configured */
	if config.BaseURL == "" {
		switch config.Provider {
		case "ollama":
			config.BaseURL = "http://localhost:11434"
		case "openai":
			config.BaseURL = "https://api.openai.com/v1"
		case "anthropic":
			config.BaseURL = "https://api.anthropic.com"
		case "google":
			config.BaseURL = "https://generativelanguage.googleapis.com/v1beta"
		}
	}

	return &config, nil
}


