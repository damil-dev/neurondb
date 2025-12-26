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
	"encoding/json"
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
	db     *database.Database
	logger *logging.Logger
}

/* NewManager creates a new sampling manager */
func NewManager(db *database.Database, logger *logging.Logger) *Manager {
	return &Manager{
		db:     db,
		logger: logger,
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
		/* Get default chat model */
		defaultModel, err := m.getDefaultChatModel(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to get default model: %w", err)
		}
		modelName = defaultModel
	}

	if modelName == "" {
		return nil, fmt.Errorf("model name cannot be empty")
	}

	/* Build prompt from messages */
	prompt := m.buildPromptFromMessages(req.Messages)
	if prompt == "" {
		return nil, fmt.Errorf("generated prompt is empty")
	}

	/* Prepare LLM parameters */
	llmParams := m.buildLLMParams(req)

	/* Call NeuronDB LLM completion function */
	query := `
		SELECT neurondb_llm_completion(
			$1::text,
			$2::text,
			$3::jsonb
		) AS completion
	`

	var completion string
	err := m.db.QueryRow(ctx, query, modelName, prompt, llmParams).Scan(&completion)
	if err != nil {
		/* Fallback: try direct SQL function call */
		fallbackQuery := `
			SELECT neurondb_llm_chat(
				$1::text,
				$2::jsonb,
				$3::jsonb
			) AS response
		`
		
		messagesJSON, _ := json.Marshal(req.Messages)
		err = m.db.QueryRow(ctx, fallbackQuery, modelName, string(messagesJSON), llmParams).Scan(&completion)
		if err != nil {
			return nil, fmt.Errorf("failed to generate completion: %w", err)
		}
	}

	return &SamplingResponse{
		Content:  completion,
		Model:     modelName,
		StopReason: "stop",
		Metadata:  req.Metadata,
	}, nil
}

/* getDefaultChatModel gets the default chat model */
func (m *Manager) getDefaultChatModel(ctx context.Context) (string, error) {
	query := `
		SELECT model_name
		FROM neurondb.llm_models
		WHERE model_type = 'chat'
			AND is_default = true
			AND status = 'available'
		LIMIT 1
	`

	var modelName string
	err := m.db.QueryRow(ctx, query).Scan(&modelName)
	if err != nil {
		/* Fallback to any available chat model */
		fallbackQuery := `
			SELECT model_name
			FROM neurondb.llm_models
			WHERE model_type = 'chat'
				AND status = 'available'
			LIMIT 1
		`
		err = m.db.QueryRow(ctx, fallbackQuery).Scan(&modelName)
		if err != nil {
			return "", fmt.Errorf("no chat model available: %w", err)
		}
	}

	return modelName, nil
}

/* buildPromptFromMessages builds a prompt string from messages */
func (m *Manager) buildPromptFromMessages(messages []Message) string {
	if len(messages) == 0 {
		return ""
	}

	var builder strings.Builder
	for i, msg := range messages {
		if i > 0 {
			builder.WriteString("\n\n")
		}

		switch strings.ToLower(msg.Role) {
		case "system":
			builder.WriteString(fmt.Sprintf("System: %s", msg.Content))
		case "user":
			builder.WriteString(fmt.Sprintf("User: %s", msg.Content))
		case "assistant":
			builder.WriteString(fmt.Sprintf("Assistant: %s", msg.Content))
		default:
			builder.WriteString(fmt.Sprintf("%s: %s", msg.Role, msg.Content))
		}
	}
	return builder.String()
}

/* buildLLMParams builds LLM parameters from request */
func (m *Manager) buildLLMParams(req SamplingRequest) json.RawMessage {
	params := make(map[string]interface{})
	
	/* Temperature validation */
	if req.Temperature != nil {
		temp := *req.Temperature
		if temp < 0.0 {
			temp = 0.0
		}
		if temp > 2.0 {
			temp = 2.0
		}
		params["temperature"] = temp
	} else {
		params["temperature"] = 0.7
	}
	
	/* MaxTokens validation */
	if req.MaxTokens != nil {
		maxTokens := *req.MaxTokens
		if maxTokens < 1 {
			maxTokens = 1
		}
		if maxTokens > 100000 {
			maxTokens = 100000
		}
		params["max_tokens"] = maxTokens
	}
	
	/* TopP validation */
	if req.TopP != nil {
		topP := *req.TopP
		if topP < 0.0 {
			topP = 0.0
		}
		if topP > 1.0 {
			topP = 1.0
		}
		params["top_p"] = topP
	}

	paramsJSON, err := json.Marshal(params)
	if err != nil {
		/* Fallback to default params if marshaling fails */
		defaultParams := map[string]interface{}{
			"temperature": 0.7,
		}
		paramsJSON, _ = json.Marshal(defaultParams)
	}
	return json.RawMessage(paramsJSON)
}

