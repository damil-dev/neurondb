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
	"github.com/neurondb/NeuronMCP/internal/tools"
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
	db           *database.Database
	logger       *logging.Logger
	llmClient    *LLMClient
	toolRegistry *tools.ToolRegistry
}

/* NewManager creates a new sampling manager */
func NewManager(db *database.Database, logger *logging.Logger) *Manager {
	return &Manager{
		db:        db,
		logger:    logger,
		llmClient: NewLLMClient(),
	}
}

/* SetToolRegistry sets the tool registry for tool-aware sampling */
func (m *Manager) SetToolRegistry(registry *tools.ToolRegistry) {
	m.toolRegistry = registry
}

/* CreateMessage creates a completion from messages with tool calling support */
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

	/* Build LLM parameters */
	temperature := 0.7
	if req.Temperature != nil {
		temperature = *req.Temperature
	}
	/* Increase default max_tokens to give model more room for complete responses */
	/* Tool calls need space for JSON formatting */
	/* Set to 8192 to allow for large prompts + complete tool call responses */
	maxTokens := 8192
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}

	/* Start agent loop with tool support */
	maxIterations := 5
	messages := make([]Message, len(req.Messages))
	copy(messages, req.Messages)

	for iteration := 0; iteration < maxIterations; iteration++ {
		m.logger.Debug(fmt.Sprintf("Agent iteration %d/%d", iteration+1, maxIterations), map[string]interface{}{
			"message_count": len(messages),
		})

		/* Build prompt with tool awareness */
		prompt := m.buildPromptWithTools(messages)

		/* Call LLM */
		completion, err := m.callLLM(ctx, prompt, temperature, maxTokens)
		if err != nil {
			return nil, fmt.Errorf("failed to call LLM: %w", err)
		}

		/* Check if LLM wants to call a tool */
		toolCall, err := m.parseToolCall(completion)
		if err != nil {
			/* Not a tool call, return as final response */
			m.logger.Debug("No tool call detected, returning response", nil)
			return &SamplingResponse{
				Content:    completion,
				Model:      modelName,
				StopReason: "stop",
				Metadata:   req.Metadata,
			}, nil
		}

		/* Execute the tool */
		m.logger.Info("Executing tool", map[string]interface{}{
			"tool_name": toolCall.Name,
			"arguments": toolCall.Arguments,
		})

		toolResult, err := m.executeTool(ctx, toolCall.Name, toolCall.Arguments)
		if err != nil {
			m.logger.Error("Tool execution failed", err, map[string]interface{}{
				"tool_name": toolCall.Name,
			})
			/* Return error to user */
			return nil, fmt.Errorf("tool execution failed: %w", err)
		}

		/* Add assistant's tool call and tool result to conversation */
		messages = append(messages, Message{
			Role:    "assistant",
			Content: fmt.Sprintf("I'll use the tool '%s' to help answer your question.", toolCall.Name),
		})

		toolResultStr := fmt.Sprintf("Tool '%s' returned: %v", toolCall.Name, toolResult)
		messages = append(messages, Message{
			Role:    "tool_result",
			Content: toolResultStr,
		})

		/* Continue loop to let LLM process the result */
	}

	/* If we've exhausted iterations, return what we have */
	return &SamplingResponse{
		Content:    "I've attempted to process your request but reached the maximum number of tool calls. Please try rephrasing your question.",
		Model:      modelName,
		StopReason: "max_tokens",
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

/* buildPromptWithTools builds a prompt including available tools */
func (m *Manager) buildPromptWithTools(messages []Message) string {
	var promptBuilder strings.Builder

	/* Add system message with tool information if we have tools */
	if m.toolRegistry != nil {
		toolDefs := m.toolRegistry.GetAllDefinitions()
		if len(toolDefs) > 0 {
			promptBuilder.WriteString("You are an AI assistant with PostgreSQL database tools.\n\n")
			promptBuilder.WriteString("Rules:\n")
			promptBuilder.WriteString("1. For database queries -> use TOOL_CALL: {\"name\": \"tool_name\", \"arguments\": {...}}\n")
			promptBuilder.WriteString("2. For general knowledge -> answer directly\n")
			promptBuilder.WriteString("3. Complete your response - don't truncate JSON\n\n")
			promptBuilder.WriteString("Key tools: postgresql_version, postgresql_stats, postgresql_connections, vector_search, generate_embedding\n")
			promptBuilder.WriteString("Example: TOOL_CALL: {\"name\": \"postgresql_version\", \"arguments\": {}}\n\n")
		}
	}	/* Add conversation messages */
	for _, msg := range messages {
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
		case "tool_result":
			promptBuilder.WriteString("Tool Result: ")
			promptBuilder.WriteString(msg.Content)
			promptBuilder.WriteString("\n\n")
		default:
			promptBuilder.WriteString(fmt.Sprintf("%s: %s\n\n", msg.Role, msg.Content))
		}
	}

	/* Add final prompt for assistant response */
	promptBuilder.WriteString("Assistant: ")
	return promptBuilder.String()
}

/* callLLM calls the NeuronDB LLM function */
func (m *Manager) callLLM(ctx context.Context, prompt string, temperature float64, maxTokens int) (string, error) {
	llmParamsJSON := fmt.Sprintf(`{"temperature": %.2f, "max_tokens": %d}`, temperature, maxTokens)

	/* Call neurondb.llm() function directly */
	/* neurondb.llm(task, model, input_text, input_array, params, max_length) returns JSONB */
	/* Task 'complete' generates text completion */
	/* Use NULL for model to use the configured default (neurondb.llm_model setting) */
	query := `SELECT (neurondb.llm('complete', NULL, $1, NULL, $2::jsonb, $3)->>'text') AS response`

	var completion string
	err := m.db.QueryRow(ctx, query, prompt, llmParamsJSON, maxTokens).Scan(&completion)
	if err != nil {
		return "", fmt.Errorf("failed to call neurondb.llm(): %w", err)
	}

	if completion == "" {
		return "", fmt.Errorf("empty response from neurondb.llm()")
	}

	return completion, nil
}

/* ToolCall represents a parsed tool call from LLM response */
type ToolCall struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

/* parseToolCall attempts to parse a tool call from LLM response */
func (m *Manager) parseToolCall(response string) (*ToolCall, error) {
	/* Look for TOOL_CALL: pattern */
	if !strings.Contains(response, "TOOL_CALL:") {
		return nil, fmt.Errorf("no tool call detected")
	}

	/* Extract JSON after TOOL_CALL: */
	parts := strings.SplitN(response, "TOOL_CALL:", 2)
	if len(parts) < 2 {
		return nil, fmt.Errorf("invalid tool call format")
	}

	jsonStr := strings.TrimSpace(parts[1])

	/* Parse JSON */
	var toolCall ToolCall
	decoder := json.NewDecoder(strings.NewReader(jsonStr))
	if err := decoder.Decode(&toolCall); err != nil {
		return nil, fmt.Errorf("failed to parse tool call JSON: %w", err)
	}

	if toolCall.Name == "" {
		return nil, fmt.Errorf("tool call missing name")
	}

	return &toolCall, nil
}

/* executeTool executes a tool by name with given arguments */
func (m *Manager) executeTool(ctx context.Context, name string, arguments map[string]interface{}) (interface{}, error) {
	if m.toolRegistry == nil {
		return nil, fmt.Errorf("tool registry not configured")
	}

	tool := m.toolRegistry.GetTool(name)
	if tool == nil {
		return nil, fmt.Errorf("tool not found: %s", name)
	}

	result, err := tool.Execute(ctx, arguments)
	if err != nil {
		return nil, fmt.Errorf("tool execution error: %w", err)
	}

	if !result.Success {
		if result.Error != nil {
			return nil, fmt.Errorf("tool returned error: %s", result.Error.Message)
		}
		return nil, fmt.Errorf("tool execution failed without error details")
	}

	return result.Data, nil
}
