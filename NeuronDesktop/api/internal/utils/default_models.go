package utils

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
)

// DefaultModelConfig represents a default model configuration template
type DefaultModelConfig struct {
	ModelProvider string
	ModelName     string
	IsFree        bool
	IsDefault     bool
	RequiresKey   bool
}

// GetDefaultModels returns a list of default model configurations to create for new profiles
func GetDefaultModels() []DefaultModelConfig {
	return []DefaultModelConfig{
		// OpenAI Models
		{
			ModelProvider: "openai",
			ModelName:     "gpt-4o",
			IsFree:        false,
			IsDefault:     true, // First one is default
			RequiresKey:   true,
		},
		{
			ModelProvider: "openai",
			ModelName:     "gpt-4-turbo",
			IsFree:        false,
			IsDefault:     false,
			RequiresKey:   true,
		},
		{
			ModelProvider: "openai",
			ModelName:     "gpt-4",
			IsFree:        false,
			IsDefault:     false,
			RequiresKey:   true,
		},
		{
			ModelProvider: "openai",
			ModelName:     "gpt-3.5-turbo",
			IsFree:        false,
			IsDefault:     false,
			RequiresKey:   true,
		},
		// Anthropic Models
		{
			ModelProvider: "anthropic",
			ModelName:     "claude-3-5-sonnet-20241022",
			IsFree:        false,
			IsDefault:     false,
			RequiresKey:   true,
		},
		{
			ModelProvider: "anthropic",
			ModelName:     "claude-3-opus-20240229",
			IsFree:        false,
			IsDefault:     false,
			RequiresKey:   true,
		},
		{
			ModelProvider: "anthropic",
			ModelName:     "claude-3-sonnet-20240229",
			IsFree:        false,
			IsDefault:     false,
			RequiresKey:   true,
		},
		{
			ModelProvider: "anthropic",
			ModelName:     "claude-3-haiku-20240307",
			IsFree:        false,
			IsDefault:     false,
			RequiresKey:   true,
		},
		// Google Models
		{
			ModelProvider: "google",
			ModelName:     "gemini-1.5-pro",
			IsFree:        false,
			IsDefault:     false,
			RequiresKey:   true,
		},
		{
			ModelProvider: "google",
			ModelName:     "gemini-pro",
			IsFree:        false,
			IsDefault:     false,
			RequiresKey:   true,
		},
		// Ollama (Free) - Optional local model
		{
			ModelProvider: "ollama",
			ModelName:     "llama2",
			IsFree:        true,
			IsDefault:     false,
			RequiresKey:   false,
		},
	}
}

// CreateDefaultModelsForProfile creates default model configurations for a new profile
func CreateDefaultModelsForProfile(ctx context.Context, queries *db.Queries, profileID string) error {
	defaultModels := GetDefaultModels()

	var defaultModelID string

	for _, modelTemplate := range defaultModels {
		modelConfig := &db.ModelConfig{
			ID:            uuid.New().String(),
			ProfileID:     profileID,
			ModelProvider: modelTemplate.ModelProvider,
			ModelName:     modelTemplate.ModelName,
			APIKey:        "", // User will set this in settings
			BaseURL:       "", // Use default URLs
			IsDefault:     modelTemplate.IsDefault,
			IsFree:        modelTemplate.IsFree,
			Metadata:      make(map[string]interface{}),
			CreatedAt:     time.Now(),
			UpdatedAt:     time.Now(),
		}

		// Set default base URL for Ollama
		if modelTemplate.ModelProvider == "ollama" {
			modelConfig.BaseURL = "http://localhost:11434"
		}

		if err := queries.CreateModelConfig(ctx, modelConfig); err != nil {
			return fmt.Errorf("failed to create default model %s/%s: %w", modelTemplate.ModelProvider, modelTemplate.ModelName, err)
		}

		if modelTemplate.IsDefault {
			defaultModelID = modelConfig.ID
		}
	}

	// Set the first model (gpt-4o) as default if we created models
	if defaultModelID != "" {
		if err := queries.SetDefaultModelConfig(ctx, profileID, defaultModelID); err != nil {
			// Log error but don't fail - models are created
			fmt.Printf("Warning: Failed to set default model config: %v\n", err)
		}
	}

	return nil
}
