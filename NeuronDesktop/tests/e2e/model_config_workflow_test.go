package e2e

import (
	"context"
	"net/http"
	"testing"

	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/testing"
)

// TestModelConfigWorkflow_CRUD tests complete model configuration CRUD operations
func TestModelConfigWorkflow_CRUD(t *testing.T) {
	tdb := testing.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testing.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()

	// Authenticate
	err := client.Authenticate(ctx, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to authenticate: %v", err)
	}

	// Create a test profile
	profile, err := testing.CreateTestProfile(ctx, tdb.Queries, client.UserID)
	if err != nil {
		t.Fatalf("Failed to create test profile: %v", err)
	}

	// Step 1: Create model config
	t.Run("create_model_config", func(t *testing.T) {
		createReq := map[string]interface{}{
			"model_provider": "openai",
			"model_name":     "gpt-4",
			"api_key":        "sk-test-key",
			"is_default":     false,
		}

		resp, err := client.Post("/api/v1/profiles/"+profile.ID+"/models", createReq)
		if err != nil {
			t.Fatalf("Create model config request failed: %v", err)
		}
		defer resp.Body.Close()

		testing.AssertStatus(t, resp, http.StatusCreated)

		var config db.ModelConfig
		if err := testing.ParseResponse(t, resp, &config); err != nil {
			t.Fatalf("Failed to parse model config response: %v", err)
		}

		if config.ModelProvider != "openai" {
			t.Errorf("Expected model_provider 'openai', got %s", config.ModelProvider)
		}
		if config.ModelName != "gpt-4" {
			t.Errorf("Expected model_name 'gpt-4', got %s", config.ModelName)
		}
	})

	// Step 2: List model configs
	t.Run("list_model_configs", func(t *testing.T) {
		resp, err := client.Get("/api/v1/profiles/" + profile.ID + "/models")
		if err != nil {
			t.Fatalf("List model configs request failed: %v", err)
		}
		defer resp.Body.Close()

		testing.AssertStatus(t, resp, http.StatusOK)

		var configs []db.ModelConfig
		if err := testing.ParseResponse(t, resp, &configs); err != nil {
			t.Fatalf("Failed to parse model configs response: %v", err)
		}

		if len(configs) == 0 {
			t.Error("Expected at least one model config")
		}
	})

	// Step 3: Update model config
	t.Run("update_model_config", func(t *testing.T) {
		// Create a config to update
		config := &db.ModelConfig{
			ProfileID:     profile.ID,
			ModelProvider: "anthropic",
			ModelName:     "claude-3-opus",
			APIKey:        "sk-old-key",
			IsDefault:     false,
		}
		if err := tdb.Queries.CreateModelConfig(ctx, config); err != nil {
			t.Fatalf("Failed to create model config: %v", err)
		}

		// Update the config
		updateReq := map[string]interface{}{
			"model_provider": "anthropic",
			"model_name":     "claude-3-opus",
			"api_key":        "sk-new-key",
		}

		resp, err := client.Put("/api/v1/profiles/"+profile.ID+"/models/"+config.ID, updateReq)
		if err != nil {
			t.Fatalf("Update model config request failed: %v", err)
		}
		defer resp.Body.Close()

		testing.AssertStatus(t, resp, http.StatusOK)

		var updatedConfig db.ModelConfig
		if err := testing.ParseResponse(t, resp, &updatedConfig); err != nil {
			t.Fatalf("Failed to parse updated model config response: %v", err)
		}

		if updatedConfig.APIKey != "sk-new-key" {
			t.Errorf("Expected api_key 'sk-new-key', got %s", updatedConfig.APIKey)
		}
	})

	// Step 4: Set default model
	t.Run("set_default_model", func(t *testing.T) {
		// Create a config to set as default
		config := &db.ModelConfig{
			ProfileID:     profile.ID,
			ModelProvider: "openai",
			ModelName:     "gpt-3.5-turbo",
			APIKey:        "sk-test-key",
			IsDefault:     false,
		}
		if err := tdb.Queries.CreateModelConfig(ctx, config); err != nil {
			t.Fatalf("Failed to create model config: %v", err)
		}

		// Set as default
		resp, err := client.Post("/api/v1/profiles/"+profile.ID+"/models/"+config.ID+"/set-default", nil)
		if err != nil {
			t.Fatalf("Set default model request failed: %v", err)
		}
		defer resp.Body.Close()

		testing.AssertStatus(t, resp, http.StatusOK)

		// Verify it's now default
		updatedConfig, err := tdb.Queries.GetModelConfig(ctx, config.ID, false)
		if err != nil {
			t.Fatalf("Failed to get model config: %v", err)
		}

		if !updatedConfig.IsDefault {
			t.Error("Expected model config to be set as default")
		}
	})

	// Step 5: Delete model config
	t.Run("delete_model_config", func(t *testing.T) {
		// Create a config to delete
		config := &db.ModelConfig{
			ProfileID:     profile.ID,
			ModelProvider: "ollama",
			ModelName:     "llama2",
			BaseURL:       "http://localhost:11434",
			IsFree:        true,
			IsDefault:     false,
		}
		if err := tdb.Queries.CreateModelConfig(ctx, config); err != nil {
			t.Fatalf("Failed to create model config: %v", err)
		}

		// Delete the config
		resp, err := client.Delete("/api/v1/profiles/" + profile.ID + "/models/" + config.ID)
		if err != nil {
			t.Fatalf("Delete model config request failed: %v", err)
		}
		defer resp.Body.Close()

		testing.AssertStatus(t, resp, http.StatusNoContent)

		// Verify deletion
		_, err = tdb.Queries.GetModelConfig(ctx, config.ID, false)
		if err == nil {
			t.Error("Expected model config to be deleted")
		}
	})
}






