package handlers

import (
	"context"
	"net/http"
	"testing"

	"github.com/neurondb/NeuronDesktop/api/internal/db"
	testutil "github.com/neurondb/NeuronDesktop/api/internal/testing"
)

func TestModelConfigHandlers_ListModelConfigs(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testutil.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()

	err := client.Authenticate(ctx, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to authenticate: %v", err)
	}

	// Create a test profile
	profile, err := testutil.CreateTestProfile(ctx, tdb.Queries, client.UserID)
	if err != nil {
		t.Fatalf("Failed to create test profile: %v", err)
	}

	// Test listing model configs
	resp, err := client.Get("/api/v1/profiles/" + profile.ID + "/models")
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	testutil.AssertStatus(t, resp, http.StatusOK)

	var configs []interface{}
	if err := testutil.ParseResponse(t, resp, &configs); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	// Should return empty array if no configs
	if configs == nil {
		t.Error("Expected configs array")
	}
}

func TestModelConfigHandlers_CreateModelConfig(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testutil.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()

	err := client.Authenticate(ctx, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to authenticate: %v", err)
	}

	// Create a test profile
	profile, err := testutil.CreateTestProfile(ctx, tdb.Queries, client.UserID)
	if err != nil {
		t.Fatalf("Failed to create test profile: %v", err)
	}

	tests := []struct {
		name           string
		request        map[string]interface{}
		expectedStatus int
	}{
		{
			name: "valid OpenAI config",
			request: map[string]interface{}{
				"model_provider": "openai",
				"model_name":     "gpt-4",
				"api_key":        "sk-test-key",
				"is_default":     false,
			},
			expectedStatus: http.StatusCreated,
		},
		{
			name: "valid Ollama config",
			request: map[string]interface{}{
				"model_provider": "ollama",
				"model_name":     "llama2",
				"base_url":       "http://localhost:11434",
				"is_free":        true,
				"is_default":     false,
			},
			expectedStatus: http.StatusCreated,
		},
		{
			name: "missing model_provider",
			request: map[string]interface{}{
				"model_name": "gpt-4",
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "missing model_name",
			request: map[string]interface{}{
				"model_provider": "openai",
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "set as default",
			request: map[string]interface{}{
				"model_provider": "openai",
				"model_name":     "gpt-3.5-turbo",
				"api_key":        "sk-test-key",
				"is_default":     true,
			},
			expectedStatus: http.StatusCreated,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Post("/api/v1/profiles/"+profile.ID+"/models", tt.request)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			testutil.AssertStatus(t, resp, tt.expectedStatus)

			if tt.expectedStatus == http.StatusCreated {
				var config db.ModelConfig
				if err := testutil.ParseResponse(t, resp, &config); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}

				if config.ModelProvider != tt.request["model_provider"] {
					t.Errorf("Expected model_provider %v, got %s", tt.request["model_provider"], config.ModelProvider)
				}
				if config.ModelName != tt.request["model_name"] {
					t.Errorf("Expected model_name %v, got %s", tt.request["model_name"], config.ModelName)
				}
			}
		})
	}
}

func TestModelConfigHandlers_UpdateModelConfig(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testutil.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()

	err := client.Authenticate(ctx, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to authenticate: %v", err)
	}

	// Create a test profile
	profile, err := testutil.CreateTestProfile(ctx, tdb.Queries, client.UserID)
	if err != nil {
		t.Fatalf("Failed to create test profile: %v", err)
	}

	// Create a model config to update
	config := &db.ModelConfig{
		ProfileID:     profile.ID,
		ModelProvider: "openai",
		ModelName:     "gpt-4",
		APIKey:        "sk-old-key",
		IsDefault:     false,
	}
	if err := tdb.Queries.CreateModelConfig(ctx, config); err != nil {
		t.Fatalf("Failed to create model config: %v", err)
	}

	tests := []struct {
		name           string
		configID       string
		request        map[string]interface{}
		expectedStatus int
	}{
		{
			name:     "successful update",
			configID: config.ID,
			request: map[string]interface{}{
				"model_provider": "openai",
				"model_name":     "gpt-4",
				"api_key":        "sk-new-key",
			},
			expectedStatus: http.StatusOK,
		},
		{
			name:     "not found",
			configID: "00000000-0000-0000-0000-000000000000",
			request: map[string]interface{}{
				"model_provider": "openai",
				"model_name":     "gpt-4",
			},
			expectedStatus: http.StatusNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Put("/api/v1/profiles/"+profile.ID+"/models/"+tt.configID, tt.request)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			testutil.AssertStatus(t, resp, tt.expectedStatus)

			if tt.expectedStatus == http.StatusOK {
				var updatedConfig db.ModelConfig
				if err := testutil.ParseResponse(t, resp, &updatedConfig); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}

				if tt.request["api_key"] != nil && updatedConfig.APIKey != tt.request["api_key"] {
					t.Errorf("Expected api_key %v, got %s", tt.request["api_key"], updatedConfig.APIKey)
				}
			}
		})
	}
}

func TestModelConfigHandlers_DeleteModelConfig(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testutil.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()

	err := client.Authenticate(ctx, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to authenticate: %v", err)
	}

	// Create a test profile
	profile, err := testutil.CreateTestProfile(ctx, tdb.Queries, client.UserID)
	if err != nil {
		t.Fatalf("Failed to create test profile: %v", err)
	}

	// Create a model config to delete
	config := &db.ModelConfig{
		ProfileID:     profile.ID,
		ModelProvider: "openai",
		ModelName:     "gpt-4",
		APIKey:        "sk-test-key",
		IsDefault:     false,
	}
	if err := tdb.Queries.CreateModelConfig(ctx, config); err != nil {
		t.Fatalf("Failed to create model config: %v", err)
	}

	tests := []struct {
		name           string
		configID       string
		expectedStatus int
	}{
		{
			name:           "successful delete",
			configID:       config.ID,
			expectedStatus: http.StatusNoContent,
		},
		{
			name:           "not found",
			configID:       "00000000-0000-0000-0000-000000000000",
			expectedStatus: http.StatusNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Delete("/api/v1/profiles/" + profile.ID + "/models/" + tt.configID)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			testutil.AssertStatus(t, resp, tt.expectedStatus)

			// Verify deletion
			if tt.expectedStatus == http.StatusNoContent {
				_, err := tdb.Queries.GetModelConfig(ctx, tt.configID, false)
				if err == nil {
					t.Error("Model config should have been deleted")
				}
			}
		})
	}
}








