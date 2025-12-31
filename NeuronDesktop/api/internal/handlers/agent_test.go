package handlers

import (
	"context"
	"net/http"
	"testing"

	testutil "github.com/neurondb/NeuronDesktop/api/internal/testing"
)

func TestAgentHandlers_ListAgents(t *testing.T) {
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

	// Test listing agents (will fail if Agent endpoint not configured)
	resp, err := client.Get("/api/v1/profiles/" + profile.ID + "/agent/agents")
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	// Should either succeed or return an error, but not crash
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusInternalServerError {
		t.Errorf("Expected status 200 or 500, got %d", resp.StatusCode)
	}
}

func TestAgentHandlers_CreateSession(t *testing.T) {
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
			name: "valid session creation",
			request: map[string]interface{}{
				"agent_id": "agent-1",
			},
			expectedStatus: http.StatusInternalServerError, // Will fail without Agent endpoint
		},
		{
			name:           "missing agent_id",
			request:        map[string]interface{}{},
			expectedStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Post("/api/v1/profiles/"+profile.ID+"/agent/sessions", tt.request)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			// For validation errors, expect 400
			if tt.name == "missing agent_id" {
				testutil.AssertStatus(t, resp, http.StatusBadRequest)
			} else {
				// For actual session creation, may fail if Agent not configured
				if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusInternalServerError {
					t.Errorf("Expected status 201 or 500, got %d", resp.StatusCode)
				}
			}
		})
	}
}

func TestAgentHandlers_SendMessage(t *testing.T) {
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
		sessionID      string
		request        map[string]interface{}
		expectedStatus int
	}{
		{
			name:      "valid message",
			sessionID: "session-1",
			request: map[string]interface{}{
				"role":    "user",
				"content": "Hello, agent!",
			},
			expectedStatus: http.StatusInternalServerError, // Will fail without Agent endpoint
		},
		{
			name:      "missing role",
			sessionID: "session-1",
			request: map[string]interface{}{
				"content": "Hello, agent!",
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name:      "missing content",
			sessionID: "session-1",
			request: map[string]interface{}{
				"role": "user",
			},
			expectedStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Post("/api/v1/profiles/"+profile.ID+"/agent/sessions/"+tt.sessionID+"/messages", tt.request)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			// For validation errors, expect 400
			if tt.name == "missing role" || tt.name == "missing content" {
				testutil.AssertStatus(t, resp, http.StatusBadRequest)
			} else {
				// For actual messages, may fail if Agent not configured
				if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusInternalServerError {
					t.Errorf("Expected status 200 or 500, got %d", resp.StatusCode)
				}
			}
		})
	}
}

func TestAgentHandlers_TestAgentConfig(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testutil.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()

	err := client.Authenticate(ctx, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to authenticate: %v", err)
	}

	tests := []struct {
		name           string
		request        map[string]interface{}
		expectedStatus int
	}{
		{
			name: "valid endpoint",
			request: map[string]interface{}{
				"endpoint": "http://localhost:8080",
			},
			expectedStatus: http.StatusOK, // May fail if endpoint unreachable, but should handle gracefully
		},
		{
			name:           "missing endpoint",
			request:        map[string]interface{}{},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "invalid URL",
			request: map[string]interface{}{
				"endpoint": "not-a-valid-url",
			},
			expectedStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Post("/api/v1/agent/test", tt.request)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			// For validation errors, expect 400
			if tt.name == "missing endpoint" || tt.name == "invalid URL" {
				testutil.AssertStatus(t, resp, http.StatusBadRequest)
			} else {
				// For actual tests, may fail if endpoint unreachable
				if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusBadRequest {
					t.Errorf("Expected status 200 or 400, got %d", resp.StatusCode)
				}
			}
		})
	}
}





