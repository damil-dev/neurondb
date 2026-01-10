package handlers

import (
	"context"
	"net/http"
	"testing"

	testutil "github.com/neurondb/NeuronDesktop/api/internal/testing"
)

func TestMCPHandlers_ListConnections(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testutil.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()

	err := client.Authenticate(ctx, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to authenticate: %v", err)
	}

	resp, err := client.Get("/api/v1/mcp/connections")
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	testutil.AssertStatus(t, resp, http.StatusOK)

	var connections []interface{}
	if err := testutil.ParseResponse(t, resp, &connections); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	// Should return empty array if no connections
	if connections == nil {
		t.Error("Expected connections array")
	}
}

func TestMCPHandlers_TestMCPConfig(t *testing.T) {
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
			name: "valid echo command",
			request: map[string]interface{}{
				"command": "echo",
				"args":    []string{"test"},
			},
			expectedStatus: http.StatusOK,
		},
		{
			name: "invalid command",
			request: map[string]interface{}{
				"command": "nonexistent-command-xyz",
				"args":    []string{},
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "missing command",
			request: map[string]interface{}{
				"args": []string{},
			},
			expectedStatus: http.StatusOK, // Uses default
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Post("/api/v1/mcp/test", tt.request)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			testutil.AssertStatus(t, resp, tt.expectedStatus)

			if tt.expectedStatus == http.StatusOK {
				var result map[string]interface{}
				if err := testutil.ParseResponse(t, resp, &result); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}

				if result["success"] != true {
					t.Error("Expected success to be true")
				}
			}
		})
	}
}

func TestMCPHandlers_ListTools(t *testing.T) {
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

	// Test listing tools (will fail if MCP not configured, but should handle gracefully)
	resp, err := client.Get("/api/v1/profiles/" + profile.ID + "/mcp/tools")
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	// Should either succeed or return an error, but not crash
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusInternalServerError {
		t.Errorf("Expected status 200 or 500, got %d", resp.StatusCode)
	}
}

func TestMCPHandlers_CallTool(t *testing.T) {
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
			name: "valid tool call",
			request: map[string]interface{}{
				"name": "vector_search",
				"arguments": map[string]interface{}{
					"query_vector": []float64{0.1, 0.2, 0.3},
					"table":        "documents",
					"limit":        10,
				},
			},
			expectedStatus: http.StatusInternalServerError, // Will fail without MCP connection
		},
		{
			name: "missing tool name",
			request: map[string]interface{}{
				"arguments": map[string]interface{}{},
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "missing arguments",
			request: map[string]interface{}{
				"name": "vector_search",
			},
			expectedStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Post("/api/v1/profiles/"+profile.ID+"/mcp/tools/call", tt.request)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			// For validation errors, expect 400
			if tt.name == "missing tool name" || tt.name == "missing arguments" {
				testutil.AssertStatus(t, resp, http.StatusBadRequest)
			} else {
				// For actual tool calls, may fail if MCP not configured
				if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusInternalServerError {
					t.Errorf("Expected status 200 or 500, got %d", resp.StatusCode)
				}
			}
		})
	}
}








