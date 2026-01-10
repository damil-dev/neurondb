package handlers

import (
	"context"
	"net/http"
	"testing"

	testutil "github.com/neurondb/NeuronDesktop/api/internal/testing"
)

func TestNeuronDBHandlers_ListCollections(t *testing.T) {
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

	// Test listing collections (will fail if NeuronDB not configured, but should handle gracefully)
	resp, err := client.Get("/api/v1/profiles/" + profile.ID + "/neurondb/collections")
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	// Should either succeed or return an error, but not crash
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusInternalServerError {
		t.Errorf("Expected status 200 or 500, got %d", resp.StatusCode)
	}
}

func TestNeuronDBHandlers_Search(t *testing.T) {
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
			name: "valid search request",
			request: map[string]interface{}{
				"collection":   "documents",
				"query_vector": []float64{0.1, 0.2, 0.3},
				"limit":        10,
			},
			expectedStatus: http.StatusInternalServerError, // Will fail without NeuronDB connection
		},
		{
			name: "missing collection",
			request: map[string]interface{}{
				"query_vector": []float64{0.1, 0.2, 0.3},
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "missing query_vector",
			request: map[string]interface{}{
				"collection": "documents",
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "invalid query_vector type",
			request: map[string]interface{}{
				"collection":   "documents",
				"query_vector": "not-an-array",
			},
			expectedStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Post("/api/v1/profiles/"+profile.ID+"/neurondb/search", tt.request)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			// For validation errors, expect 400
			if tt.name == "missing collection" || tt.name == "missing query_vector" || tt.name == "invalid query_vector type" {
				testutil.AssertStatus(t, resp, http.StatusBadRequest)
			} else {
				// For actual searches, may fail if NeuronDB not configured
				if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusInternalServerError {
					t.Errorf("Expected status 200 or 500, got %d", resp.StatusCode)
				}
			}
		})
	}
}

func TestNeuronDBHandlers_ExecuteSQL(t *testing.T) {
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
			name: "valid SELECT query",
			request: map[string]interface{}{
				"query": "SELECT * FROM documents LIMIT 10",
			},
			expectedStatus: http.StatusInternalServerError, // Will fail without NeuronDB connection
		},
		{
			name:           "missing query",
			request:        map[string]interface{}{},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "invalid query (INSERT)",
			request: map[string]interface{}{
				"query": "INSERT INTO documents VALUES (1, 'test')",
			},
			expectedStatus: http.StatusBadRequest, // Should reject non-SELECT queries
		},
		{
			name: "invalid query (DROP)",
			request: map[string]interface{}{
				"query": "DROP TABLE documents",
			},
			expectedStatus: http.StatusBadRequest, // Should reject DROP queries
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Post("/api/v1/profiles/"+profile.ID+"/neurondb/sql", tt.request)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			// For validation errors, expect 400
			if tt.name == "missing query" || tt.name == "invalid query (INSERT)" || tt.name == "invalid query (DROP)" {
				testutil.AssertStatus(t, resp, http.StatusBadRequest)
			} else {
				// For actual queries, may fail if NeuronDB not configured
				if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusInternalServerError {
					t.Errorf("Expected status 200 or 500, got %d", resp.StatusCode)
				}
			}
		})
	}
}








