package handlers

import (
	"context"
	"net/http"
	"testing"

	testutil "github.com/neurondb/NeuronDesktop/api/internal/testing"
)

func TestMetricsHandlers_GetMetrics(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testutil.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()

	err := client.Authenticate(ctx, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to authenticate: %v", err)
	}

	// Test getting metrics
	resp, err := client.Get("/api/v1/metrics")
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	testutil.AssertStatus(t, resp, http.StatusOK)

	var metrics map[string]interface{}
	if err := testutil.ParseResponse(t, resp, &metrics); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	// Check for expected metric fields
	expectedFields := []string{"requests", "response_time", "connections", "endpoints", "errors"}
	for _, field := range expectedFields {
		if metrics[field] == nil {
			t.Errorf("Expected metrics to contain field: %s", field)
		}
	}
}

func TestMetricsHandlers_ResetMetrics(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testutil.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()

	err := client.Authenticate(ctx, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to authenticate: %v", err)
	}

	// Test resetting metrics
	resp, err := client.Post("/api/v1/metrics/reset", nil)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	testutil.AssertStatus(t, resp, http.StatusOK)

	var result map[string]interface{}
	if err := testutil.ParseResponse(t, resp, &result); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if result["message"] != "Metrics reset" {
		t.Error("Expected reset confirmation message")
	}

	// Verify metrics were reset by getting them again
	resp2, err := client.Get("/api/v1/metrics")
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp2.Body.Close()

	var metrics map[string]interface{}
	if err := testutil.ParseResponse(t, resp2, &metrics); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	// Check that request counts are reset
	if requests, ok := metrics["requests"].(map[string]interface{}); ok {
		if total, ok := requests["total"].(float64); ok && total != 0 {
			// Allow some requests from our test calls
			if total > 10 {
				t.Errorf("Expected metrics to be mostly reset, got total: %v", total)
			}
		}
	}
}
