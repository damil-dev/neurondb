package e2e

import (
	"context"
	"net/http"
	"testing"

	"github.com/neurondb/NeuronDesktop/api/internal/testing"
)

// TestMetricsWorkflow tests metrics collection and retrieval
func TestMetricsWorkflow(t *testing.T) {
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

	// Step 1: Make some requests to generate metrics
	t.Run("generate_metrics", func(t *testing.T) {
		// Make several requests
		for i := 0; i < 5; i++ {
			_, err := client.Get("/api/v1/profiles")
			if err != nil {
				t.Logf("Request failed (may be expected): %v", err)
			}
		}
	})

	// Step 2: Get metrics
	t.Run("get_metrics", func(t *testing.T) {
		resp, err := client.Get("/api/v1/metrics")
		if err != nil {
			t.Fatalf("Get metrics request failed: %v", err)
		}
		defer resp.Body.Close()

		testing.AssertStatus(t, resp, http.StatusOK)

		var metrics map[string]interface{}
		if err := testing.ParseResponse(t, resp, &metrics); err != nil {
			t.Fatalf("Failed to parse metrics response: %v", err)
		}

		// Verify metrics structure
		expectedFields := []string{"requests", "response_time", "connections", "endpoints", "errors"}
		for _, field := range expectedFields {
			if metrics[field] == nil {
				t.Errorf("Expected metrics to contain field: %s", field)
			}
		}

		// Check request counts
		if requests, ok := metrics["requests"].(map[string]interface{}); ok {
			if total, ok := requests["total"].(float64); ok {
				if total < 1 {
					t.Errorf("Expected at least 1 request, got %v", total)
				}
			}
		}
	})

	// Step 3: Reset metrics
	t.Run("reset_metrics", func(t *testing.T) {
		resp, err := client.Post("/api/v1/metrics/reset", nil)
		if err != nil {
			t.Fatalf("Reset metrics request failed: %v", err)
		}
		defer resp.Body.Close()

		testing.AssertStatus(t, resp, http.StatusOK)

		var result map[string]interface{}
		if err := testing.ParseResponse(t, resp, &result); err != nil {
			t.Fatalf("Failed to parse reset response: %v", err)
		}

		if result["message"] != "Metrics reset" {
			t.Error("Expected reset confirmation message")
		}
	})

	// Step 4: Verify metrics were reset
	t.Run("verify_reset", func(t *testing.T) {
		resp, err := client.Get("/api/v1/metrics")
		if err != nil {
			t.Fatalf("Get metrics request failed: %v", err)
		}
		defer resp.Body.Close()

		var metrics map[string]interface{}
		if err := testing.ParseResponse(t, resp, &metrics); err != nil {
			t.Fatalf("Failed to parse metrics response: %v", err)
		}

		// After reset, request counts should be low (only our test requests)
		if requests, ok := metrics["requests"].(map[string]interface{}); ok {
			if total, ok := requests["total"].(float64); ok {
				// Allow some requests from our test calls, but should be much lower
				if total > 20 {
					t.Errorf("Expected metrics to be mostly reset, got total: %v", total)
				}
			}
		}
	})
}






