package e2e

import (
	"context"
	"net/http"
	"testing"

	"github.com/neurondb/NeuronDesktop/api/internal/testing"
)

// TestProfileWorkflow_CRUD tests complete profile CRUD operations
func TestProfileWorkflow_CRUD(t *testing.T) {
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

	// Step 1: List profiles (should auto-create default)
	t.Run("list_profiles", func(t *testing.T) {
		resp, err := client.Get("/api/v1/profiles")
		if err != nil {
			t.Fatalf("List profiles request failed: %v", err)
		}
		defer resp.Body.Close()

		testing.AssertStatus(t, resp, http.StatusOK)

		var profiles []map[string]interface{}
		if err := testing.ParseResponse(t, resp, &profiles); err != nil {
			t.Fatalf("Failed to parse profiles response: %v", err)
		}

		if len(profiles) == 0 {
			t.Error("Expected at least one profile (auto-created default)")
		}
	})

	// Step 2: Get a specific profile
	t.Run("get_profile", func(t *testing.T) {
		// First, get the list to find a profile ID
		resp, err := client.Get("/api/v1/profiles")
		if err != nil {
			t.Fatalf("List profiles request failed: %v", err)
		}
		defer resp.Body.Close()

		var profiles []map[string]interface{}
		if err := testing.ParseResponse(t, resp, &profiles); err != nil {
			t.Fatalf("Failed to parse profiles response: %v", err)
		}

		if len(profiles) == 0 {
			t.Fatal("No profiles available for get test")
		}

		profileID := profiles[0]["id"].(string)

		// Get the profile
		resp2, err := client.Get("/api/v1/profiles/" + profileID)
		if err != nil {
			t.Fatalf("Get profile request failed: %v", err)
		}
		defer resp2.Body.Close()

		testing.AssertStatus(t, resp2, http.StatusOK)

		var profile map[string]interface{}
		if err := testing.ParseResponse(t, resp2, &profile); err != nil {
			t.Fatalf("Failed to parse profile response: %v", err)
		}

		if profile["id"] != profileID {
			t.Errorf("Expected profile ID %s, got %v", profileID, profile["id"])
		}
	})

	// Step 3: Update profile
	t.Run("update_profile", func(t *testing.T) {
		// Get a profile to update
		resp, err := client.Get("/api/v1/profiles")
		if err != nil {
			t.Fatalf("List profiles request failed: %v", err)
		}
		defer resp.Body.Close()

		var profiles []map[string]interface{}
		if err := testing.ParseResponse(t, resp, &profiles); err != nil {
			t.Fatalf("Failed to parse profiles response: %v", err)
		}

		if len(profiles) == 0 {
			t.Fatal("No profiles available for update test")
		}

		profileID := profiles[0]["id"].(string)

		// Update the profile
		updateReq := map[string]interface{}{
			"name": "Updated Profile Name",
			"neurondb_dsn": "host=localhost port=5432 user=neurondb dbname=neurondb",
			"mcp_config": map[string]interface{}{
				"command": "echo",
				"args":    []string{"updated"},
			},
		}

		resp2, err := client.Put("/api/v1/profiles/"+profileID, updateReq)
		if err != nil {
			t.Fatalf("Update profile request failed: %v", err)
		}
		defer resp2.Body.Close()

		testing.AssertStatus(t, resp2, http.StatusOK)

		var updatedProfile map[string]interface{}
		if err := testing.ParseResponse(t, resp2, &updatedProfile); err != nil {
			t.Fatalf("Failed to parse updated profile response: %v", err)
		}

		if updatedProfile["name"] != "Updated Profile Name" {
			t.Errorf("Expected profile name 'Updated Profile Name', got %v", updatedProfile["name"])
		}
	})

	// Step 4: Delete profile (create a new one first to avoid deleting default)
	t.Run("delete_profile", func(t *testing.T) {
		// Create a test profile to delete
		profile, err := testing.CreateTestProfile(ctx, tdb.Queries, client.UserID)
		if err != nil {
			t.Fatalf("Failed to create test profile: %v", err)
		}

		// Delete the profile
		resp, err := client.Delete("/api/v1/profiles/" + profile.ID)
		if err != nil {
			t.Fatalf("Delete profile request failed: %v", err)
		}
		defer resp.Body.Close()

		testing.AssertStatus(t, resp, http.StatusNoContent)

		// Verify deletion
		resp2, err := client.Get("/api/v1/profiles/" + profile.ID)
		if err != nil {
			t.Fatalf("Get deleted profile request failed: %v", err)
		}
		defer resp2.Body.Close()

		testing.AssertStatus(t, resp2, http.StatusNotFound)
	})
}

// TestProfileWorkflow_Isolation tests that users can only access their own profiles
func TestProfileWorkflow_Isolation(t *testing.T) {
	tdb := testing.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testing.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()

	// Create user1 and profile
	err := client.Authenticate(ctx, "user1", "password123")
	if err != nil {
		t.Fatalf("Failed to authenticate user1: %v", err)
	}
	user1ID := client.UserID

	profile1, err := testing.CreateTestProfile(ctx, tdb.Queries, user1ID)
	if err != nil {
		t.Fatalf("Failed to create profile1: %v", err)
	}

	// Switch to user2
	err = client.Authenticate(ctx, "user2", "password123")
	if err != nil {
		t.Fatalf("Failed to authenticate user2: %v", err)
	}

	// User2 should not be able to access user1's profile
	resp, err := client.Get("/api/v1/profiles/" + profile1.ID)
	if err != nil {
		t.Fatalf("Get profile request failed: %v", err)
	}
	defer resp.Body.Close()

	testing.AssertStatus(t, resp, http.StatusForbidden)

	// User2 should not be able to update user1's profile
	updateReq := map[string]interface{}{
		"name": "Hacked Profile",
	}
	resp2, err := client.Put("/api/v1/profiles/"+profile1.ID, updateReq)
	if err != nil {
		t.Fatalf("Update profile request failed: %v", err)
	}
	defer resp2.Body.Close()

	testing.AssertStatus(t, resp2, http.StatusForbidden)

	// User2 should not be able to delete user1's profile
	resp3, err := client.Delete("/api/v1/profiles/" + profile1.ID)
	if err != nil {
		t.Fatalf("Delete profile request failed: %v", err)
	}
	defer resp3.Body.Close()

	testing.AssertStatus(t, resp3, http.StatusForbidden)
}








