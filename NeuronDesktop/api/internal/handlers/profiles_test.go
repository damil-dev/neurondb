package handlers

import (
	"context"
	"net/http"
	"testing"

	testutil "github.com/neurondb/NeuronDesktop/api/internal/testing"
)

func TestProfileHandlers_ListProfiles(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testutil.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()

	// Test without authentication
	t.Run("unauthorized", func(t *testing.T) {
		resp, err := client.Get("/api/v1/profiles")
		if err != nil {
			t.Fatalf("Request failed: %v", err)
		}
		defer resp.Body.Close()

		testutil.AssertStatus(t, resp, http.StatusUnauthorized)
	})

	// Test with authentication but no profiles
	t.Run("no profiles", func(t *testing.T) {
		err := client.Authenticate(ctx, "testuser", "password123")
		if err != nil {
			t.Fatalf("Failed to authenticate: %v", err)
		}

		resp, err := client.Get("/api/v1/profiles")
		if err != nil {
			t.Fatalf("Request failed: %v", err)
		}
		defer resp.Body.Close()

		testutil.AssertStatus(t, resp, http.StatusOK)

		var profiles []interface{}
		if err := testutil.ParseResponse(t, resp, &profiles); err != nil {
			t.Fatalf("Failed to parse response: %v", err)
		}

		// Should auto-create default profile
		if len(profiles) == 0 {
			t.Error("Expected at least one profile (auto-created default)")
		}
	})

	// Test with existing profiles
	t.Run("with profiles", func(t *testing.T) {
		err := client.Authenticate(ctx, "testuser2", "password123")
		if err != nil {
			t.Fatalf("Failed to authenticate: %v", err)
		}

		// Create a test profile
		profile, err := testutil.CreateTestProfile(ctx, tdb.Queries, client.UserID)
		if err != nil {
			t.Fatalf("Failed to create test profile: %v", err)
		}

		resp, err := client.Get("/api/v1/profiles")
		if err != nil {
			t.Fatalf("Request failed: %v", err)
		}
		defer resp.Body.Close()

		testutil.AssertStatus(t, resp, http.StatusOK)

		var profiles []map[string]interface{}
		if err := testutil.ParseResponse(t, resp, &profiles); err != nil {
			t.Fatalf("Failed to parse response: %v", err)
		}

		found := false
		for _, p := range profiles {
			if p["id"] == profile.ID {
				found = true
				break
			}
		}

		if !found {
			t.Error("Expected to find created profile in list")
		}
	})
}

func TestProfileHandlers_GetProfile(t *testing.T) {
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
		profileID      string
		expectedStatus int
	}{
		{
			name:           "successful get",
			profileID:      profile.ID,
			expectedStatus: http.StatusOK,
		},
		{
			name:           "not found",
			profileID:      "00000000-0000-0000-0000-000000000000",
			expectedStatus: http.StatusNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Get("/api/v1/profiles/" + tt.profileID)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			testutil.AssertStatus(t, resp, tt.expectedStatus)

			if tt.expectedStatus == http.StatusOK {
				var profileResp map[string]interface{}
				if err := testutil.ParseResponse(t, resp, &profileResp); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}

				if profileResp["id"] != profile.ID {
					t.Errorf("Expected profile ID %s, got %v", profile.ID, profileResp["id"])
				}
			}
		})
	}
}

func TestProfileHandlers_UpdateProfile(t *testing.T) {
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
		profileID      string
		request        map[string]interface{}
		expectedStatus int
	}{
		{
			name:      "successful update",
			profileID: profile.ID,
			request: map[string]interface{}{
				"name":         "Updated Profile Name",
				"neurondb_dsn": "host=localhost port=5432 user=neurondb dbname=neurondb",
				"mcp_config": map[string]interface{}{
					"command": "echo",
					"args":    []string{"updated"},
				},
			},
			expectedStatus: http.StatusOK,
		},
		{
			name:      "not found",
			profileID: "00000000-0000-0000-0000-000000000000",
			request: map[string]interface{}{
				"name": "Updated Profile Name",
			},
			expectedStatus: http.StatusNotFound,
		},
		{
			name:      "invalid request",
			profileID: profile.ID,
			request: map[string]interface{}{
				"name": "", // Empty name should fail validation
			},
			expectedStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Put("/api/v1/profiles/"+tt.profileID, tt.request)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			testutil.AssertStatus(t, resp, tt.expectedStatus)

			if tt.expectedStatus == http.StatusOK {
				var profileResp map[string]interface{}
				if err := testutil.ParseResponse(t, resp, &profileResp); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}

				if profileResp["name"] != tt.request["name"] {
					t.Errorf("Expected name %v, got %v", tt.request["name"], profileResp["name"])
				}
			}
		})
	}
}

func TestProfileHandlers_DeleteProfile(t *testing.T) {
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
		profileID      string
		expectedStatus int
	}{
		{
			name:           "successful delete",
			profileID:      profile.ID,
			expectedStatus: http.StatusNoContent,
		},
		{
			name:           "not found",
			profileID:      "00000000-0000-0000-0000-000000000000",
			expectedStatus: http.StatusNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Delete("/api/v1/profiles/" + tt.profileID)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			testutil.AssertStatus(t, resp, tt.expectedStatus)

			// Verify deletion
			if tt.expectedStatus == http.StatusNoContent {
				_, err := tdb.Queries.GetProfile(ctx, tt.profileID)
				if err == nil {
					t.Error("Profile should have been deleted")
				}
			}
		})
	}
}

func TestProfileHandlers_ProfileIsolation(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testutil.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()

	// Create two users
	err := client.Authenticate(ctx, "user1", "password123")
	if err != nil {
		t.Fatalf("Failed to authenticate: %v", err)
	}
	user1ID := client.UserID

	profile1, err := testutil.CreateTestProfile(ctx, tdb.Queries, user1ID)
	if err != nil {
		t.Fatalf("Failed to create profile: %v", err)
	}

	// Switch to user2
	err = client.Authenticate(ctx, "user2", "password123")
	if err != nil {
		t.Fatalf("Failed to authenticate: %v", err)
	}

	// User2 should not be able to access user1's profile
	resp, err := client.Get("/api/v1/profiles/" + profile1.ID)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	testutil.AssertStatus(t, resp, http.StatusForbidden)
}
