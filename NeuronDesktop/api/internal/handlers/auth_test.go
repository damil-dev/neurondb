package handlers

import (
	"context"
	"net/http"
	"testing"

	testutil "github.com/neurondb/NeuronDesktop/api/internal/testing"
)

func TestAuthHandlers_Register(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testutil.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()

	tests := []struct {
		name           string
		request        map[string]interface{}
		expectedStatus int
		checkResponse  func(*testing.T, *http.Response)
	}{
		{
			name: "successful registration",
			request: map[string]interface{}{
				"username": "testuser",
				"password": "password123",
			},
			expectedStatus: http.StatusOK,
			checkResponse: func(t *testing.T, resp *http.Response) {
				var authResp map[string]interface{}
				if err := testutil.ParseResponse(t, resp, &authResp); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
				if authResp["token"] == nil {
					t.Error("Expected token in response")
				}
				if authResp["user_id"] == nil {
					t.Error("Expected user_id in response")
				}
			},
		},
		{
			name: "missing username",
			request: map[string]interface{}{
				"password": "password123",
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "missing password",
			request: map[string]interface{}{
				"username": "testuser",
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "password too short",
			request: map[string]interface{}{
				"username": "testuser",
				"password": "12345",
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "duplicate username",
			request: map[string]interface{}{
				"username": "testuser",
				"password": "password123",
			},
			expectedStatus: http.StatusConflict,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create user first for duplicate test
			if tt.name == "duplicate username" {
				_, err := testutil.CreateTestUser(ctx, tdb.Queries, "testuser", "password123")
				if err != nil {
					t.Fatalf("Failed to create test user: %v", err)
				}
			}

			resp, err := client.Post("/api/v1/auth/register", tt.request)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			testutil.AssertStatus(t, resp, tt.expectedStatus)

			if tt.checkResponse != nil {
				tt.checkResponse(t, resp)
			}
		})
	}
}

func TestAuthHandlers_Login(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testutil.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()

	// Create test user
	user, err := testutil.CreateTestUser(ctx, tdb.Queries, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to create test user: %v", err)
	}

	tests := []struct {
		name           string
		request        map[string]interface{}
		expectedStatus int
		checkResponse  func(*testing.T, *http.Response)
	}{
		{
			name: "successful login",
			request: map[string]interface{}{
				"username": "testuser",
				"password": "password123",
			},
			expectedStatus: http.StatusOK,
			checkResponse: func(t *testing.T, resp *http.Response) {
				var authResp map[string]interface{}
				if err := testutil.ParseResponse(t, resp, &authResp); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
				if authResp["token"] == nil {
					t.Error("Expected token in response")
				}
				if authResp["user_id"] != user.ID {
					t.Errorf("Expected user_id %s, got %v", user.ID, authResp["user_id"])
				}
			},
		},
		{
			name: "invalid username",
			request: map[string]interface{}{
				"username": "nonexistent",
				"password": "password123",
			},
			expectedStatus: http.StatusUnauthorized,
		},
		{
			name: "invalid password",
			request: map[string]interface{}{
				"username": "testuser",
				"password": "wrongpassword",
			},
			expectedStatus: http.StatusUnauthorized,
		},
		{
			name: "missing username",
			request: map[string]interface{}{
				"password": "password123",
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "missing password",
			request: map[string]interface{}{
				"username": "testuser",
			},
			expectedStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Post("/api/v1/auth/login", tt.request)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			testutil.AssertStatus(t, resp, tt.expectedStatus)

			if tt.checkResponse != nil {
				tt.checkResponse(t, resp)
			}
		})
	}
}

func TestAuthHandlers_GetCurrentUser(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testutil.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()

	// Test without authentication
	t.Run("unauthorized", func(t *testing.T) {
		resp, err := client.Get("/api/v1/auth/me")
		if err != nil {
			t.Fatalf("Request failed: %v", err)
		}
		defer resp.Body.Close()

		testutil.AssertStatus(t, resp, http.StatusUnauthorized)
	})

	// Test with authentication
	t.Run("authenticated", func(t *testing.T) {
		err := client.Authenticate(ctx, "testuser", "password123")
		if err != nil {
			t.Fatalf("Failed to authenticate: %v", err)
		}

		resp, err := client.Get("/api/v1/auth/me")
		if err != nil {
			t.Fatalf("Request failed: %v", err)
		}
		defer resp.Body.Close()

		testutil.AssertStatus(t, resp, http.StatusOK)

		var userResp map[string]interface{}
		if err := testutil.ParseResponse(t, resp, &userResp); err != nil {
			t.Fatalf("Failed to parse response: %v", err)
		}

		if userResp["user_id"] != client.UserID {
			t.Errorf("Expected user_id %s, got %v", client.UserID, userResp["user_id"])
		}
		if userResp["username"] != client.Username {
			t.Errorf("Expected username %s, got %v", client.Username, userResp["username"])
		}
	})
}

func TestAuthHandlers_Logout(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testutil.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()

	err := client.Authenticate(ctx, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to authenticate: %v", err)
	}

	resp, err := client.Post("/api/v1/auth/logout", nil)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	testutil.AssertStatus(t, resp, http.StatusOK)

	var logoutResp map[string]interface{}
	if err := testutil.ParseResponse(t, resp, &logoutResp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if logoutResp["message"] != "logged out" {
		t.Error("Expected logout message")
	}
}
