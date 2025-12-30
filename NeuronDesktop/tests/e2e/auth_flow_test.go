package e2e

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/neurondb/NeuronDesktop/api/internal/testing"
)

// TestAuthFlow_RegisterAndLogin tests the complete authentication flow
func TestAuthFlow_RegisterAndLogin(t *testing.T) {
	tdb := testing.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testing.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()

	// Step 1: Register a new user
	t.Run("register", func(t *testing.T) {
		registerReq := map[string]interface{}{
			"username": "e2euser",
			"password": "password123",
		}

		resp, err := client.Post("/api/v1/auth/register", registerReq)
		if err != nil {
			t.Fatalf("Register request failed: %v", err)
		}
		defer resp.Body.Close()

		testing.AssertStatus(t, resp, http.StatusOK)

		var authResp map[string]interface{}
		if err := testing.ParseResponse(t, resp, &authResp); err != nil {
			t.Fatalf("Failed to parse register response: %v", err)
		}

		if authResp["token"] == nil {
			t.Fatal("Expected token in register response")
		}
		if authResp["user_id"] == nil {
			t.Fatal("Expected user_id in register response")
		}
	})

	// Step 2: Login with the registered user
	t.Run("login", func(t *testing.T) {
		loginReq := map[string]interface{}{
			"username": "e2euser",
			"password": "password123",
		}

		resp, err := client.Post("/api/v1/auth/login", loginReq)
		if err != nil {
			t.Fatalf("Login request failed: %v", err)
		}
		defer resp.Body.Close()

		testing.AssertStatus(t, resp, http.StatusOK)

		var authResp map[string]interface{}
		if err := testing.ParseResponse(t, resp, &authResp); err != nil {
			t.Fatalf("Failed to parse login response: %v", err)
		}

		if authResp["token"] == nil {
			t.Fatal("Expected token in login response")
		}

		// Store token for subsequent requests
		client.Token = authResp["token"].(string)
		client.UserID = authResp["user_id"].(string)
		client.Username = authResp["username"].(string)
	})

	// Step 3: Get current user info
	t.Run("get_current_user", func(t *testing.T) {
		resp, err := client.Get("/api/v1/auth/me")
		if err != nil {
			t.Fatalf("Get current user request failed: %v", err)
		}
		defer resp.Body.Close()

		testing.AssertStatus(t, resp, http.StatusOK)

		var userResp map[string]interface{}
		if err := testing.ParseResponse(t, resp, &userResp); err != nil {
			t.Fatalf("Failed to parse user response: %v", err)
		}

		if userResp["user_id"] != client.UserID {
			t.Errorf("Expected user_id %s, got %v", client.UserID, userResp["user_id"])
		}
		if userResp["username"] != client.Username {
			t.Errorf("Expected username %s, got %v", client.Username, userResp["username"])
		}
	})

	// Step 4: Logout
	t.Run("logout", func(t *testing.T) {
		resp, err := client.Post("/api/v1/auth/logout", nil)
		if err != nil {
			t.Fatalf("Logout request failed: %v", err)
		}
		defer resp.Body.Close()

		testing.AssertStatus(t, resp, http.StatusOK)
	})
}

// TestAuthFlow_InvalidCredentials tests authentication with invalid credentials
func TestAuthFlow_InvalidCredentials(t *testing.T) {
	tdb := testing.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testing.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()

	// Create a user
	_, err := testing.CreateTestUser(ctx, tdb.Queries, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to create test user: %v", err)
	}

	tests := []struct {
		name           string
		username       string
		password       string
		expectedStatus int
	}{
		{
			name:           "wrong password",
			username:       "testuser",
			password:       "wrongpassword",
			expectedStatus: http.StatusUnauthorized,
		},
		{
			name:           "wrong username",
			username:       "nonexistent",
			password:       "password123",
			expectedStatus: http.StatusUnauthorized,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loginReq := map[string]interface{}{
				"username": tt.username,
				"password": tt.password,
			}

			resp, err := client.Post("/api/v1/auth/login", loginReq)
			if err != nil {
				t.Fatalf("Login request failed: %v", err)
			}
			defer resp.Body.Close()

			testing.AssertStatus(t, resp, tt.expectedStatus)
		})
	}
}

// TestAuthFlow_ProtectedEndpoints tests that protected endpoints require authentication
func TestAuthFlow_ProtectedEndpoints(t *testing.T) {
	tdb := testing.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testing.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	// Test accessing protected endpoints without authentication
	protectedEndpoints := []struct {
		method string
		path   string
		body   interface{}
	}{
		{"GET", "/api/v1/profiles", nil},
		{"GET", "/api/v1/auth/me", nil},
		{"GET", "/api/v1/metrics", nil},
	}

	for _, endpoint := range protectedEndpoints {
		t.Run(endpoint.method+"_"+endpoint.path, func(t *testing.T) {
			var resp *http.Response
			var err error

			switch endpoint.method {
			case "GET":
				resp, err = client.Get(endpoint.path)
			case "POST":
				resp, err = client.Post(endpoint.path, endpoint.body)
			case "PUT":
				resp, err = client.Put(endpoint.path, endpoint.body)
			case "DELETE":
				resp, err = client.Delete(endpoint.path)
			}

			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			testing.AssertStatus(t, resp, http.StatusUnauthorized)
		})
	}
}





