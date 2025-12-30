package testing

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/auth"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
)

// TestClient provides HTTP client for testing with authentication
type TestClient struct {
	Server   *httptest.Server
	Router   *mux.Router
	Queries  *db.Queries
	Token    string
	UserID   string
	Username string
	IsAdmin  bool
}

// ServerSetupFunc is a function type for setting up a test server
// This allows handlers package to provide the setup without creating import cycles
type ServerSetupFunc func(*db.Queries) *httptest.Server

// DefaultServerSetup is set by handlers package to avoid import cycle
var DefaultServerSetup ServerSetupFunc

// NewTestClient creates a new test HTTP client
func NewTestClient(t *testing.T, queries *db.Queries) *TestClient {
	t.Helper()

	var server *httptest.Server
	if DefaultServerSetup != nil {
		server = DefaultServerSetup(queries)
	} else {
		// Fallback: create minimal server if setup function not provided
		router := mux.NewRouter()
		router.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{"status": "ok"})
		}).Methods("GET")
		server = httptest.NewServer(router)
	}

	return &TestClient{
		Server:  server,
		Router:  nil,
		Queries: queries,
	}
}

// Authenticate creates a test user and authenticates
func (tc *TestClient) Authenticate(ctx context.Context, username, password string) error {
	// Create user
	user, err := CreateTestUser(ctx, tc.Queries, username, password)
	if err != nil {
		return err
	}

	// Generate token
	token, err := auth.GenerateToken(user.ID, user.Username, user.IsAdmin)
	if err != nil {
		return err
	}

	tc.Token = token
	tc.UserID = user.ID
	tc.Username = user.Username
	tc.IsAdmin = user.IsAdmin

	return nil
}

// AuthenticateAsAdmin creates an admin user and authenticates
func (tc *TestClient) AuthenticateAsAdmin(ctx context.Context, username, password string) error {
	// Create admin user
	user, err := CreateTestAdmin(ctx, tc.Queries, username, password)
	if err != nil {
		return err
	}

	// Generate token
	token, err := auth.GenerateToken(user.ID, user.Username, user.IsAdmin)
	if err != nil {
		return err
	}

	tc.Token = token
	tc.UserID = user.ID
	tc.Username = user.Username
	tc.IsAdmin = user.IsAdmin

	return nil
}

// Do performs an HTTP request
func (tc *TestClient) Do(method, path string, body interface{}) (*http.Response, error) {
	var reqBody io.Reader
	if body != nil {
		jsonData, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reqBody = bytes.NewBuffer(jsonData)
	}

	req, err := http.NewRequest(method, tc.Server.URL+path, reqBody)
	if err != nil {
		return nil, err
	}

	if reqBody != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	if tc.Token != "" {
		req.Header.Set("Authorization", "Bearer "+tc.Token)
	}

	return http.DefaultClient.Do(req)
}

// Get performs a GET request
func (tc *TestClient) Get(path string) (*http.Response, error) {
	return tc.Do("GET", path, nil)
}

// Post performs a POST request
func (tc *TestClient) Post(path string, body interface{}) (*http.Response, error) {
	return tc.Do("POST", path, body)
}

// Put performs a PUT request
func (tc *TestClient) Put(path string, body interface{}) (*http.Response, error) {
	return tc.Do("PUT", path, body)
}

// Delete performs a DELETE request
func (tc *TestClient) Delete(path string) (*http.Response, error) {
	return tc.Do("DELETE", path, nil)
}

// ParseResponse parses JSON response
func ParseResponse(t *testing.T, resp *http.Response, v interface{}) error {
	t.Helper()

	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	return json.NewDecoder(resp.Body).Decode(v)
}

// AssertStatus asserts response status code
func AssertStatus(t *testing.T, resp *http.Response, expected int) {
	t.Helper()

	if resp.StatusCode != expected {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("Expected status %d, got %d. Body: %s", expected, resp.StatusCode, string(body))
	}
}
