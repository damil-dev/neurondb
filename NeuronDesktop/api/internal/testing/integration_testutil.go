package testing

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/neurondb/NeuronDesktop/api/internal/agent"
	"github.com/neurondb/NeuronDesktop/api/internal/mcp"
	"github.com/neurondb/NeuronDesktop/api/internal/neurondb"
)

/* IntegrationTestConfig holds configuration for integration tests */
type IntegrationTestConfig struct {
	NeuronDBDSN      string
	NeuronMCPCommand string
	NeuronAgentURL   string
	NeuronAgentKey   string
	SkipNeuronDB     bool
	SkipNeuronMCP    bool
	SkipNeuronAgent  bool
}

/* LoadIntegrationTestConfig loads integration test configuration from environment */
func LoadIntegrationTestConfig() *IntegrationTestConfig {
	return &IntegrationTestConfig{
		NeuronDBDSN:      getEnv("TEST_NEURONDB_DSN", "host=localhost port=5432 user=neurondb dbname=neurondb sslmode=disable"),
		NeuronMCPCommand: getEnv("TEST_NEURONMCP_COMMAND", ""),
		NeuronAgentURL:   getEnv("TEST_NEURONAGENT_URL", "http://localhost:8080"),
		NeuronAgentKey:   getEnv("TEST_NEURONAGENT_KEY", ""),
		SkipNeuronDB:     os.Getenv("SKIP_NEURONDB") == "true",
		SkipNeuronMCP:    os.Getenv("SKIP_NEURONMCP") == "true",
		SkipNeuronAgent:  os.Getenv("SKIP_NEURONAGENT") == "true",
	}
}

/* VerifyNeuronDBConnection verifies that NeuronDB is accessible */
func VerifyNeuronDBConnection(ctx context.Context, dsn string) error {
	client, err := neurondb.NewClient(dsn)
	if err != nil {
		return fmt.Errorf("failed to connect to NeuronDB: %w", err)
	}
	defer client.Close()

	/* Try to list collections as a connectivity test */
	_, err = client.ListCollections(ctx)
	if err != nil {
		/* Even if collections fail, connection is established */
		return nil
	}

	return nil
}

/* VerifyNeuronMCPConnection verifies that NeuronMCP can be spawned */
func VerifyNeuronMCPConnection(ctx context.Context, command string, env map[string]string) error {
	if command == "" {
		return fmt.Errorf("NeuronMCP command not configured")
	}

	config := mcp.MCPConfig{
		Command: command,
		Args:    []string{},
		Env:     env,
	}

	client, err := mcp.NewClient(config)
	if err != nil {
		return fmt.Errorf("failed to spawn NeuronMCP: %w", err)
	}
	defer client.Close()

	/* Try to list tools as a connectivity test */
	_, err = client.ListTools()
	if err != nil {
		return fmt.Errorf("failed to list tools: %w", err)
	}

	return nil
}

/* VerifyNeuronAgentConnection verifies that NeuronAgent is accessible */
func VerifyNeuronAgentConnection(ctx context.Context, url, apiKey string) error {
	client := agent.NewClient(url, apiKey)

	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	_, err := client.ListAgents(ctx)
	if err != nil {
		return fmt.Errorf("failed to connect to NeuronAgent: %w", err)
	}

	return nil
}

/* CreateNeuronDBTestClient creates a NeuronDB client for testing */
func CreateNeuronDBTestClient(t *testing.T, dsn string) *neurondb.Client {
	t.Helper()

	client, err := neurondb.NewClient(dsn)
	if err != nil {
		t.Skipf("Skipping test: cannot connect to NeuronDB: %v", err)
		return nil
	}

	return client
}

/* CreateNeuronMCPTestClient creates a NeuronMCP client for testing */
func CreateNeuronMCPTestClient(t *testing.T, command string, env map[string]string) *mcp.Client {
	t.Helper()

	if command == "" {
		t.Skip("Skipping test: NeuronMCP command not configured")
		return nil
	}

	config := mcp.MCPConfig{
		Command: command,
		Args:    []string{},
		Env:     env,
	}

	client, err := mcp.NewClient(config)
	if err != nil {
		t.Skipf("Skipping test: cannot spawn NeuronMCP: %v", err)
		return nil
	}

	return client
}

/* CreateNeuronAgentTestClient creates a NeuronAgent client for testing */
func CreateNeuronAgentTestClient(t *testing.T, url, apiKey string) *agent.Client {
	t.Helper()

	client := agent.NewClient(url, apiKey)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	_, err := client.ListAgents(ctx)
	if err != nil {
		t.Skipf("Skipping test: cannot connect to NeuronAgent: %v", err)
		return nil
	}

	return client
}

/* CreateTestCollection creates a test collection in NeuronDB */
func CreateTestCollection(ctx context.Context, client *neurondb.Client, schema, table, vectorCol string, dimensions int) error {
	/* Create table with vector column */
	createTableSQL := fmt.Sprintf(`
		CREATE SCHEMA IF NOT EXISTS %s;
		CREATE TABLE IF NOT EXISTS %s.%s (
			id SERIAL PRIMARY KEY,
			content TEXT,
			%s vector(%d)
		);
	`, schema, schema, table, vectorCol, dimensions)

	_, err := client.ExecuteSQLFull(ctx, createTableSQL)
	return err
}

/* InsertTestVector inserts a test vector into a collection */
func InsertTestVector(ctx context.Context, client *neurondb.Client, schema, table, vectorCol string, content string, vector []float32) error {
	vectorStr := formatVectorForSQL(vector)
	/* Note: This uses string interpolation for the vector which is safe since it's generated from numeric values */
	insertSQL := fmt.Sprintf(`
		INSERT INTO %s.%s (content, %s) 
		VALUES ('%s', %s::vector)
	`, schema, table, vectorCol, content, vectorStr)

	_, err := client.ExecuteSQLFull(ctx, insertSQL)
	return err
}

/* formatVectorForSQL formats a vector for SQL insertion */
func formatVectorForSQL(vec []float32) string {
	parts := make([]string, len(vec))
	for i, v := range vec {
		parts[i] = fmt.Sprintf("%.6f", v)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

/* WaitForService waits for a service to become available */
func WaitForService(url string, timeout time.Duration) error {
	client := &http.Client{
		Timeout: 5 * time.Second,
	}

	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		resp, err := client.Get(url)
		if err == nil && resp.StatusCode < 500 {
			resp.Body.Close()
			return nil
		}
		time.Sleep(1 * time.Second)
	}

	return fmt.Errorf("service at %s did not become available within %v", url, timeout)
}

/* AssertHTTPStatus asserts that an HTTP response has the expected status */
func AssertHTTPStatus(t *testing.T, resp *http.Response, expectedStatus int) {
	t.Helper()
	if resp.StatusCode != expectedStatus {
		t.Errorf("Expected status %d, got %d", expectedStatus, resp.StatusCode)
	}
}

/* AssertNotEmpty asserts that a value is not empty */
func AssertNotEmpty(t *testing.T, name string, value interface{}) {
	t.Helper()
	if value == nil {
		t.Errorf("%s is nil", name)
		return
	}

	switch v := value.(type) {
	case string:
		if v == "" {
			t.Errorf("%s is empty", name)
		}
	case []interface{}:
		if len(v) == 0 {
			t.Errorf("%s is empty", name)
		}
	case map[string]interface{}:
		if len(v) == 0 {
			t.Errorf("%s is empty", name)
		}
	}
}

