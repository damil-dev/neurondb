# NeuronDesktop Testing Guide

This directory contains comprehensive tests for NeuronDesktop, including unit tests, integration tests, and end-to-end tests.

## Test Structure

```
tests/
├── e2e/                    # End-to-end workflow tests
│   ├── auth_flow_test.go
│   ├── profile_workflow_test.go
│   ├── model_config_workflow_test.go
│   └── metrics_workflow_test.go
├── setup_test_env.sh      # Test environment setup script
├── cleanup_test_env.sh    # Test environment cleanup script
├── run_all_tests.sh       # Master test runner
└── readme.md              # This file
```

## Test Categories

### Unit Tests
Located in `api/internal/handlers/*_test.go` and `api/internal/db/*_test.go`

- **Handler Tests**: Test individual API endpoint handlers
  - Authentication (register, login, logout)
  - Profile management (CRUD operations)
  - MCP operations
  - NeuronDB operations
  - Agent operations
  - Model configuration
  - Metrics

- **Database Tests**: Test database query operations
  - User CRUD
  - Profile CRUD
  - Model config CRUD
  - Query validation

### Integration Tests
Located in `api/internal/mcp/client_test.go`, `api/internal/neurondb/client_test.go`, `api/internal/agent/client_test.go`

- **MCP Client**: Test MCP server communication
- **NeuronDB Client**: Test NeuronDB operations
- **Agent Client**: Test NeuronAgent API communication

### End-to-End Tests
Located in `tests/e2e/*_test.go`

- **Auth Flow**: Complete authentication workflows
- **Profile Workflow**: Full profile management lifecycle
- **Model Config Workflow**: Model configuration management
- **Metrics Workflow**: Metrics collection and monitoring

## Running Tests

### Prerequisites

1. **PostgreSQL**: A PostgreSQL instance must be running
2. **Go**: Go 1.21+ must be installed
3. **Test Database**: Will be created automatically if it doesn't exist

### Environment Variables

```bash
# Test database configuration
export TEST_DB_HOST=localhost
export TEST_DB_PORT=5432
export TEST_DB_USER=neurondesk
export TEST_DB_PASSWORD=neurondesk
export TEST_DB_NAME=neurondesk_test

# Optional: External service endpoints for integration tests
export TEST_NEURONDB_DSN="host=localhost port=5432 user=neurondb dbname=neurondb"
export TEST_AGENT_ENDPOINT="http://localhost:8080"
```

### Quick Start

Run all tests:

```bash
cd NeuronDesktop
./tests/run_all_tests.sh
```

### Running Specific Test Categories

**Unit tests only:**
```bash
cd api
go test ./internal/handlers/... ./internal/db/...
```

**Integration tests only:**
```bash
cd api
go test ./internal/mcp/... ./internal/neurondb/... ./internal/agent/...
```

**End-to-end tests only:**
```bash
cd api
go test ../../tests/e2e/...
```

**Specific test file:**
```bash
cd api
go test -v ./internal/handlers/auth_test.go ./internal/handlers/auth.go
```

**Specific test function:**
```bash
cd api
go test -v -run TestAuthHandlers_Register ./internal/handlers/...
```

### Test Coverage

Generate coverage report:

```bash
cd api
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out -o coverage.html
```

View coverage in browser:
```bash
open coverage.html  # macOS
xdg-open coverage.html  # Linux
```

## Test Infrastructure

### Test Utilities

Located in `api/internal/testing/`:

- **testutil.go**: Database setup, fixtures, helpers
- **httpclient.go**: HTTP test client with authentication
- **mocks.go**: Mock services for testing

### Test Database

Tests use a separate test database (`neurondesk_test` by default) to avoid affecting development data.

The test database is:
- Created automatically if it doesn't exist
- Truncated between test runs (not dropped)
- Isolated from production data

## Writing New Tests

### Handler Test Example

```go
func TestMyHandler_MyEndpoint(t *testing.T) {
    tdb := testing.SetupTestDB(t)
    defer tdb.CleanupTestDB(t)

    client := testing.NewTestClient(t, tdb.Queries)
    defer client.Server.Close()

    ctx := context.Background()
    err := client.Authenticate(ctx, "testuser", "password123")
    if err != nil {
        t.Fatalf("Failed to authenticate: %v", err)
    }

    resp, err := client.Get("/api/v1/my-endpoint")
    if err != nil {
        t.Fatalf("Request failed: %v", err)
    }
    defer resp.Body.Close()

    testing.AssertStatus(t, resp, http.StatusOK)
}
```

### Integration Test Example

```go
func TestMyClient_Operation(t *testing.T) {
    client, err := mypackage.NewClient("connection-string")
    if err != nil {
        t.Skipf("Skipping test: cannot connect: %v", err)
    }
    defer client.Close()

    ctx := context.Background()
    result, err := client.Operation(ctx)
    if err != nil {
        t.Fatalf("Operation failed: %v", err)
    }

    // Verify result
    if result == nil {
        t.Error("Expected result")
    }
}
```

## Best Practices

1. **Isolation**: Each test should be independent and not rely on other tests
2. **Cleanup**: Always clean up test data using `defer tdb.CleanupTestDB(t)`
3. **Skip External Services**: Integration tests should skip gracefully if external services aren't available
4. **Error Messages**: Provide clear error messages when tests fail
5. **Coverage**: Aim for 80%+ code coverage on critical paths

## Troubleshooting

### Tests Fail with Database Connection Errors

1. Check PostgreSQL is running: `pg_isready`
2. Verify connection settings in environment variables
3. Ensure test database user has proper permissions

### Integration Tests Fail

Integration tests may fail if external services (NeuronDB, NeuronAgent, NeuronMCP) are not running. This is expected and tests will skip gracefully.

To run integration tests:
1. Start NeuronDB: `docker compose up neurondb`
2. Start NeuronAgent: `docker compose up neuronagent`
3. Start NeuronMCP: Ensure `neurondb-mcp` binary is available

### Tests Are Slow

- Use `-short` flag to skip long-running tests: `go test -short ./...`
- Run specific test categories instead of all tests
- Use test database on localhost (not remote)

## CI/CD Integration

Tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    export TEST_DB_HOST=localhost
    export TEST_DB_PORT=5432
    ./tests/run_all_tests.sh
```

## Coverage Goals

- **Unit Tests**: 80%+ coverage on handlers and database operations
- **Integration Tests**: Cover all external service interactions
- **E2E Tests**: Cover all critical user workflows

## Contributing

When adding new features:
1. Write unit tests for new handlers
2. Add integration tests for external service interactions
3. Add E2E tests for new user workflows
4. Ensure all tests pass before submitting PR






