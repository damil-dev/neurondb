package integration

import (
	"context"
	"testing"

	testutil "github.com/neurondb/NeuronDesktop/api/internal/testing"
	"github.com/neurondb/NeuronDesktop/api/internal/neurondb"
)

func TestNeuronDBIntegration_Connection(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronDB {
		t.Skip("Skipping NeuronDB tests (SKIP_NEURONDB=true)")
	}

	ctx := context.Background()

	/* Verify connection */
	err := testutil.VerifyNeuronDBConnection(ctx, config.NeuronDBDSN)
	if err != nil {
		t.Fatalf("Failed to connect to NeuronDB: %v", err)
	}

	/* Create client */
	client := testutil.CreateNeuronDBTestClient(t, config.NeuronDBDSN)
	if client == nil {
		return
	}
	defer client.Close()

	/* Test listing collections */
	collections, err := client.ListCollections(ctx)
	if err != nil {
		t.Fatalf("Failed to list collections: %v", err)
	}

	t.Logf("Found %d collections", len(collections))
}

func TestNeuronDBIntegration_CreateAndListCollection(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronDB {
		t.Skip("Skipping NeuronDB tests (SKIP_NEURONDB=true)")
	}

	ctx := context.Background()
	client := testutil.CreateNeuronDBTestClient(t, config.NeuronDBDSN)
	if client == nil {
		return
	}
	defer client.Close()

	/* Create test collection */
	schema := "test_schema"
	table := "test_collection"
	vectorCol := "embedding"
	dimensions := 128

	err := testutil.CreateTestCollection(ctx, client, schema, table, vectorCol, dimensions)
	if err != nil {
		t.Fatalf("Failed to create test collection: %v", err)
	}

	/* List collections and verify */
	collections, err := client.ListCollections(ctx)
	if err != nil {
		t.Fatalf("Failed to list collections: %v", err)
	}

	found := false
	for _, coll := range collections {
		if coll.Schema == schema && coll.Name == table {
			found = true
			if coll.VectorCol != vectorCol {
				t.Errorf("Expected vector column %s, got %s", vectorCol, coll.VectorCol)
			}
			break
		}
	}

	if !found {
		t.Error("Test collection not found in list")
	}

	/* Cleanup */
	cleanupSQL := "DROP TABLE IF EXISTS test_schema.test_collection CASCADE; DROP SCHEMA IF EXISTS test_schema CASCADE;"
	client.ExecuteSQLFull(ctx, cleanupSQL)
}

func TestNeuronDBIntegration_SQLExecution_SelectOnly(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronDB {
		t.Skip("Skipping NeuronDB tests (SKIP_NEURONDB=true)")
	}

	ctx := context.Background()
	client := testutil.CreateNeuronDBTestClient(t, config.NeuronDBDSN)
	if client == nil {
		return
	}
	defer client.Close()

	/* Test SELECT query */
	result, err := client.ExecuteSQL(ctx, "SELECT 1 as test_value")
	if err != nil {
		t.Fatalf("Failed to execute SELECT query: %v", err)
	}

	results, ok := result.([]map[string]interface{})
	if !ok {
		t.Fatalf("Expected []map[string]interface{}, got %T", result)
	}

	if len(results) == 0 {
		t.Error("Expected at least one result")
	}

	/* Test that dangerous operations are blocked */
	dangerousQueries := []string{
		"DROP TABLE test",
		"DELETE FROM test",
		"UPDATE test SET x=1",
		"TRUNCATE TABLE test",
		"CREATE TABLE test (id INT)",
	}

	for _, query := range dangerousQueries {
		_, err := client.ExecuteSQL(ctx, query)
		if err == nil {
			t.Errorf("Expected error for dangerous query: %s", query)
		}
	}
}

func TestNeuronDBIntegration_SQLExecution_FullAccess(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronDB {
		t.Skip("Skipping NeuronDB tests (SKIP_NEURONDB=true)")
	}

	ctx := context.Background()
	client := testutil.CreateNeuronDBTestClient(t, config.NeuronDBDSN)
	if client == nil {
		return
	}
	defer client.Close()

	/* Test CREATE TABLE */
	createSQL := `
		CREATE SCHEMA IF NOT EXISTS test_full;
		CREATE TABLE IF NOT EXISTS test_full.test_table (
			id SERIAL PRIMARY KEY,
			name TEXT
		);
	`
	result, err := client.ExecuteSQLFull(ctx, createSQL)
	if err != nil {
		t.Fatalf("Failed to execute CREATE TABLE: %v", err)
	}

	/* Test INSERT */
	insertSQL := "INSERT INTO test_full.test_table (name) VALUES ('test')"
	result, err = client.ExecuteSQLFull(ctx, insertSQL)
	if err != nil {
		t.Fatalf("Failed to execute INSERT: %v", err)
	}

	/* Test SELECT */
	selectSQL := "SELECT * FROM test_full.test_table"
	result, err = client.ExecuteSQLFull(ctx, selectSQL)
	if err != nil {
		t.Fatalf("Failed to execute SELECT: %v", err)
	}

	results, ok := result.([]map[string]interface{})
	if !ok {
		t.Fatalf("Expected []map[string]interface{}, got %T", result)
	}

	if len(results) == 0 {
		t.Error("Expected at least one result from SELECT")
	}

	/* Cleanup */
	cleanupSQL := "DROP SCHEMA IF EXISTS test_full CASCADE;"
	client.ExecuteSQLFull(ctx, cleanupSQL)
}

func TestNeuronDBIntegration_ConnectionPooling(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronDB {
		t.Skip("Skipping NeuronDB tests (SKIP_NEURONDB=true)")
	}

	ctx := context.Background()

	/* Create multiple clients */
	clients := make([]*neurondb.Client, 5)
	for i := 0; i < 5; i++ {
		client := testutil.CreateNeuronDBTestClient(t, config.NeuronDBDSN)
		if client == nil {
			return
		}
		clients[i] = client
	}

	/* Use all clients concurrently */
	done := make(chan bool, len(clients))
	for i, client := range clients {
		go func(idx int, c *neurondb.Client) {
			defer c.Close()
			_, err := c.ListCollections(ctx)
			if err != nil {
				t.Errorf("Client %d failed: %v", idx, err)
			}
			done <- true
		}(i, client)
	}

	/* Wait for all to complete */
	for i := 0; i < len(clients); i++ {
		<-done
	}
}

