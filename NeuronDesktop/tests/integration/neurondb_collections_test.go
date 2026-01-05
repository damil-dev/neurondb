package integration

import (
	"context"
	"fmt"
	"testing"

	testutil "github.com/neurondb/NeuronDesktop/api/internal/testing"
	"github.com/neurondb/NeuronDesktop/api/internal/neurondb"
)

func TestNeuronDBIntegration_ListCollections(t *testing.T) {
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

	collections, err := client.ListCollections(ctx)
	if err != nil {
		t.Fatalf("Failed to list collections: %v", err)
	}

	/* Verify collection structure */
	for _, coll := range collections {
		if coll.Name == "" {
			t.Error("Collection missing name")
		}
		if coll.Schema == "" {
			t.Error("Collection missing schema")
		}
		if coll.VectorCol == "" {
			t.Error("Collection missing vector column")
		}
	}
}

func TestNeuronDBIntegration_CollectionMetadata(t *testing.T) {
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

	/* Create test collection with index */
	schema := "test_metadata"
	table := "test_collection"
	vectorCol := "embedding"
	dimensions := 128

	err := testutil.CreateTestCollection(ctx, client, schema, table, vectorCol, dimensions)
	if err != nil {
		t.Fatalf("Failed to create test collection: %v", err)
	}
	defer func() {
		cleanupSQL := "DROP SCHEMA IF EXISTS test_metadata CASCADE;"
		client.ExecuteSQLFull(ctx, cleanupSQL)
	}()

	/* Create index */
	indexSQL := `
		CREATE INDEX IF NOT EXISTS test_collection_embedding_idx 
		ON test_metadata.test_collection 
		USING hnsw (embedding vector_cosine_ops)
		WITH (m = 16, ef_construction = 64);
	`
	_, err = client.ExecuteSQLFull(ctx, indexSQL)
	if err != nil {
		/* Index creation might fail if extension not available, skip */
		t.Logf("Index creation failed (might not have extension): %v", err)
	}

	/* List collections and verify metadata */
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
			/* Verify row count */
			if coll.RowCount < 0 {
				t.Error("Invalid row count")
			}
			break
		}
	}

	if !found {
		t.Error("Test collection not found in list")
	}
}

func TestNeuronDBIntegration_CollectionIndexes(t *testing.T) {
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
	schema := "test_indexes"
	table := "test_collection"
	vectorCol := "embedding"
	dimensions := 64

	err := testutil.CreateTestCollection(ctx, client, schema, table, vectorCol, dimensions)
	if err != nil {
		t.Fatalf("Failed to create test collection: %v", err)
	}
	defer func() {
		cleanupSQL := "DROP SCHEMA IF EXISTS test_indexes CASCADE;"
		client.ExecuteSQLFull(ctx, cleanupSQL)
	}()

	/* List collections and check for indexes */
	collections, err := client.ListCollections(ctx)
	if err != nil {
		t.Fatalf("Failed to list collections: %v", err)
	}

	for _, coll := range collections {
		if coll.Schema == schema && coll.Name == table {
			/* Indexes might be empty if extension not available */
			if len(coll.Indexes) > 0 {
				for _, idx := range coll.Indexes {
					if idx.Name == "" {
						t.Error("Index missing name")
					}
					if idx.Type == "" {
						t.Error("Index missing type")
					}
				}
			}
			break
		}
	}
}

func TestNeuronDBIntegration_MultipleCollections(t *testing.T) {
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

	/* Create multiple test collections */
	schemas := []string{"test_multi1", "test_multi2", "test_multi3"}
	table := "test_collection"
	vectorCol := "embedding"
	dimensions := 64

	for _, schema := range schemas {
		err := testutil.CreateTestCollection(ctx, client, schema, table, vectorCol, dimensions)
		if err != nil {
			t.Fatalf("Failed to create test collection in %s: %v", schema, err)
		}
	}
	defer func() {
		for _, schema := range schemas {
			cleanupSQL := fmt.Sprintf("DROP SCHEMA IF EXISTS %s CASCADE;", schema)
			client.ExecuteSQLFull(ctx, cleanupSQL)
		}
	}()

	/* List collections and verify all are present */
	collections, err := client.ListCollections(ctx)
	if err != nil {
		t.Fatalf("Failed to list collections: %v", err)
	}

	foundCount := 0
	for _, coll := range collections {
		for _, schema := range schemas {
			if coll.Schema == schema && coll.Name == table {
				foundCount++
				break
			}
		}
	}

	if foundCount != len(schemas) {
		t.Errorf("Expected %d collections, found %d", len(schemas), foundCount)
	}
}

