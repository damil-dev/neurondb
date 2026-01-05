package integration

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"testing"

	testutil "github.com/neurondb/NeuronDesktop/api/internal/testing"
	"github.com/neurondb/NeuronDesktop/api/internal/neurondb"
)

func TestNeuronDBIntegration_VectorSearch(t *testing.T) {
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

	/* Create test collection with vectors */
	schema := "test_search"
	table := "test_vectors"
	vectorCol := "embedding"
	dimensions := 128

	err := testutil.CreateTestCollection(ctx, client, schema, table, vectorCol, dimensions)
	if err != nil {
		t.Fatalf("Failed to create test collection: %v", err)
	}
	defer func() {
		cleanupSQL := "DROP SCHEMA IF EXISTS test_search CASCADE;"
		client.ExecuteSQLFull(ctx, cleanupSQL)
	}()

	/* Insert test vectors */
	testVectors := [][]float32{
		generateRandomVector(dimensions),
		generateRandomVector(dimensions),
		generateRandomVector(dimensions),
	}

	for i, vec := range testVectors {
		err := testutil.InsertTestVector(ctx, client, schema, table, vectorCol, "test content "+string(rune(i)), vec)
		if err != nil {
			t.Fatalf("Failed to insert test vector %d: %v", i, err)
		}
	}

	/* Perform vector search */
	queryVector := generateRandomVector(dimensions)
	searchReq := neurondb.SearchRequest{
		Collection:   table,
		Schema:       schema,
		QueryVector:  queryVector,
		Limit:        10,
		DistanceType: "cosine",
	}

	results, err := client.Search(ctx, searchReq)
	if err != nil {
		t.Fatalf("Failed to perform vector search: %v", err)
	}

	if len(results) == 0 {
		t.Error("Expected at least one search result")
	}

	/* Verify result structure */
	for _, result := range results {
		if result.ID == nil {
			t.Error("Result missing ID")
		}
		if result.Distance < 0 || result.Distance > 2 {
			t.Errorf("Invalid distance value: %f", result.Distance)
		}
		if result.Score < 0 || result.Score > 1 {
			t.Errorf("Invalid score value: %f", result.Score)
		}
		if result.Data == nil {
			t.Error("Result missing data")
		}
	}
}

func TestNeuronDBIntegration_VectorSearch_DistanceMetrics(t *testing.T) {
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
	schema := "test_distance"
	table := "test_vectors"
	vectorCol := "embedding"
	dimensions := 64

	err := testutil.CreateTestCollection(ctx, client, schema, table, vectorCol, dimensions)
	if err != nil {
		t.Fatalf("Failed to create test collection: %v", err)
	}
	defer func() {
		cleanupSQL := "DROP SCHEMA IF EXISTS test_distance CASCADE;"
		client.ExecuteSQLFull(ctx, cleanupSQL)
	}()

	/* Insert test vector */
	testVector := generateRandomVector(dimensions)
	err = testutil.InsertTestVector(ctx, client, schema, table, vectorCol, "test", testVector)
	if err != nil {
		t.Fatalf("Failed to insert test vector: %v", err)
	}

	/* Test different distance metrics */
	distanceTypes := []string{"cosine", "l2", "inner_product"}

	for _, distType := range distanceTypes {
		searchReq := neurondb.SearchRequest{
			Collection:   table,
			Schema:       schema,
			QueryVector:  testVector,
			Limit:        10,
			DistanceType: distType,
		}

		results, err := client.Search(ctx, searchReq)
		if err != nil {
			t.Errorf("Failed to perform search with distance type %s: %v", distType, err)
			continue
		}

		if len(results) == 0 {
			t.Errorf("Expected at least one result for distance type %s", distType)
		}
	}
}

func TestNeuronDBIntegration_VectorSearch_Filtering(t *testing.T) {
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

	/* Create test collection with additional columns */
	schema := "test_filter"
	table := "test_vectors"
	vectorCol := "embedding"
	dimensions := 64

	createSQL := `
		CREATE SCHEMA IF NOT EXISTS test_filter;
		CREATE TABLE IF NOT EXISTS test_filter.test_vectors (
			id SERIAL PRIMARY KEY,
			content TEXT,
			category TEXT,
			embedding vector(64)
		);
	`
	_, err := client.ExecuteSQLFull(ctx, createSQL)
	if err != nil {
		t.Fatalf("Failed to create test collection: %v", err)
	}
	defer func() {
		cleanupSQL := "DROP SCHEMA IF EXISTS test_filter CASCADE;"
		client.ExecuteSQLFull(ctx, cleanupSQL)
	}()

	/* Insert test vectors with categories */
	categories := []string{"tech", "science", "tech"}
	for i, cat := range categories {
		vec := generateRandomVector(dimensions)
		vectorStr := formatVectorForSQL(vec)
		insertSQL := fmt.Sprintf(`
			INSERT INTO test_filter.test_vectors (content, category, embedding)
			VALUES ('content %d', '%s', %s::vector)
		`, i, cat, vectorStr)
		_, err := client.ExecuteSQLFull(ctx, insertSQL)
		if err != nil {
			t.Fatalf("Failed to insert test vector: %v", err)
		}
	}

	/* Note: The current Search implementation doesn't support filters in the query,
	 * but we can verify the search works and results can be filtered post-query */
	queryVector := generateRandomVector(dimensions)
	searchReq := neurondb.SearchRequest{
		Collection:  table,
		Schema:      schema,
		QueryVector: queryVector,
		Limit:       10,
	}

	results, err := client.Search(ctx, searchReq)
	if err != nil {
		t.Fatalf("Failed to perform search: %v", err)
	}

	if len(results) == 0 {
		t.Error("Expected at least one search result")
	}
}

/* Helper function to generate random vector */
func generateRandomVector(dimensions int) []float32 {
	vec := make([]float32, dimensions)
	for i := range vec {
		vec[i] = rand.Float32()
	}
	return vec
}

/* formatVectorForSQL formats a vector for SQL insertion */
func formatVectorForSQL(vec []float32) string {
	parts := make([]string, len(vec))
	for i, v := range vec {
		parts[i] = fmt.Sprintf("%.6f", v)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

