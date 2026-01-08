/*-------------------------------------------------------------------------
 *
 * resources_test.go
 *    Unit tests for NeuronMCP resources
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/test/unit/resources_test.go
 *
 *-------------------------------------------------------------------------
 */

package unit

import (
	"context"
	"testing"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/resources"
)

/* TestResourceManager tests resource manager functionality */
func TestResourceManager(t *testing.T) {
	db := database.NewDatabase()
	manager := resources.NewManager(db)

	if manager == nil {
		t.Fatal("NewManager returned nil")
	}

	/* Verify resources are registered */
	resourceList := manager.ListResources()
	if len(resourceList) == 0 {
		t.Fatal("No resources registered")
	}

	/* Expected resources */
	expectedResources := map[string]bool{
		"neurondb://schema":      false,
		"neurondb://models":      false,
		"neurondb://indexes":     false,
		"neurondb://config":      false,
		"neurondb://workers":     false,
		"neurondb://vector_stats": false,
		"neurondb://index_health": false,
		"neurondb://datasets":    false,
		"neurondb://collections": false,
	}

	for _, res := range resourceList {
		if _, ok := expectedResources[res.URI]; ok {
			expectedResources[res.URI] = true
		}
	}

	/* Verify all expected resources are present */
	for uri, found := range expectedResources {
		if !found {
			t.Errorf("Expected resource %s not found", uri)
		}
	}
}

/* TestResourceDefinitions tests resource definition completeness */
func TestResourceDefinitions(t *testing.T) {
	db := database.NewDatabase()
	manager := resources.NewManager(db)

	resourceList := manager.ListResources()

	/* Verify we have 9 resources */
	if len(resourceList) != 9 {
		t.Errorf("Expected 9 resources, got %d", len(resourceList))
	}

	/* Verify all resources have required fields */
	for _, res := range resourceList {
		if res.URI == "" {
			t.Error("Resource has empty URI")
		}
		if res.Name == "" {
			t.Errorf("Resource %s has empty name", res.URI)
		}
		if res.Description == "" {
			t.Errorf("Resource %s has empty description", res.URI)
		}
		if res.MimeType == "" {
			t.Errorf("Resource %s has empty MIME type", res.URI)
		}
	}
}

/* TestResourceAccess tests resource access (may require DB connection) */
func TestResourceAccess(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping resource access test in short mode")
	}

	db := database.NewDatabase()
	manager := resources.NewManager(db)

	ctx := context.Background()

	/* Test accessing each resource */
	resourceList := manager.ListResources()
	for _, resDef := range resourceList {
		response, err := manager.HandleResource(ctx, resDef.URI)
		if err != nil {
			/* Resource access may fail due to DB connection, but should not panic */
			if _, ok := err.(*resources.ResourceNotFoundError); !ok {
				/* Not a "not found" error, which is expected for DB connection issues */
				t.Logf("Resource %s access failed (may be expected): %v", resDef.URI, err)
			}
			continue
		}

		if response == nil {
			t.Errorf("Resource %s returned nil response", resDef.URI)
			continue
		}

		if response.Contents == nil || len(response.Contents) == 0 {
			t.Errorf("Resource %s returned empty contents", resDef.URI)
		}
	}
}

/* TestResourceNotFound tests handling of invalid resource URIs */
func TestResourceNotFound(t *testing.T) {
	db := database.NewDatabase()
	manager := resources.NewManager(db)

	ctx := context.Background()

	/* Test with invalid URI */
	_, err := manager.HandleResource(ctx, "neurondb://invalid")
	if err == nil {
		t.Error("Expected error for invalid resource URI")
	}

	if _, ok := err.(*resources.ResourceNotFoundError); !ok {
		t.Errorf("Expected ResourceNotFoundError, got %T", err)
	}
}

