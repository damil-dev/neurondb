//go:build test
// +build test

package handlers

import (
	"github.com/neurondb/NeuronDesktop/api/internal/testing"
)

func init() {
	// Register server setup function to avoid import cycles
	// This file is only compiled when building tests
	testing.DefaultServerSetup = SetupTestServer
}





