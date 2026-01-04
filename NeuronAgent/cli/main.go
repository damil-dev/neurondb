/*-------------------------------------------------------------------------
 *
 * main.go
 *    Main entry point for neuronagent-cli
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronAgent/cli/main.go
 *
 *-------------------------------------------------------------------------
 */

package main

import (
	"github.com/neurondb/NeuronAgent/cli/cmd"
)

func main() {
	cmd.Execute()
}
