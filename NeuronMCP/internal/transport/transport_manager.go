/*-------------------------------------------------------------------------
 *
 * transport_manager.go
 *    Multi-transport coordinator for NeuronMCP
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/transport/transport_manager.go
 *
 *-------------------------------------------------------------------------
 */

package transport

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* TransportType represents a transport type */
type TransportType string

const (
	TransportStdio TransportType = "stdio"
	TransportHTTP  TransportType = "http"
	TransportSSE   TransportType = "sse"
)

/* Manager manages multiple transports */
type Manager struct {
	transports map[TransportType]interface{}
	mcpServer  *mcp.Server
}

/* NewManager creates a new transport manager */
func NewManager(mcpServer *mcp.Server) *Manager {
	return &Manager{
		transports: make(map[TransportType]interface{}),
		mcpServer:  mcpServer,
	}
}

/* StartHTTP starts the HTTP transport */
func (m *Manager) StartHTTP(addr string) error {
	httpTransport := NewHTTPTransport(addr, m.mcpServer)
	m.transports[TransportHTTP] = httpTransport
	return httpTransport.Start()
}

/* Shutdown shuts down all transports */
func (m *Manager) Shutdown(ctx context.Context) error {
	for transportType, transport := range m.transports {
		switch transportType {
		case TransportHTTP:
			if httpTransport, ok := transport.(*HTTPTransport); ok {
				if err := httpTransport.Shutdown(ctx); err != nil {
					return fmt.Errorf("failed to shutdown HTTP transport: %w", err)
				}
			}
		}
	}
	return nil
}





