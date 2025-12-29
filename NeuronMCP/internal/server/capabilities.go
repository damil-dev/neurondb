/*-------------------------------------------------------------------------
 *
 * capabilities.go
 *    Server capabilities and version negotiation
 *
 * Exposes server version, tool versions, model versions, and feature flags.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/server/capabilities.go
 *
 *-------------------------------------------------------------------------
 */

package server

import (
	"sync"

	"github.com/neurondb/NeuronMCP/internal/tools"
	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* CapabilitiesManager manages server capabilities and version information */
type CapabilitiesManager struct {
	mu             sync.RWMutex
	serverVersion  string
	serverName     string
	toolRegistry   *tools.ToolRegistry
	featureFlags   map[string]bool
	modelVersions  map[string]string
}

/* NewCapabilitiesManager creates a new capabilities manager */
func NewCapabilitiesManager(serverName, serverVersion string, toolRegistry *tools.ToolRegistry) *CapabilitiesManager {
	return &CapabilitiesManager{
		serverVersion: serverVersion,
		serverName:    serverName,
		toolRegistry:  toolRegistry,
		featureFlags: map[string]bool{
			"pagination":        true,
			"streaming":         true,
			"dry_run":           true,
			"idempotency":       true,
			"audit_logging":     true,
			"scoped_auth":       true,
			"rate_limiting":     true,
			"output_validation": true,
			"tool_versioning":   true,
			"deprecation":       true,
			"composite_tools":   true,
			"resource_catalog":  true,
		},
		modelVersions: map[string]string{
			"default_embedding": "1.0.0",
			"default_llm":       "1.0.0",
		},
	}
}

/* GetServerInfo returns server information */
func (cm *CapabilitiesManager) GetServerInfo() mcp.ServerInfo {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	return mcp.ServerInfo{
		Name:    cm.serverName,
		Version: cm.serverVersion,
	}
}

/* GetServerCapabilities returns server capabilities */
func (cm *CapabilitiesManager) GetServerCapabilities() mcp.ServerCapabilities {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	/* Get all tool definitions to extract versions */
	toolDefs := cm.toolRegistry.GetAllDefinitions()
	toolVersions := make(map[string]string)
	for _, def := range toolDefs {
		if def.Version != "" {
			toolVersions[def.Name] = def.Version
		}
	}

	return mcp.ServerCapabilities{
		Tools: mcp.ToolsCapability{
			ListChanged: true,
		},
		Resources: mcp.ResourcesCapability{
			Subscribe:   false,
			ListChanged: true,
		},
		Experimental: map[string]interface{}{
			"feature_flags":  cm.featureFlags,
			"tool_versions":  toolVersions,
			"model_versions": cm.modelVersions,
			"server_version": cm.serverVersion,
		},
	}
}

/* SetFeatureFlag sets a feature flag */
func (cm *CapabilitiesManager) SetFeatureFlag(name string, enabled bool) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.featureFlags[name] = enabled
}

/* GetFeatureFlag gets a feature flag */
func (cm *CapabilitiesManager) GetFeatureFlag(name string) bool {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.featureFlags[name]
}

/* SetModelVersion sets a model version */
func (cm *CapabilitiesManager) SetModelVersion(modelName, version string) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.modelVersions[modelName] = version
}

/* GetModelVersion gets a model version */
func (cm *CapabilitiesManager) GetModelVersion(modelName string) string {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.modelVersions[modelName]
}





