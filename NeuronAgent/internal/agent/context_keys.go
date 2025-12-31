/*-------------------------------------------------------------------------
 *
 * context_keys.go
 *    Context keys for agent runtime
 *
 * Provides context keys for passing agent and session IDs through context.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/context_keys.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"

	"github.com/google/uuid"
)

type contextKey string

const (
	agentIDContextKey   contextKey = "agent_id"
	sessionIDContextKey contextKey = "session_id"
)

/* WithAgentID adds agent ID to context */
func WithAgentID(ctx context.Context, agentID uuid.UUID) context.Context {
	return context.WithValue(ctx, agentIDContextKey, agentID)
}

/* WithSessionID adds session ID to context */
func WithSessionID(ctx context.Context, sessionID uuid.UUID) context.Context {
	return context.WithValue(ctx, sessionIDContextKey, sessionID)
}

/* GetAgentIDFromContext gets agent ID from context */
func GetAgentIDFromContext(ctx context.Context) (uuid.UUID, bool) {
	agentID, ok := ctx.Value(agentIDContextKey).(uuid.UUID)
	return agentID, ok
}

/* GetSessionIDFromContext gets session ID from context */
func GetSessionIDFromContext(ctx context.Context) (uuid.UUID, bool) {
	sessionID, ok := ctx.Value(sessionIDContextKey).(uuid.UUID)
	return sessionID, ok
}
