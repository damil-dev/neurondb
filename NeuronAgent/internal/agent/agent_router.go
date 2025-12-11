/*-------------------------------------------------------------------------
 *
 * agent_router.go
 *    Agent routing and delegation
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/agent_router.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

type AgentRouter struct {
	queries *db.Queries
}

/* NewAgentRouter creates a new agent router */
func NewAgentRouter(queries *db.Queries) *AgentRouter {
	return &AgentRouter{queries: queries}
}

/* RouteToAgent routes a task to the most appropriate agent */
func (r *AgentRouter) RouteToAgent(ctx context.Context, task string, availableAgents []uuid.UUID) (uuid.UUID, error) {
	if len(availableAgents) == 0 {
		return uuid.Nil, fmt.Errorf("agent routing failed: task_length=%d, no_agents_available=true", len(task))
	}

	/* Simple routing based on agent names and descriptions */
	/* In production, this would use ML/LLM to match task to agent specialization */
	for _, agentID := range availableAgents {
		agent, err := r.queries.GetAgentByID(ctx, agentID)
		if err != nil {
			continue
		}

		/* Check if agent's name or description matches task keywords */
		if r.matchesTask(agent, task) {
			return agentID, nil
		}
	}

	/* Default to first available agent */
	return availableAgents[0], nil
}

/* matchesTask checks if an agent matches a task */
func (r *AgentRouter) matchesTask(agent *db.Agent, task string) bool {
	taskLower := strings.ToLower(task)
	nameLower := strings.ToLower(agent.Name)
	
	/* Check name keywords */
	keywords := []string{"code", "data", "research", "analysis", "sql", "http"}
	for _, keyword := range keywords {
		if strings.Contains(nameLower, keyword) && strings.Contains(taskLower, keyword) {
			return true
		}
	}

	/* Check description */
	if agent.Description != nil {
		descLower := strings.ToLower(*agent.Description)
		for _, keyword := range keywords {
			if strings.Contains(descLower, keyword) && strings.Contains(taskLower, keyword) {
				return true
			}
		}
	}

	return false
}

/* GetSpecializedAgents gets agents specialized for a task type */
func (r *AgentRouter) GetSpecializedAgents(ctx context.Context, taskType string) ([]uuid.UUID, error) {
	query := `SELECT id FROM neurondb_agent.agents
		WHERE name ILIKE $1 OR description ILIKE $1
		ORDER BY created_at DESC`
	
	var agents []struct {
		ID uuid.UUID `db:"id"`
	}
	
	err := r.queries.GetDB().SelectContext(ctx, &agents, query, "%"+taskType+"%")
	if err != nil {
		return nil, fmt.Errorf("specialized agents retrieval failed: task_type='%s', error=%w", taskType, err)
	}

	agentIDs := make([]uuid.UUID, len(agents))
	for i, agent := range agents {
		agentIDs[i] = agent.ID
	}

	return agentIDs, nil
}

