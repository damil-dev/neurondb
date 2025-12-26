/*-------------------------------------------------------------------------
 *
 * batch_handlers.go
 *    API handlers for batch operations
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/batch_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"encoding/json"
	"net/http"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* BatchCreateAgents creates multiple agents */
func (h *Handlers) BatchCreateAgents(w http.ResponseWriter, r *http.Request) {
	var reqs []CreateAgentRequest
	if err := json.NewDecoder(r.Body).Decode(&reqs); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "agent", "", nil))
		return
	}

	results := make([]BatchResult, 0, len(reqs))
	for i, req := range reqs {
		if err := ValidateCreateAgentRequest(&req); err != nil {
			results = append(results, BatchResult{
				Index: i,
				Success: false,
				Error: err.Error(),
			})
			continue
		}

		agent := &db.Agent{
			Name:         req.Name,
			Description:  req.Description,
			SystemPrompt: req.SystemPrompt,
			ModelName:    req.ModelName,
			MemoryTable:  req.MemoryTable,
			EnabledTools: req.EnabledTools,
			Config:       db.FromMap(req.Config),
		}

		if err := h.queries.CreateAgent(r.Context(), agent); err != nil {
			results = append(results, BatchResult{
				Index: i,
				Success: false,
				Error: err.Error(),
			})
			continue
		}

		results = append(results, BatchResult{
			Index: i,
			Success: true,
			Data: toAgentResponse(agent),
		})
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"results": results,
		"total": len(reqs),
		"success": countSuccess(results),
		"failed": countFailed(results),
	})
}

/* BatchDeleteAgents deletes multiple agents */
func (h *Handlers) BatchDeleteAgents(w http.ResponseWriter, r *http.Request) {
	var req struct {
		IDs []uuid.UUID `json:"ids"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "agent", "", nil))
		return
	}

	results := make([]BatchResult, 0, len(req.IDs))
	for i, id := range req.IDs {
		if err := h.queries.DeleteAgent(r.Context(), id); err != nil {
			results = append(results, BatchResult{
				Index: i,
				Success: false,
				Error: err.Error(),
			})
			continue
		}

		results = append(results, BatchResult{
			Index: i,
			Success: true,
		})
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"results": results,
		"total": len(req.IDs),
		"success": countSuccess(results),
		"failed": countFailed(results),
	})
}

/* BatchDeleteMessages deletes multiple messages */
func (h *Handlers) BatchDeleteMessages(w http.ResponseWriter, r *http.Request) {
	var req struct {
		IDs []int64 `json:"ids"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "message", "", nil))
		return
	}

	results := make([]BatchResult, 0, len(req.IDs))
	for i, id := range req.IDs {
		if err := h.queries.DeleteMessage(r.Context(), id); err != nil {
			results = append(results, BatchResult{
				Index: i,
				Success: false,
				Error: err.Error(),
			})
			continue
		}

		results = append(results, BatchResult{
			Index: i,
			Success: true,
		})
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"results": results,
		"total": len(req.IDs),
		"success": countSuccess(results),
		"failed": countFailed(results),
	})
}

/* BatchDeleteTools deletes multiple tools */
func (h *Handlers) BatchDeleteTools(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Names []string `json:"names"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "tool", "", nil))
		return
	}

	results := make([]BatchResult, 0, len(req.Names))
	for i, name := range req.Names {
		if err := h.queries.DeleteTool(r.Context(), name); err != nil {
			results = append(results, BatchResult{
				Index: i,
				Success: false,
				Error: err.Error(),
			})
			continue
		}

		results = append(results, BatchResult{
			Index: i,
			Success: true,
		})
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"results": results,
		"total": len(req.Names),
		"success": countSuccess(results),
		"failed": countFailed(results),
	})
}

/* BatchResult represents the result of a batch operation */
type BatchResult struct {
	Index   int         `json:"index"`
	Success bool        `json:"success"`
	Error   string      `json:"error,omitempty"`
	Data    interface{} `json:"data,omitempty"`
}

func countSuccess(results []BatchResult) int {
	count := 0
	for _, r := range results {
		if r.Success {
			count++
		}
	}
	return count
}

func countFailed(results []BatchResult) int {
	count := 0
	for _, r := range results {
		if !r.Success {
			count++
		}
	}
	return count
}

