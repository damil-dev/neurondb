package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/agent"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/neurondb"
)

/* UnifiedQueryHandlers handles unified query endpoints */
type UnifiedQueryHandlers struct {
	queries *db.Queries
}

/* NewUnifiedQueryHandlers creates new unified query handlers */
func NewUnifiedQueryHandlers(queries *db.Queries) *UnifiedQueryHandlers {
	return &UnifiedQueryHandlers{
		queries: queries,
	}
}

/* UnifiedQueryRequest represents a unified query request */
type UnifiedQueryRequest struct {
	QueryType string                 `json:"query_type"` // "sql", "vector", "agent", "hybrid"
	Query     string                 `json:"query"`
	Params    map[string]interface{} `json:"params,omitempty"`
	Context   map[string]interface{} `json:"context,omitempty"`
}

/* UnifiedQueryResponse represents a unified query response */
type UnifiedQueryResponse struct {
	Results    interface{}            `json:"results"`
	Metadata   map[string]interface{} `json:"metadata"`
	Duration   time.Duration          `json:"duration_ms"`
	QueryType  string                 `json:"query_type"`
	Components []string                `json:"components_used"`
}

/* ExecuteUnifiedQuery executes a unified query across components */
func (h *UnifiedQueryHandlers) ExecuteUnifiedQuery(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	var req UnifiedQueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, err, map[string]interface{}{
			"message": "Invalid request body",
		})
		return
	}

	/* Get profile */
	profile, err := h.queries.GetProfile(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusNotFound, err, map[string]interface{}{
			"message": "Profile not found",
		})
		return
	}

	startTime := time.Now()
	response := UnifiedQueryResponse{
		Metadata:   make(map[string]interface{}),
		Components: []string{},
	}

	/* Route query based on type */
	switch req.QueryType {
	case "sql":
		response.Results, response.Metadata, err = h.executeSQLQuery(profile, req)
		response.Components = append(response.Components, "neurondb")
	case "vector":
		response.Results, response.Metadata, err = h.executeVectorQuery(profile, req)
		response.Components = append(response.Components, "neurondb")
	case "agent":
		response.Results, response.Metadata, err = h.executeAgentQuery(profile, req)
		response.Components = append(response.Components, "neuronagent")
	case "hybrid":
		/* Execute hybrid query across multiple components */
		response.Results, response.Metadata, err = h.executeHybridQuery(profile, req)
		response.Components = []string{"neurondb", "neuronagent", "mcp"}
	default:
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid query type"), map[string]interface{}{
			"message": "Invalid query type. Must be: sql, vector, agent, or hybrid",
		})
		return
	}

	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, map[string]interface{}{
			"message": "Query execution failed",
		})
		return
	}

	response.Duration = time.Since(startTime)
	response.QueryType = req.QueryType
	response.Metadata["duration_ms"] = response.Duration.Milliseconds()

	WriteSuccess(w, response, http.StatusOK)
}

/* executeSQLQuery executes a SQL query */
func (h *UnifiedQueryHandlers) executeSQLQuery(profile *db.Profile, req UnifiedQueryRequest) (interface{}, map[string]interface{}, error) {
	if profile.NeuronDBDSN == "" {
		return nil, nil, fmt.Errorf("NeuronDB connection not configured")
	}

	ctx := context.Background()
	client, err := neurondb.NewClient(profile.NeuronDBDSN)
	if err != nil {
		return nil, nil, err
	}
	defer client.Close()

	results, err := client.ExecuteSQL(ctx, req.Query)
	if err != nil {
		return nil, nil, err
	}

	metadata := map[string]interface{}{}
	if resultsList, ok := results.([]interface{}); ok {
		metadata["rows_affected"] = len(resultsList)
	}

	return results, metadata, nil
}

/* executeVectorQuery executes a vector search query */
func (h *UnifiedQueryHandlers) executeVectorQuery(profile *db.Profile, req UnifiedQueryRequest) (interface{}, map[string]interface{}, error) {
	if profile.NeuronDBDSN == "" {
		return nil, nil, fmt.Errorf("NeuronDB connection not configured")
	}

	ctx := context.Background()
	client, err := neurondb.NewClient(profile.NeuronDBDSN)
	if err != nil {
		return nil, nil, err
	}
	defer client.Close()

	/* Extract vector search parameters */
	collection := ""
	if coll, ok := req.Params["collection"].(string); ok {
		collection = coll
	} else if profile.DefaultCollection != "" {
		collection = profile.DefaultCollection
	}

	if collection == "" {
		return nil, nil, fmt.Errorf("collection not specified")
	}

	limit := 10
	if l, ok := req.Params["limit"].(float64); ok {
		limit = int(l)
	}

	/* Extract query vector - if req.Query is text, we'd need to embed it first */
	/* For now, assume req.Query contains a vector string or req.Params has query_vector */
	queryVector := []float32{}
	if qv, ok := req.Params["query_vector"].([]interface{}); ok {
		for _, v := range qv {
			if f, ok := v.(float64); ok {
				queryVector = append(queryVector, float32(f))
			}
		}
	}

	if len(queryVector) == 0 {
		return nil, nil, fmt.Errorf("query_vector not provided in params")
	}

	distanceType := "cosine"
	if dt, ok := req.Params["distance_type"].(string); ok {
		distanceType = dt
	}

	searchReq := neurondb.SearchRequest{
		Collection:   collection,
		QueryVector:  queryVector,
		Limit:        limit,
		DistanceType: distanceType,
	}

	results, err := client.Search(ctx, searchReq)
	if err != nil {
		return nil, nil, err
	}

	metadata := map[string]interface{}{
		"collection": collection,
		"limit":      limit,
		"results":    len(results),
	}

	return results, metadata, nil
}

/* executeAgentQuery executes an agent query */
func (h *UnifiedQueryHandlers) executeAgentQuery(profile *db.Profile, req UnifiedQueryRequest) (interface{}, map[string]interface{}, error) {
	if profile.AgentEndpoint == "" {
		return nil, nil, fmt.Errorf("NeuronAgent connection not configured")
	}

	ctx := context.Background()
	client := agent.NewClient(profile.AgentEndpoint, profile.AgentAPIKey)

	/* Get or create session */
	sessionID := ""
	if sid, ok := req.Params["session_id"].(string); ok {
		sessionID = sid
	} else {
		/* Create new session */
		agentID := ""
		if aid, ok := req.Params["agent_id"].(string); ok {
			agentID = aid
		}
		session, err := client.CreateSession(ctx, agent.CreateSessionRequest{
			AgentID: agentID,
		})
		if err != nil {
			return nil, nil, err
		}
		sessionID = session.ID
	}

	/* Send message */
	message, err := client.SendMessage(ctx, sessionID, agent.SendMessageRequest{
		Role:    "user",
		Content: req.Query,
	})
	if err != nil {
		return nil, nil, err
	}

	metadata := map[string]interface{}{
		"session_id": sessionID,
		"message_id": message.ID,
	}

	return message, metadata, nil
}

/* executeHybridQuery executes a hybrid query across multiple components */
func (h *UnifiedQueryHandlers) executeHybridQuery(profile *db.Profile, req UnifiedQueryRequest) (interface{}, map[string]interface{}, error) {
	/* This is a simplified hybrid query - in production, this would orchestrate
	   queries across NeuronDB, NeuronAgent, and NeuronMCP */
	
	results := map[string]interface{}{}
	metadata := map[string]interface{}{}

	/* Step 1: Vector search in NeuronDB */
	if profile.NeuronDBDSN != "" {
		vectorResults, _, err := h.executeVectorQuery(profile, UnifiedQueryRequest{
			QueryType: "vector",
			Query:     req.Query,
			Params:    req.Params,
		})
		if err == nil {
			results["vector_search"] = vectorResults
		}
	}

	/* Step 2: Agent processing */
	if profile.AgentEndpoint != "" {
		agentResults, agentMeta, err := h.executeAgentQuery(profile, UnifiedQueryRequest{
			QueryType: "agent",
			Query:     req.Query,
			Params:    req.Params,
		})
		if err == nil {
			results["agent_response"] = agentResults
			metadata["agent"] = agentMeta
		}
	}

	return results, metadata, nil
}

