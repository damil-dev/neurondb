package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/neurondb"
)

/* DatasetHandlers handles dataset ingestion and management */
type DatasetHandlers struct {
	queries *db.Queries
}

/* NewDatasetHandlers creates new dataset handlers */
func NewDatasetHandlers(queries *db.Queries) *DatasetHandlers {
	return &DatasetHandlers{queries: queries}
}

/* IngestRequest represents a dataset ingestion request */
type IngestRequest struct {
	SourceType   string                 `json:"source_type"` // "file", "url", "s3", "github", "huggingface"
	SourcePath   string                 `json:"source_path"`
	Format       string                 `json:"format,omitempty"` // "csv", "json", "jsonl", "parquet"
	TableName    string                 `json:"table_name,omitempty"`
	SchemaName   string                 `json:"schema_name,omitempty"`
	AutoEmbed    bool                   `json:"auto_embed,omitempty"`
	EmbeddingModel string               `json:"embedding_model,omitempty"`
	CreateIndex  bool                   `json:"create_index,omitempty"`
	Config       map[string]interface{} `json:"config,omitempty"`
}

/* IngestResponse represents the ingestion response */
type IngestResponse struct {
	JobID        string    `json:"job_id"`
	Status       string    `json:"status"`
	TableName    string    `json:"table_name,omitempty"`
	RowsIngested int64     `json:"rows_ingested,omitempty"`
	CreatedAt    time.Time `json:"created_at"`
}

/* IngestDataset ingests a dataset from various sources */
func (h *DatasetHandlers) IngestDataset(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	var req IngestRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	// Validate request
	if req.SourceType == "" || req.SourcePath == "" {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("source_type and source_path are required"), nil)
		return
	}

	// Get profile for database connection
	profile, err := h.queries.GetProfile(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusNotFound, fmt.Errorf("profile not found"), nil)
		return
	}

	// Create NeuronDB client
	client, err := neurondb.NewClient(profile.NeuronDBDSN)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to create NeuronDB client: %w", err), nil)
		return
	}

	// Generate job ID
	jobID := fmt.Sprintf("ingest_%d", time.Now().Unix())

	// Start ingestion asynchronously (in production, use a job queue)
	go func() {
		ctx := context.Background()
		
		// Use MCP load_dataset tool via SQL
		query := fmt.Sprintf(`
			SELECT neurondb_mcp_tool_call(
				'load_dataset',
				jsonb_build_object(
					'source_type', %s,
					'source_path', %s,
					'format', %s,
					'schema_name', %s,
					'table_name', %s,
					'auto_embed', %s,
					'embedding_model', %s,
					'create_indexes', %s
				)
			)
		`, 
			quoteSQLString(req.SourceType),
			quoteSQLString(req.SourcePath),
			quoteSQLString(req.Format),
			quoteSQLString(req.SchemaName),
			quoteSQLString(req.TableName),
			fmt.Sprintf("%t", req.AutoEmbed),
			quoteSQLString(req.EmbeddingModel),
			fmt.Sprintf("%t", req.CreateIndex),
		)

		_, err := client.ExecuteSQLFull(ctx, query)
		if err != nil {
			// Log error (in production, update job status)
			fmt.Printf("Ingestion failed for job %s: %v\n", jobID, err)
		}
	}()

	response := IngestResponse{
		JobID:     jobID,
		Status:    "queued",
		TableName: req.TableName,
		CreatedAt: time.Now(),
	}

	WriteSuccess(w, response, http.StatusAccepted)
}

/* GetIngestStatus gets the status of an ingestion job */
func (h *DatasetHandlers) GetIngestStatus(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	jobID := vars["job_id"]

	// In production, query job status from database
	// For now, return a placeholder
	response := map[string]interface{}{
		"job_id":  jobID,
		"status":  "completed",
		"progress": 100,
	}

	WriteSuccess(w, response, http.StatusOK)
}

/* ListIngestJobs lists all ingestion jobs */
func (h *DatasetHandlers) ListIngestJobs(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	// In production, query jobs from database
	// For now, return empty list
	WriteSuccess(w, []interface{}{}, http.StatusOK)
}

func quoteSQLString(s string) string {
	if s == "" {
		return "NULL"
	}
	return fmt.Sprintf("'%s'", s)
}

