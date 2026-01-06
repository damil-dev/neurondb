package handlers

import (
	"fmt"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/neurondb"
)

/* ObservabilityHandlers handles observability endpoints */
type ObservabilityHandlers struct {
	queries *db.Queries
}

/* NewObservabilityHandlers creates new observability handlers */
func NewObservabilityHandlers(queries *db.Queries) *ObservabilityHandlers {
	return &ObservabilityHandlers{queries: queries}
}

/* DBHealth represents database health status */
type DBHealth struct {
	Status      string            `json:"status"`
	Version     string            `json:"version,omitempty"`
	Connections int               `json:"connections,omitempty"`
	Uptime      string            `json:"uptime,omitempty"`
	Metrics     map[string]interface{} `json:"metrics,omitempty"`
}

/* IndexHealth represents index health status */
type IndexHealth struct {
	TableName   string  `json:"table_name"`
	IndexName   string  `json:"index_name"`
	IndexType   string  `json:"index_type"`
	Status      string  `json:"status"`
	Size        string  `json:"size,omitempty"`
	BuildProgress float64 `json:"build_progress,omitempty"`
	LastMaintenance time.Time `json:"last_maintenance,omitempty"`
}

/* WorkerStatus represents background worker status */
type WorkerStatus struct {
	WorkerName  string    `json:"worker_name"`
	Status      string    `json:"status"`
	LastRun     time.Time `json:"last_run,omitempty"`
	NextRun     time.Time `json:"next_run,omitempty"`
	JobsProcessed int64   `json:"jobs_processed,omitempty"`
	Errors      int64     `json:"errors,omitempty"`
}

/* GetDBHealth gets database health information */
func (h *ObservabilityHandlers) GetDBHealth(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	profile, err := h.queries.GetProfile(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusNotFound, fmt.Errorf("profile not found"), nil)
		return
	}

	client, err := neurondb.NewClient(profile.NeuronDBDSN)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to create client"), nil)
		return
	}

	// Query database health
	healthQuery := `
		SELECT 
			version() as version,
			(SELECT count(*) FROM pg_stat_activity) as connections,
			(SELECT date_trunc('second', current_timestamp - pg_postmaster_start_time())) as uptime
	`

	rows, err := client.ExecuteSQL(r.Context(), healthQuery)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to query health: %w", err), nil)
		return
	}

	health := DBHealth{
		Status: "healthy",
		Metrics: make(map[string]interface{}),
	}

	rowsList, ok := rows.([]map[string]interface{})
	if ok && len(rowsList) > 0 && len(rowsList[0]) > 0 {
		if version, ok := rowsList[0]["version"].(string); ok {
			health.Version = version
		}
		if conns, ok := rowsList[0]["connections"].(int64); ok {
			health.Connections = int(conns)
		}
	}

	// Check NeuronDB extension
	extQuery := `SELECT extversion FROM pg_extension WHERE extname = 'neurondb'`
	extRowsResult, err := client.ExecuteSQL(r.Context(), extQuery)
	if err != nil {
		health.Status = "degraded"
		health.Metrics["extension"] = "not_loaded"
	} else {
		extRowsList, ok := extRowsResult.([]map[string]interface{})
		if !ok || len(extRowsList) == 0 {
			health.Status = "degraded"
			health.Metrics["extension"] = "not_loaded"
		} else {
			health.Metrics["extension"] = "loaded"
			if version, ok := extRowsList[0]["extversion"].(string); ok {
				health.Metrics["extension_version"] = version
			}
		}
	}

	WriteSuccess(w, health, http.StatusOK)
}

/* GetIndexHealth gets index health information */
func (h *ObservabilityHandlers) GetIndexHealth(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	profile, err := h.queries.GetProfile(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusNotFound, fmt.Errorf("profile not found"), nil)
		return
	}

	client, err := neurondb.NewClient(profile.NeuronDBDSN)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to create client"), nil)
		return
	}

	// Query index health
	query := `
		SELECT 
			schemaname,
			tablename,
			indexname,
			idx_scan,
			idx_tup_read,
			idx_tup_fetch,
			pg_size_pretty(pg_relation_size(indexrelid)) as size
		FROM pg_stat_user_indexes
		WHERE indexname LIKE '%hnsw%' OR indexname LIKE '%ivf%'
		ORDER BY tablename, indexname
	`

	rowsResult, err := client.ExecuteSQL(r.Context(), query)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to query indexes: %w", err), nil)
		return
	}

	rowsList, ok := rowsResult.([]map[string]interface{})
	if !ok {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("unexpected result type from ExecuteSQL"), nil)
		return
	}

	var indexes []IndexHealth
	for _, row := range rowsList {
		idx := IndexHealth{
			Status: "healthy",
		}

		if table, ok := row["tablename"].(string); ok {
			idx.TableName = table
		}
		if index, ok := row["indexname"].(string); ok {
			idx.IndexName = index
			if index[:4] == "hnsw" {
				idx.IndexType = "HNSW"
			} else if index[:3] == "ivf" {
				idx.IndexType = "IVF"
			}
		}
		if size, ok := row["size"].(string); ok {
			idx.Size = size
		}
		if scans, ok := row["idx_scan"].(int64); ok && scans == 0 {
			idx.Status = "unused"
		}

		indexes = append(indexes, idx)
	}

	WriteSuccess(w, indexes, http.StatusOK)
}

/* GetWorkerStatus gets background worker status */
func (h *ObservabilityHandlers) GetWorkerStatus(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	profile, err := h.queries.GetProfile(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusNotFound, fmt.Errorf("profile not found"), nil)
		return
	}

	client, err := neurondb.NewClient(profile.NeuronDBDSN)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to create client"), nil)
		return
	}

	// Query worker status from NeuronDB
	query := `
		SELECT 
			worker_name,
			status,
			last_run,
			next_run,
			jobs_processed,
			errors
		FROM neurondb.worker_status
		ORDER BY worker_name
	`

	rowsResult, err := client.ExecuteSQL(r.Context(), query)
	if err != nil {
		// Workers table might not exist, return empty list
		WriteSuccess(w, []WorkerStatus{}, http.StatusOK)
		return
	}

	rowsList, ok := rowsResult.([]map[string]interface{})
	if !ok {
		// Return empty list if unexpected type
		WriteSuccess(w, []WorkerStatus{}, http.StatusOK)
		return
	}

	var workers []WorkerStatus
	for _, row := range rowsList {
		w := WorkerStatus{
			Status: "unknown",
		}

		if name, ok := row["worker_name"].(string); ok {
			w.WorkerName = name
		}
		if status, ok := row["status"].(string); ok {
			w.Status = status
		}
		if processed, ok := row["jobs_processed"].(int64); ok {
			w.JobsProcessed = processed
		}
		if errors, ok := row["errors"].(int64); ok {
			w.Errors = errors
		}

		workers = append(workers, w)
	}

	WriteSuccess(w, workers, http.StatusOK)
}

/* GetUsageStats gets usage statistics (costs, tokens, etc.) */
func (h *ObservabilityHandlers) GetUsageStats(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	// Query usage from audit logs and request logs
	query := `
		SELECT 
			COUNT(*) as total_requests,
			SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as errors,
			AVG(duration_ms) as avg_duration_ms,
			SUM(CASE WHEN metadata->>'tokens' IS NOT NULL 
				THEN (metadata->>'tokens')::int ELSE 0 END) as total_tokens
		FROM request_logs
		WHERE profile_id = $1
		AND created_at > NOW() - INTERVAL '30 days'
	`

	stats := make(map[string]interface{})
	var totalRequests, errors, totalTokens int64
	var avgDurationMs float64
	err := h.queries.GetDB().QueryRowContext(r.Context(), query, profileID).Scan(
		&totalRequests,
		&errors,
		&avgDurationMs,
		&totalTokens,
	)
	if err == nil {
		stats["total_requests"] = totalRequests
		stats["errors"] = errors
		stats["avg_duration_ms"] = avgDurationMs
		stats["total_tokens"] = totalTokens
	}
	if err != nil {
		// Table might not exist or no data
		stats = map[string]interface{}{
			"total_requests": 0,
			"errors": 0,
			"avg_duration_ms": 0,
			"total_tokens": 0,
		}
	}

	WriteSuccess(w, stats, http.StatusOK)
}

