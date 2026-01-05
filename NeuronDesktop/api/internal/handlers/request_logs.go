package handlers

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
)

/* RequestLogHandlers handles request log endpoints */
type RequestLogHandlers struct {
	queries *db.Queries
}

/* NewRequestLogHandlers creates new request log handlers */
func NewRequestLogHandlers(queries *db.Queries) *RequestLogHandlers {
	return &RequestLogHandlers{
		queries: queries,
	}
}

/* ListLogs lists request logs with filtering */
func (h *RequestLogHandlers) ListLogs(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	/* Parse query parameters */
	limitStr := r.URL.Query().Get("limit")
	limit := 100
	if limitStr != "" {
		if parsed, err := strconv.Atoi(limitStr); err == nil && parsed > 0 && parsed <= 1000 {
			limit = parsed
		}
	}

	statusCodeStr := r.URL.Query().Get("status_code")
	var statusCode *int
	if statusCodeStr != "" {
		if parsed, err := strconv.Atoi(statusCodeStr); err == nil {
			statusCode = &parsed
		}
	}

	endpoint := r.URL.Query().Get("endpoint")
	startDateStr := r.URL.Query().Get("start_date")
	endDateStr := r.URL.Query().Get("end_date")

	var startDate, endDate *time.Time
	if startDateStr != "" {
		if parsed, err := time.Parse(time.RFC3339, startDateStr); err == nil {
			startDate = &parsed
		}
	}
	if endDateStr != "" {
		if parsed, err := time.Parse(time.RFC3339, endDateStr); err == nil {
			endDate = &parsed
		}
	}

	/* Get logs from database */
	logs, err := h.queries.ListRequestLogs(r.Context(), &profileID, limit)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	/* Apply filters */
	filteredLogs := make([]db.RequestLog, 0)
	for _, log := range logs {
		/* Filter by status code */
		if statusCode != nil && log.StatusCode != *statusCode {
			continue
		}

		/* Filter by endpoint */
		if endpoint != "" && log.Endpoint != endpoint {
			continue
		}

		/* Filter by date range */
		if startDate != nil && log.CreatedAt.Before(*startDate) {
			continue
		}
		if endDate != nil && log.CreatedAt.After(*endDate) {
			continue
		}

		filteredLogs = append(filteredLogs, log)
	}

	WriteSuccess(w, filteredLogs, http.StatusOK)
}

/* GetLog gets a single request log */
func (h *RequestLogHandlers) GetLog(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	logID := vars["id"]

	/* Get all logs and find the one matching ID and profile */
	logs, err := h.queries.ListRequestLogs(r.Context(), &profileID, 1000)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	for _, log := range logs {
		if log.ID == logID {
			WriteSuccess(w, log, http.StatusOK)
			return
		}
	}

	WriteError(w, r, http.StatusNotFound, fmt.Errorf("log not found"), nil)
}

/* DeleteLog deletes a request log */
func (h *RequestLogHandlers) DeleteLog(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	logID := vars["id"]

	/* Delete log from database */
	ctx := r.Context()
	query := `DELETE FROM request_logs WHERE id = $1`
	_, err := h.queries.GetDB().ExecContext(ctx, query, logID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, map[string]interface{}{"success": true}, http.StatusOK)
}

/* ExportLogs exports logs as JSON or CSV */
func (h *RequestLogHandlers) ExportLogs(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	format := r.URL.Query().Get("format")
	if format == "" {
		format = "json"
	}

	/* Get logs */
	logs, err := h.queries.ListRequestLogs(r.Context(), &profileID, 10000)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	if format == "csv" {
		/* Export as CSV */
		w.Header().Set("Content-Type", "text/csv")
		w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=logs-%s-%s.csv", profileID, time.Now().Format("20060102")))
		
		writer := csv.NewWriter(w)
		defer writer.Flush()

		/* Write header */
		writer.Write([]string{"ID", "Profile ID", "Endpoint", "Method", "Status Code", "Duration (ms)", "Created At"})

		/* Write rows */
		for _, log := range logs {
			writer.Write([]string{
				log.ID,
				func() string {
					if log.ProfileID != nil {
						return *log.ProfileID
					}
					return ""
				}(),
				log.Endpoint,
				log.Method,
				strconv.Itoa(log.StatusCode),
				strconv.Itoa(log.DurationMS),
				log.CreatedAt.Format(time.RFC3339),
			})
		}
	} else {
		/* Export as JSON */
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=logs-%s-%s.json", profileID, time.Now().Format("20060102")))
		json.NewEncoder(w).Encode(logs)
	}
}

