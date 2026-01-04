package handlers

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
)

/* AnalyticsHandlers handles analytics endpoints */
type AnalyticsHandlers struct {
	queries *db.Queries
	db      *sql.DB
}

/* NewAnalyticsHandlers creates new analytics handlers */
func NewAnalyticsHandlers(queries *db.Queries, database *sql.DB) *AnalyticsHandlers {
	return &AnalyticsHandlers{
		queries: queries,
		db:      database,
	}
}

/* GetAnalytics returns analytics data */
func (h *AnalyticsHandlers) GetAnalytics(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	/* Get time range from query params (default: last 7 days) */
	timeRange := r.URL.Query().Get("range")
	if timeRange == "" {
		timeRange = "7d"
	}

	var startTime time.Time
	switch timeRange {
	case "1d":
		startTime = time.Now().Add(-24 * time.Hour)
	case "7d":
		startTime = time.Now().Add(-7 * 24 * time.Hour)
	case "30d":
		startTime = time.Now().Add(-30 * 24 * time.Hour)
	case "90d":
		startTime = time.Now().Add(-90 * 24 * time.Hour)
	default:
		startTime = time.Now().Add(-7 * 24 * time.Hour)
	}

	ctx := r.Context()

	analytics := map[string]interface{}{
		"time_range": timeRange,
		"start_time": startTime,
		"end_time":   time.Now(),
	}

	requestStats, err := h.getRequestStats(ctx, profileID, startTime)
	if err == nil {
		analytics["requests"] = requestStats
	}

	endpointUsage, err := h.getEndpointUsage(ctx, profileID, startTime)
	if err == nil {
		analytics["endpoints"] = endpointUsage
	}

	errorStats, err := h.getErrorStats(ctx, profileID, startTime)
	if err == nil {
		analytics["errors"] = errorStats
	}

	responseTimeStats, err := h.getResponseTimeStats(ctx, profileID, startTime)
	if err == nil {
		analytics["response_times"] = responseTimeStats
	}

	dailyActivity, err := h.getDailyActivity(ctx, profileID, startTime)
	if err == nil {
		analytics["daily_activity"] = dailyActivity
	}

	WriteSuccess(w, analytics, http.StatusOK)
}

/* getRequestStats gets request statistics */
func (h *AnalyticsHandlers) getRequestStats(ctx context.Context, profileID string, startTime time.Time) (map[string]interface{}, error) {
	var query string
	var args []interface{}

	if profileID != "" {
		query = `
			SELECT 
				COUNT(*) as total,
				COUNT(*) FILTER (WHERE status_code < 400) as successful,
				COUNT(*) FILTER (WHERE status_code >= 400) as failed,
				AVG(duration_ms) as avg_duration,
				MIN(duration_ms) as min_duration,
				MAX(duration_ms) as max_duration
			FROM request_logs
			WHERE profile_id = $1 AND created_at >= $2
		`
		args = []interface{}{profileID, startTime}
	} else {
		query = `
			SELECT 
				COUNT(*) as total,
				COUNT(*) FILTER (WHERE status_code < 400) as successful,
				COUNT(*) FILTER (WHERE status_code >= 400) as failed,
				AVG(duration_ms) as avg_duration,
				MIN(duration_ms) as min_duration,
				MAX(duration_ms) as max_duration
			FROM request_logs
			WHERE created_at >= $1
		`
		args = []interface{}{startTime}
	}

	var total, successful, failed sql.NullInt64
	var avgDuration, minDuration, maxDuration sql.NullFloat64

	err := h.db.QueryRowContext(ctx, query, args...).Scan(
		&total, &successful, &failed, &avgDuration, &minDuration, &maxDuration,
	)
	if err != nil {
		return nil, err
	}

	stats := map[string]interface{}{
		"total":      total.Int64,
		"successful": successful.Int64,
		"failed":     failed.Int64,
	}

	if avgDuration.Valid {
		stats["avg_duration_ms"] = avgDuration.Float64
	}
	if minDuration.Valid {
		stats["min_duration_ms"] = minDuration.Float64
	}
	if maxDuration.Valid {
		stats["max_duration_ms"] = maxDuration.Float64
	}

	return stats, nil
}

/* getEndpointUsage gets endpoint usage statistics */
func (h *AnalyticsHandlers) getEndpointUsage(ctx context.Context, profileID string, startTime time.Time) ([]map[string]interface{}, error) {
	var query string
	var args []interface{}

	if profileID != "" {
		query = `
			SELECT endpoint, method, COUNT(*) as count, AVG(duration_ms) as avg_duration
			FROM request_logs
			WHERE profile_id = $1 AND created_at >= $2
			GROUP BY endpoint, method
			ORDER BY count DESC
			LIMIT 20
		`
		args = []interface{}{profileID, startTime}
	} else {
		query = `
			SELECT endpoint, method, COUNT(*) as count, AVG(duration_ms) as avg_duration
			FROM request_logs
			WHERE created_at >= $1
			GROUP BY endpoint, method
			ORDER BY count DESC
			LIMIT 20
		`
		args = []interface{}{startTime}
	}

	rows, err := h.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []map[string]interface{}
	for rows.Next() {
		var endpoint, method string
		var count sql.NullInt64
		var avgDuration sql.NullFloat64

		if err := rows.Scan(&endpoint, &method, &count, &avgDuration); err != nil {
			continue
		}

		result := map[string]interface{}{
			"endpoint": endpoint,
			"method":    method,
			"count":     count.Int64,
		}

		if avgDuration.Valid {
			result["avg_duration_ms"] = avgDuration.Float64
		}

		results = append(results, result)
	}

	return results, nil
	}

/* getErrorStats gets error statistics */
func (h *AnalyticsHandlers) getErrorStats(ctx context.Context, profileID string, startTime time.Time) ([]map[string]interface{}, error) {
	var query string
	var args []interface{}

	if profileID != "" {
		query = `
			SELECT status_code, COUNT(*) as count
			FROM request_logs
			WHERE profile_id = $1 AND created_at >= $2 AND status_code >= 400
			GROUP BY status_code
			ORDER BY count DESC
		`
		args = []interface{}{profileID, startTime}
	} else {
		query = `
			SELECT status_code, COUNT(*) as count
			FROM request_logs
			WHERE created_at >= $1 AND status_code >= 400
			GROUP BY status_code
			ORDER BY count DESC
		`
		args = []interface{}{startTime}
	}

	rows, err := h.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []map[string]interface{}
	for rows.Next() {
		var statusCode int
		var count sql.NullInt64

		if err := rows.Scan(&statusCode, &count); err != nil {
			continue
		}

		results = append(results, map[string]interface{}{
			"status_code": statusCode,
			"count":       count.Int64,
		})
	}

	return results, nil
}

/* getResponseTimeStats gets response time statistics */
func (h *AnalyticsHandlers) getResponseTimeStats(ctx context.Context, profileID string, startTime time.Time) (map[string]interface{}, error) {
	var query string
	var args []interface{}

	if profileID != "" {
		query = `
			SELECT 
				PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms) as p50,
				PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95,
				PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) as p99
			FROM request_logs
			WHERE profile_id = $1 AND created_at >= $2
		`
		args = []interface{}{profileID, startTime}
	} else {
		query = `
			SELECT 
				PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms) as p50,
				PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95,
				PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) as p99
			FROM request_logs
			WHERE created_at >= $1
		`
		args = []interface{}{startTime}
	}

	var p50, p95, p99 sql.NullFloat64

	err := h.db.QueryRowContext(ctx, query, args...).Scan(&p50, &p95, &p99)
	if err != nil {
		return nil, err
	}

	stats := map[string]interface{}{}
	if p50.Valid {
		stats["p50_ms"] = p50.Float64
	}
	if p95.Valid {
		stats["p95_ms"] = p95.Float64
	}
	if p99.Valid {
		stats["p99_ms"] = p99.Float64
	}

	return stats, nil
}

/* getDailyActivity gets daily activity data */
func (h *AnalyticsHandlers) getDailyActivity(ctx context.Context, profileID string, startTime time.Time) ([]map[string]interface{}, error) {
	var query string
	var args []interface{}

	if profileID != "" {
		query = `
			SELECT 
				DATE(created_at) as date,
				COUNT(*) as requests,
				COUNT(*) FILTER (WHERE status_code < 400) as successful,
				COUNT(*) FILTER (WHERE status_code >= 400) as failed,
				AVG(duration_ms) as avg_duration
			FROM request_logs
			WHERE profile_id = $1 AND created_at >= $2
			GROUP BY DATE(created_at)
			ORDER BY date ASC
		`
		args = []interface{}{profileID, startTime}
	} else {
		query = `
			SELECT 
				DATE(created_at) as date,
				COUNT(*) as requests,
				COUNT(*) FILTER (WHERE status_code < 400) as successful,
				COUNT(*) FILTER (WHERE status_code >= 400) as failed,
				AVG(duration_ms) as avg_duration
			FROM request_logs
			WHERE created_at >= $1
			GROUP BY DATE(created_at)
			ORDER BY date ASC
		`
		args = []interface{}{startTime}
	}

	rows, err := h.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []map[string]interface{}
	for rows.Next() {
		var date time.Time
		var requests, successful, failed sql.NullInt64
		var avgDuration sql.NullFloat64

		if err := rows.Scan(&date, &requests, &successful, &failed, &avgDuration); err != nil {
			continue
		}

		result := map[string]interface{}{
			"date":       date.Format("2006-01-02"),
			"requests":   requests.Int64,
			"successful": successful.Int64,
			"failed":     failed.Int64,
		}

		if avgDuration.Valid {
			result["avg_duration_ms"] = avgDuration.Float64
		}

		results = append(results, result)
	}

	return results, nil
}

/* ExportAnalytics exports analytics data as JSON */
func (h *AnalyticsHandlers) ExportAnalytics(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	timeRange := r.URL.Query().Get("range")
	if timeRange == "" {
		timeRange = "30d"
	}

	analyticsReq := &http.Request{
		URL: r.URL,
	}
	analyticsReq = analyticsReq.WithContext(r.Context())

	analyticsData := make(map[string]interface{})
	
	var startTime time.Time
	switch timeRange {
	case "1d":
		startTime = time.Now().Add(-24 * time.Hour)
	case "7d":
		startTime = time.Now().Add(-7 * 24 * time.Hour)
	case "30d":
		startTime = time.Now().Add(-30 * 24 * time.Hour)
	case "90d":
		startTime = time.Now().Add(-90 * 24 * time.Hour)
	default:
		startTime = time.Now().Add(-30 * 24 * time.Hour)
	}

	ctx := r.Context()
	analyticsData["time_range"] = timeRange
	analyticsData["start_time"] = startTime
	analyticsData["end_time"] = time.Now()

	if requestStats, err := h.getRequestStats(ctx, profileID, startTime); err == nil {
		analyticsData["requests"] = requestStats
	}
	if endpointUsage, err := h.getEndpointUsage(ctx, profileID, startTime); err == nil {
		analyticsData["endpoints"] = endpointUsage
	}
	if errorStats, err := h.getErrorStats(ctx, profileID, startTime); err == nil {
		analyticsData["errors"] = errorStats
	}
	if responseTimeStats, err := h.getResponseTimeStats(ctx, profileID, startTime); err == nil {
		analyticsData["response_times"] = responseTimeStats
	}
	if dailyActivity, err := h.getDailyActivity(ctx, profileID, startTime); err == nil {
		analyticsData["daily_activity"] = dailyActivity
	}

	/* Set headers for file download */
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=analytics-%s-%s.json", profileID, time.Now().Format("20060102")))
	json.NewEncoder(w).Encode(analyticsData)
}

