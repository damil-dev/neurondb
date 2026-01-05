package handlers

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/auth"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
)

/* AuditLogHandlers handles audit log endpoints */
type AuditLogHandlers struct {
	queries *db.Queries
}

/* NewAuditLogHandlers creates new audit log handlers */
func NewAuditLogHandlers(queries *db.Queries) *AuditLogHandlers {
	return &AuditLogHandlers{
		queries: queries,
	}
}

/* ListAuditLogs lists audit logs with filtering (admin only) */
func (h *AuditLogHandlers) ListAuditLogs(w http.ResponseWriter, r *http.Request) {
	/* Check if user is admin */
	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("unauthorized"), nil)
		return
	}

	/* Get user to check admin status */
	user, err := h.queries.GetUserByID(r.Context(), userID)
	if err != nil || !user.IsAdmin {
		WriteError(w, r, http.StatusForbidden, fmt.Errorf("admin access required"), nil)
		return
	}

	/* Parse query parameters */
	limitStr := r.URL.Query().Get("limit")
	limit := 100
	if limitStr != "" {
		if parsed, err := strconv.Atoi(limitStr); err == nil && parsed > 0 && parsed <= 1000 {
			limit = parsed
		}
	}

	filterUserID := r.URL.Query().Get("user_id")
	filterAction := r.URL.Query().Get("action")
	filterResourceType := r.URL.Query().Get("resource_type")

	var userIDFilter, actionFilter, resourceTypeFilter *string
	if filterUserID != "" {
		userIDFilter = &filterUserID
	}
	if filterAction != "" {
		actionFilter = &filterAction
	}
	if filterResourceType != "" {
		resourceTypeFilter = &filterResourceType
	}

	/* Get logs */
	logs, err := h.queries.ListAuditLogs(r.Context(), userIDFilter, actionFilter, resourceTypeFilter, limit)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, logs, http.StatusOK)
}

/* GetAuditLog gets a single audit log entry (admin only) */
func (h *AuditLogHandlers) GetAuditLog(w http.ResponseWriter, r *http.Request) {
	/* Check if user is admin */
	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("unauthorized"), nil)
		return
	}

	user, err := h.queries.GetUserByID(r.Context(), userID)
	if err != nil || !user.IsAdmin {
		WriteError(w, r, http.StatusForbidden, fmt.Errorf("admin access required"), nil)
		return
	}

	vars := mux.Vars(r)
	logID := vars["id"]

	/* Get all logs and find matching one */
	logs, err := h.queries.ListAuditLogs(r.Context(), nil, nil, nil, 10000)
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

	WriteError(w, r, http.StatusNotFound, fmt.Errorf("audit log not found"), nil)
}

/* LogAuditEvent is a helper function to log audit events */
func LogAuditEvent(ctx context.Context, queries *db.Queries, userID, action, resourceType string, resourceID *string, details map[string]interface{}, r *http.Request) {
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		var ipAddress, userAgent *string
		if r != nil {
			ip := r.RemoteAddr
			ipAddress = &ip
			ua := r.UserAgent()
			userAgent = &ua
		}

		auditLog := &db.AuditLog{
			UserID:       userID,
			Action:       action,
			ResourceType: resourceType,
			ResourceID:   resourceID,
			Details:      details,
			IPAddress:    ipAddress,
			UserAgent:    userAgent,
		}

		/* Ignore errors - audit logging should not break operations */
		_ = queries.CreateAuditLog(ctx, auditLog)
	}()
}

