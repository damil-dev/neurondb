package handlers

import (
	"context"
	"database/sql"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/agent"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/metrics"
	"github.com/neurondb/NeuronDesktop/api/internal/neurondb"
)

/* DashboardHandlers handles dashboard endpoints */
type DashboardHandlers struct {
	queries *db.Queries
	db      *sql.DB
	metrics *metrics.Metrics
}

/* NewDashboardHandlers creates new dashboard handlers */
func NewDashboardHandlers(database *sql.DB) *DashboardHandlers {
	return &DashboardHandlers{
		queries: db.NewQueries(database),
		db:      database,
		metrics: metrics.GetGlobalMetrics(),
	}
}

/* DashboardStats represents dashboard statistics */
type DashboardStats struct {
	SystemMetrics    map[string]interface{} `json:"system_metrics"`
	NeuronDBStats    *NeuronDBStats        `json:"neurondb_stats,omitempty"`
	NeuronAgentStats *AgentStats          `json:"neuronagent_stats,omitempty"`
	MCPStats         *MCPStats            `json:"mcp_stats,omitempty"`
	RecentActivity   []ActivityItem       `json:"recent_activity"`
	HealthStatus     HealthStatus         `json:"health_status"`
}

/* NeuronDBStats represents NeuronDB statistics */
type NeuronDBStats struct {
	CollectionsCount int64   `json:"collections_count"`
	TotalVectors      int64   `json:"total_vectors"`
	IndexesCount     int64   `json:"indexes_count"`
	QueryCount       int64   `json:"query_count"`
	AvgQueryTime     float64 `json:"avg_query_time"`
}

/* AgentStats represents NeuronAgent statistics */
type AgentStats struct {
	AgentsCount      int64   `json:"agents_count"`
	SessionsCount    int64   `json:"sessions_count"`
	MessagesCount    int64   `json:"messages_count"`
	AvgResponseTime  float64 `json:"avg_response_time"`
}

/* MCPStats represents MCP statistics */
type MCPStats struct {
	ToolsCount       int64 `json:"tools_count"`
	ToolsCalled      int64 `json:"tools_called"`
	ActiveConnections int `json:"active_connections"`
}

/* ActivityItem represents a recent activity item */
type ActivityItem struct {
	ID          string    `json:"id"`
	Type        string    `json:"type"`
	Description string    `json:"description"`
	Timestamp   time.Time `json:"timestamp"`
	User        string    `json:"user,omitempty"`
}

/* HealthStatus represents overall health status */
type HealthStatus struct {
	Status      string            `json:"status"`
	Components  map[string]string `json:"components"`
	LastChecked time.Time         `json:"last_checked"`
}

/* GetDashboard returns dashboard statistics */
func (h *DashboardHandlers) GetDashboard(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	stats := DashboardStats{
		SystemMetrics:  h.metrics.GetStats(),
		RecentActivity:  []ActivityItem{},
		HealthStatus: HealthStatus{
			Status:      "healthy",
			Components:  make(map[string]string),
			LastChecked: time.Now(),
		},
	}

	/* Get profile */
	profile, err := h.queries.GetProfile(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusNotFound, err, map[string]interface{}{
			"message": "Profile not found",
		})
		return
	}

	/* Get NeuronDB stats if configured */
	if profile.NeuronDBDSN != "" {
		neurondbClient, err := neurondb.NewClient(profile.NeuronDBDSN)
		if err == nil {
			defer neurondbClient.Close()
			ndbStats := h.getNeuronDBStats(r.Context(), neurondbClient)
			stats.NeuronDBStats = ndbStats
			stats.HealthStatus.Components["neurondb"] = "healthy"
		} else {
			stats.HealthStatus.Components["neurondb"] = "unhealthy"
		}
	}

	/* Get NeuronAgent stats if configured */
	if profile.AgentEndpoint != "" {
		agentClient := agent.NewClient(profile.AgentEndpoint, profile.AgentAPIKey)
		agentStats := h.getAgentStats(r.Context(), agentClient)
		stats.NeuronAgentStats = agentStats
		stats.HealthStatus.Components["neuronagent"] = "healthy"
	}

	/* Get MCP stats */
	mcpStats := h.getMCPStats()
	stats.MCPStats = mcpStats
	stats.HealthStatus.Components["mcp"] = "healthy"

	/* Get recent activity from audit logs */
	stats.RecentActivity = h.getRecentActivity(profileID)

	WriteSuccess(w, stats, http.StatusOK)
}

/* getNeuronDBStats retrieves NeuronDB statistics */
func (h *DashboardHandlers) getNeuronDBStats(ctx context.Context, client *neurondb.Client) *NeuronDBStats {
	stats := &NeuronDBStats{}

	/* Get collections count */
	collections, _ := client.ListCollections(ctx)
	stats.CollectionsCount = int64(len(collections))

	/* Get total vectors and indexes (sum across all collections) */
	totalIndexes := int64(0)
	for _, coll := range collections {
		stats.TotalVectors += coll.RowCount
		totalIndexes += int64(len(coll.Indexes))
	}
	stats.IndexesCount = totalIndexes

	return stats
}

/* getAgentStats retrieves NeuronAgent statistics */
func (h *DashboardHandlers) getAgentStats(ctx context.Context, client *agent.Client) *AgentStats {
	stats := &AgentStats{}

	/* Get agents count */
	agents, err := client.ListAgents(ctx)
	if err == nil {
		stats.AgentsCount = int64(len(agents))
	}

	return stats
}

/* getMCPStats retrieves MCP statistics */
func (h *DashboardHandlers) getMCPStats() *MCPStats {
	return &MCPStats{
		ToolsCount:       0, /* Will be populated from active MCP connections */
		ToolsCalled:       0,
		ActiveConnections: 0,
	}
}

/* getRecentActivity retrieves recent activity from audit logs */
func (h *DashboardHandlers) getRecentActivity(profileID string) []ActivityItem {
	/* Query audit logs for recent activity */
	query := `
		SELECT id, action_type, description, created_at, user_id
		FROM audit_logs
		WHERE profile_id = $1
		ORDER BY created_at DESC
		LIMIT 10
	`

	rows, err := h.db.Query(query, profileID)
	if err != nil {
		return []ActivityItem{}
	}
	defer rows.Close()

	var activities []ActivityItem
	for rows.Next() {
		var activity ActivityItem
		var userID sql.NullString
		err := rows.Scan(
			&activity.ID,
			&activity.Type,
			&activity.Description,
			&activity.Timestamp,
			&userID,
		)
		if err != nil {
			continue
		}
		if userID.Valid {
			activity.User = userID.String
		}
		activities = append(activities, activity)
	}

	return activities
}

