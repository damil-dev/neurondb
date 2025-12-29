package handlers

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"

	_ "github.com/jackc/pgx/v5/stdlib"
)

// DatabaseTestRequest represents a database connection test request
type DatabaseTestRequest struct {
	Host     string `json:"host"`
	Port     string `json:"port"`
	Database string `json:"database"`
	User     string `json:"user"`
	Password string `json:"password"`
}

// DatabaseTestResponse represents the response from a database test
type DatabaseTestResponse struct {
	Success      bool     `json:"success"`
	Message      string   `json:"message"`
	SchemaExists bool     `json:"schema_exists"`
	MissingTables []string `json:"missing_tables,omitempty"`
	DSN          string   `json:"dsn,omitempty"`
}

// DatabaseTestHandlers handles database connection testing
type DatabaseTestHandlers struct{}

// NewDatabaseTestHandlers creates a new database test handlers instance
func NewDatabaseTestHandlers() *DatabaseTestHandlers {
	return &DatabaseTestHandlers{}
}

// TestConnection tests a PostgreSQL database connection and checks for schema
func (h *DatabaseTestHandlers) TestConnection(w http.ResponseWriter, r *http.Request) {
	var req DatabaseTestRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	// Validate required fields
	if req.Host == "" || req.Database == "" || req.User == "" {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("host, database, and user are required"), nil)
		return
	}

	// Build DSN
	dsn := h.buildDSN(req)

	// Test connection
	conn, err := sql.Open("pgx", dsn)
	if err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("failed to open database connection: %w", err), nil)
		return
	}
	defer conn.Close()

	// Test with context timeout
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	if err := conn.PingContext(ctx); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("connection failed: %w", err), nil)
		return
	}

	// Check if NeuronDesktop schema exists
	schemaExists, missingTables := h.checkSchema(ctx, conn)

	response := DatabaseTestResponse{
		Success:      true,
		SchemaExists: schemaExists,
		MissingTables: missingTables,
		DSN:          dsn,
	}

	if schemaExists {
		response.Message = "Connection successful and NeuronDesktop schema is configured."
	} else {
		response.Message = "Connection successful, but NeuronDesktop schema is not configured. Please run neurondesktop.sql on this database."
	}

	WriteSuccess(w, response, http.StatusOK)
}

// buildDSN builds a PostgreSQL DSN from connection parameters
func (h *DatabaseTestHandlers) buildDSN(req DatabaseTestRequest) string {
	if req.Port == "" {
		req.Port = "5432"
	}

	if req.Password != "" {
		return fmt.Sprintf("postgresql://%s:%s@%s:%s/%s",
			encodeURIComponent(req.User),
			encodeURIComponent(req.Password),
			req.Host,
			req.Port,
			req.Database)
	}
	return fmt.Sprintf("postgresql://%s@%s:%s/%s",
		encodeURIComponent(req.User),
		req.Host,
		req.Port,
		req.Database)
}

// encodeURIComponent encodes a string for use in a URI
func encodeURIComponent(s string) string {
	return url.QueryEscape(s)
}

// checkSchema checks if the NeuronDesktop schema exists
func (h *DatabaseTestHandlers) checkSchema(ctx context.Context, conn *sql.DB) (bool, []string) {
	requiredTables := []string{
		"users",
		"profiles",
		"api_keys",
		"request_logs",
		"model_configs",
		"app_settings",
	}

	var missingTables []string
	allExist := true

	for _, table := range requiredTables {
		var exists bool
		query := "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)"
		if err := conn.QueryRowContext(ctx, query, table).Scan(&exists); err != nil {
			// If we can't check, assume it doesn't exist
			exists = false
		}

		if !exists {
			missingTables = append(missingTables, table)
			allExist = false
		}
	}

	return allExist, missingTables
}

