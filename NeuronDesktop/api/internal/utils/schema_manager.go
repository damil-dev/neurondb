package utils

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	_ "github.com/jackc/pgx/v5/stdlib"
)

// SchemaManager handles database schema initialization and management
type SchemaManager struct {
	schemaPath string
}

// NewSchemaManager creates a new schema manager instance
func NewSchemaManager() *SchemaManager {
	return &SchemaManager{
		schemaPath: findSchemaFile(),
	}
}

// InitSchema initializes the complete schema for a PostgreSQL database connection
// This is a convenience function that uses SchemaManager internally
func InitSchema(ctx context.Context, dsn string) error {
	manager := NewSchemaManager()
	return manager.Initialize(ctx, dsn)
}

// Initialize initializes the database schema for the given DSN
func (sm *SchemaManager) Initialize(ctx context.Context, dsn string) error {
	if sm.schemaPath == "" {
		return fmt.Errorf("neurondesktop.sql file not found")
	}

	// Open connection
	db, err := sql.Open("pgx", dsn)
	if err != nil {
		return fmt.Errorf("failed to open database connection: %w", err)
	}
	defer db.Close()

	// Test connection
	if err := db.PingContext(ctx); err != nil {
		return fmt.Errorf("failed to ping database: %w", err)
	}

	// Read schema file
	schemaSQL, err := os.ReadFile(sm.schemaPath)
	if err != nil {
		return fmt.Errorf("failed to read schema file: %w", err)
	}

	// Execute schema
	return sm.executeSchema(ctx, db, string(schemaSQL))
}

// executeSchema executes the SQL schema statements
func (sm *SchemaManager) executeSchema(ctx context.Context, db *sql.DB, schemaSQL string) error {
	statements := splitSQL(schemaSQL)

	for i, stmt := range statements {
		stmt = strings.TrimSpace(stmt)
		if stmt == "" {
			continue
		}

		if err := sm.executeStatement(ctx, db, stmt); err != nil {
			// Log error but continue - schema might be partially initialized
			fmt.Printf("Warning: Failed to execute schema statement %d: %v\n", i+1, err)
		}
	}

	return nil
}

// executeStatement executes a single SQL statement
func (sm *SchemaManager) executeStatement(ctx context.Context, db *sql.DB, stmt string) error {
	_, err := db.ExecContext(ctx, stmt)
	if err != nil {
		// Check if error is due to resource already existing (acceptable)
		if sm.isAcceptableError(err, stmt) {
			return nil // Ignore acceptable errors
		}
		return fmt.Errorf("failed to execute statement: %w", err)
	}
	return nil
}

// isAcceptableError checks if an error is acceptable (resource already exists)
func (sm *SchemaManager) isAcceptableError(err error, stmt string) bool {
	errMsg := err.Error()
	upperStmt := strings.ToUpper(stmt)

	if strings.Contains(errMsg, "already exists") {
		// Acceptable for extensions, tables, constraints, indexes
		if strings.Contains(upperStmt, "CREATE EXTENSION") ||
			strings.Contains(upperStmt, "CREATE TABLE") ||
			strings.Contains(upperStmt, "CONSTRAINT") ||
			strings.Contains(upperStmt, "CREATE INDEX") {
			return true
		}
	}

	return false
}

// findSchemaFile finds the neurondesktop.sql file relative to the current working directory or executable
func findSchemaFile() string {
	// Try different possible paths
	possiblePaths := []string{
		"neurondesktop.sql",
		"api/neurondesktop.sql",
		"NeuronDesktop/api/neurondesktop.sql",
		"./neurondesktop.sql",
		"./api/neurondesktop.sql",
	}

	// Get current working directory
	cwd, _ := os.Getwd()

	for _, path := range possiblePaths {
		fullPath := filepath.Join(cwd, path)
		if info, err := os.Stat(fullPath); err == nil && !info.IsDir() {
			return fullPath
		}
	}

	// Try relative to executable
	if exePath, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exePath)
		for _, path := range possiblePaths {
			fullPath := filepath.Join(exeDir, path)
			if info, err := os.Stat(fullPath); err == nil && !info.IsDir() {
				return fullPath
			}
		}
	}

	return ""
}

// splitSQL splits SQL into individual statements (package-level helper)
func splitSQL(sql string) []string {
	var statements []string
	
	// Remove comments
	lines := strings.Split(sql, "\n")
	var cleanLines []string
	inBlockComment := false
	
	for _, line := range lines {
		// Handle block comments
		if strings.Contains(line, "/*") {
			inBlockComment = true
		}
		if strings.Contains(line, "*/") {
			inBlockComment = false
			continue
		}
		if inBlockComment {
			continue
		}
		
		// Remove single-line comments
		if commentIdx := strings.Index(line, "--"); commentIdx != -1 {
			line = line[:commentIdx]
		}
		
		cleanLines = append(cleanLines, line)
	}
	
	// Join and split by semicolons
	fullSQL := strings.Join(cleanLines, "\n")
	parts := strings.Split(fullSQL, ";")
	
	for _, part := range parts {
		stmt := strings.TrimSpace(part)
		if stmt != "" {
			statements = append(statements, stmt)
		}
	}
	
	return statements
}

