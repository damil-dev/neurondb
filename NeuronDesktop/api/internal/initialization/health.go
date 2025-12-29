package initialization

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/logging"
)

// HealthChecker performs comprehensive health checks
type HealthChecker struct {
	queries *db.Queries
	logger  *logging.Logger
}

// NewHealthChecker creates a new health checker instance
func NewHealthChecker(queries *db.Queries, logger *logging.Logger) *HealthChecker {
	return &HealthChecker{
		queries: queries,
		logger:  logger,
	}
}

// HealthStatus represents the overall health status
type HealthStatus struct {
	Status      string                 `json:"status"`      // "healthy", "degraded", "unhealthy"
	Timestamp   time.Time              `json:"timestamp"`
	Checks      map[string]CheckResult `json:"checks"`
	Overall     bool                   `json:"overall"`
	Version     string                 `json:"version,omitempty"`
}

// CheckResult represents the result of an individual health check
type CheckResult struct {
	Status      string        `json:"status"`      // "pass", "warn", "fail"
	Message     string        `json:"message"`
	Duration    time.Duration `json:"duration"`
	LastChecked time.Time     `json:"last_checked"`
}

// CheckAll performs all health checks
func (hc *HealthChecker) CheckAll(ctx context.Context) HealthStatus {
	checks := make(map[string]CheckResult)

	// Database connectivity check
	checks["database"] = hc.checkDatabase(ctx)

	// Admin user check
	checks["admin_user"] = hc.checkAdminUser(ctx)

	// Default profile check
	checks["default_profile"] = hc.checkDefaultProfile(ctx)

	// Schema validation check
	if profile, _ := hc.queries.GetDefaultProfile(ctx); profile != nil {
		checks["profile_schema"] = hc.checkProfileSchema(ctx, profile)
	}

	// Determine overall status
	overall := true
	status := "healthy"
	for _, check := range checks {
		if check.Status == "fail" {
			overall = false
			status = "unhealthy"
			break
		} else if check.Status == "warn" && status == "healthy" {
			status = "degraded"
		}
	}

	return HealthStatus{
		Status:    status,
		Timestamp: time.Now(),
		Checks:    checks,
		Overall:   overall,
	}
}

// checkDatabase checks database connectivity
func (hc *HealthChecker) checkDatabase(ctx context.Context) CheckResult {
	start := time.Now()
	
	// This would require access to the database connection
	// For now, we'll check if we can query the users table
	_, err := hc.queries.GetUserByUsername(ctx, "admin")
	duration := time.Since(start)
	_ = start // Prevent unused variable warning

	if err == sql.ErrNoRows || err == nil {
		return CheckResult{
			Status:      "pass",
			Message:     "Database connection is healthy",
			Duration:    duration,
			LastChecked: time.Now(),
		}
	}

	return CheckResult{
		Status:      "fail",
		Message:     fmt.Sprintf("Database connection failed: %v", err),
		Duration:    duration,
		LastChecked: time.Now(),
	}
}

// checkAdminUser checks if admin user exists
func (hc *HealthChecker) checkAdminUser(ctx context.Context) CheckResult {
	start := time.Now()
	_, err := hc.queries.GetUserByUsername(ctx, "admin")
	duration := time.Since(start)

	if err == nil {
		return CheckResult{
			Status:      "pass",
			Message:     "Admin user exists",
			Duration:    duration,
			LastChecked: time.Now(),
		}
	}

	return CheckResult{
		Status:      "warn",
		Message:     "Admin user not found (may need initialization)",
		Duration:    duration,
		LastChecked: time.Now(),
	}
}

// checkDefaultProfile checks if default profile exists
func (hc *HealthChecker) checkDefaultProfile(ctx context.Context) CheckResult {
	start := time.Now()
	profile, err := hc.queries.GetDefaultProfile(ctx)
	duration := time.Since(start)

	if err == nil && profile != nil {
		return CheckResult{
			Status:      "pass",
			Message:     fmt.Sprintf("Default profile exists: %s", profile.Name),
			Duration:    duration,
			LastChecked: time.Now(),
		}
	}

	return CheckResult{
		Status:      "warn",
		Message:     "Default profile not found (may need initialization)",
		Duration:    duration,
		LastChecked: time.Now(),
	}
}

// checkProfileSchema validates profile database schema
func (hc *HealthChecker) checkProfileSchema(ctx context.Context, profile *db.Profile) CheckResult {
	start := time.Now()

	conn, err := sql.Open("pgx", profile.NeuronDBDSN)
	if err != nil {
		return CheckResult{
			Status:      "fail",
			Message:     fmt.Sprintf("Failed to open profile database: %v", err),
			Duration:    time.Since(start),
			LastChecked: time.Now(),
		}
	}
	defer conn.Close()

	// Check if essential tables exist
	requiredTables := []string{"profiles", "users", "model_configs"}
	for _, table := range requiredTables {
		var exists bool
		query := "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)"
		if err := conn.QueryRowContext(ctx, query, table).Scan(&exists); err != nil || !exists {
			return CheckResult{
				Status:      "warn",
				Message:     fmt.Sprintf("Required table '%s' not found in profile database", table),
				Duration:    time.Since(start),
				LastChecked: time.Now(),
			}
		}
	}

	return CheckResult{
		Status:      "pass",
		Message:     "Profile database schema is valid",
		Duration:    time.Since(start),
		LastChecked: time.Now(),
	}
}

