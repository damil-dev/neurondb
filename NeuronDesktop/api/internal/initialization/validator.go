package initialization

import (
	"context"
	"fmt"
	"net/url"
	"regexp"
	"strings"

	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/logging"
)

/* Validator validates configuration and data */
type Validator struct {
	logger *logging.Logger
}

/* NewValidator creates a new validator instance */
func NewValidator(logger *logging.Logger) *Validator {
	return &Validator{
		logger: logger,
	}
}

/* ValidationResult represents the result of validation */
type ValidationResult struct {
	Valid    bool
	Errors   []string
	Warnings []string
}

/* ValidateAdminUser validates admin user configuration */
func (v *Validator) ValidateAdminUser(ctx context.Context, queries *db.Queries) ValidationResult {
	result := ValidationResult{
		Valid:    true,
		Errors:   []string{},
		Warnings: []string{},
	}

	adminUser, err := queries.GetUserByUsername(ctx, "admin")
	if err != nil {
		result.Errors = append(result.Errors, "Admin user does not exist")
		result.Valid = false
		return result
	}

	if adminUser.Username != "admin" {
		result.Warnings = append(result.Warnings, fmt.Sprintf("Admin user has unexpected username: %s", adminUser.Username))
	}

	if adminUser.PasswordHash == "" {
		result.Errors = append(result.Errors, "Admin user password hash is empty")
		result.Valid = false
	}

	return result
}

/* ValidateProfile validates profile configuration */
func (v *Validator) ValidateProfile(ctx context.Context, queries *db.Queries) ValidationResult {
	result := ValidationResult{
		Valid:    true,
		Errors:   []string{},
		Warnings: []string{},
	}

	profile, err := queries.GetDefaultProfile(ctx)
	if err != nil || profile == nil {
		result.Errors = append(result.Errors, "Default profile does not exist")
		result.Valid = false
		return result
	}

	if profile.Name == "" {
		result.Errors = append(result.Errors, "Profile name is empty")
		result.Valid = false
	}

	dsnResult := v.ValidateDSN(profile.NeuronDBDSN)
	if !dsnResult.Valid {
		result.Errors = append(result.Errors, dsnResult.Errors...)
		result.Valid = false
	}
	result.Warnings = append(result.Warnings, dsnResult.Warnings...)

	if profile.UserID == "" {
		result.Warnings = append(result.Warnings, "Profile user ID is empty")
	}

	return result
}

/* ValidateDSN validates a PostgreSQL DSN */
func (v *Validator) ValidateDSN(dsn string) ValidationResult {
	result := ValidationResult{
		Valid:    true,
		Errors:   []string{},
		Warnings: []string{},
	}

	if dsn == "" {
		result.Errors = append(result.Errors, "DSN is empty")
		result.Valid = false
		return result
	}

	/* Try to parse as URL */
	parsed, err := url.Parse(dsn)
	if err != nil {
		result.Errors = append(result.Errors, fmt.Sprintf("Invalid DSN format: %v", err))
		result.Valid = false
		return result
	}

	/* Validate scheme */
	if parsed.Scheme != "postgres" && parsed.Scheme != "postgresql" {
		result.Errors = append(result.Errors, fmt.Sprintf("Invalid DSN scheme: %s (expected postgres or postgresql)", parsed.Scheme))
		result.Valid = false
	}

	/* Validate host */
	if parsed.Hostname() == "" {
		result.Errors = append(result.Errors, "DSN host is empty")
		result.Valid = false
	}

	/* Validate database name */
	dbName := strings.TrimPrefix(parsed.Path, "/")
	if dbName == "" {
		result.Warnings = append(result.Warnings, "DSN database name is empty")
	}

	/* Validate port (if specified) */
	if parsed.Port() != "" {
		if !isValidPort(parsed.Port()) {
			result.Errors = append(result.Errors, fmt.Sprintf("Invalid port: %s", parsed.Port()))
			result.Valid = false
		}
	}

	return result
}

/* ValidateUsername validates a username */
func (v *Validator) ValidateUsername(username string) ValidationResult {
	result := ValidationResult{
		Valid:    true,
		Errors:   []string{},
		Warnings: []string{},
	}

	if username == "" {
		result.Errors = append(result.Errors, "Username is empty")
		result.Valid = false
		return result
	}

	/* Username should be 3-50 characters */
	if len(username) < 3 {
		result.Errors = append(result.Errors, "Username must be at least 3 characters")
		result.Valid = false
	}

	if len(username) > 50 {
		result.Errors = append(result.Errors, "Username must be at most 50 characters")
		result.Valid = false
	}

	/* Username should only contain alphanumeric and underscore */
	usernameRegex := regexp.MustCompile(`^[a-zA-Z0-9_]+$`)
	if !usernameRegex.MatchString(username) {
		result.Errors = append(result.Errors, "Username can only contain letters, numbers, and underscores")
		result.Valid = false
	}

	return result
}

/* isValidPort checks if a port string is valid */
func isValidPort(port string) bool {
	if port == "" {
		return true // Port is optional
	}

	/* Simple numeric check */
	portRegex := regexp.MustCompile(`^[0-9]+$`)
	if !portRegex.MatchString(port) {
		return false
	}

	/* Check range (1-65535) */
	if len(port) > 5 {
		return false
	}

	return true
}

/* ValidateAll performs all validation checks */
func (v *Validator) ValidateAll(ctx context.Context, queries *db.Queries) ValidationResult {
	result := ValidationResult{
		Valid:    true,
		Errors:   []string{},
		Warnings: []string{},
	}

	/* Validate admin user */
	adminResult := v.ValidateAdminUser(ctx, queries)
	if !adminResult.Valid {
		result.Valid = false
	}
	result.Errors = append(result.Errors, adminResult.Errors...)
	result.Warnings = append(result.Warnings, adminResult.Warnings...)

	/* Validate profile */
	profileResult := v.ValidateProfile(ctx, queries)
	if !profileResult.Valid {
		result.Valid = false
	}
	result.Errors = append(result.Errors, profileResult.Errors...)
	result.Warnings = append(result.Warnings, profileResult.Warnings...)

	return result
}






