package utils

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"
)

// ValidationError represents a validation error
type ValidationError struct {
	Field   string
	Message string
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("%s: %s", e.Field, e.Message)
}

// ValidateProfile validates a profile configuration
func ValidateProfile(name, dsn string, mcpConfig map[string]interface{}) []error {
	var errors []error

	// Validate name
	if strings.TrimSpace(name) == "" {
		errors = append(errors, &ValidationError{
			Field:   "name",
			Message: "name is required",
		})
	} else if len(name) > 100 {
		errors = append(errors, &ValidationError{
			Field:   "name",
			Message: "name must be less than 100 characters",
		})
	}

	// Validate DSN
	if strings.TrimSpace(dsn) == "" {
		errors = append(errors, &ValidationError{
			Field:   "neurondb_dsn",
			Message: "NeuronDB DSN is required",
		})
	} else if !isValidDSN(dsn) {
		errors = append(errors, &ValidationError{
			Field:   "neurondb_dsn",
			Message: "invalid DSN format",
		})
	}

	// Validate MCP config
	if mcpConfig != nil {
		if command, ok := mcpConfig["command"].(string); ok {
			if strings.TrimSpace(command) == "" {
				errors = append(errors, &ValidationError{
					Field:   "mcp_config.command",
					Message: "MCP command is required if mcp_config is provided",
				})
			}
		}
	}

	return errors
}

// ValidateAPIKey validates an API key format
func ValidateAPIKey(key string) error {
	if len(key) < 32 {
		return &ValidationError{
			Field:   "api_key",
			Message: "API key must be at least 32 characters",
		}
	}

	// Check for valid base64 URL characters
	base64URLRegex := regexp.MustCompile(`^[A-Za-z0-9_-]+$`)
	if !base64URLRegex.MatchString(key) {
		return &ValidationError{
			Field:   "api_key",
			Message: "API key contains invalid characters",
		}
	}

	return nil
}

// ValidateSearchRequest validates a search request
func ValidateSearchRequest(collection string, limit int, distanceType string) []error {
	var errors []error

	if strings.TrimSpace(collection) == "" {
		errors = append(errors, &ValidationError{
			Field:   "collection",
			Message: "collection is required",
		})
	}

	if limit < 1 || limit > 1000 {
		errors = append(errors, &ValidationError{
			Field:   "limit",
			Message: "limit must be between 1 and 1000",
		})
	}

	validDistanceTypes := map[string]bool{
		"l2":            true,
		"cosine":        true,
		"inner_product": true,
		"euclidean":     true,
		"dot":           true,
	}

	if distanceType != "" && !validDistanceTypes[strings.ToLower(distanceType)] {
		errors = append(errors, &ValidationError{
			Field:   "distance_type",
			Message: fmt.Sprintf("invalid distance type. Must be one of: %v", getKeys(validDistanceTypes)),
		})
	}

	return errors
}

// ValidateSQL validates SQL query for safety
func ValidateSQL(query string) error {
	queryUpper := strings.ToUpper(strings.TrimSpace(query))

	// Must start with SELECT
	if !strings.HasPrefix(queryUpper, "SELECT") {
		return &ValidationError{
			Field:   "query",
			Message: "only SELECT queries are allowed",
		}
	}

	// Check for dangerous operations
	dangerous := []string{
		"DROP", "TRUNCATE", "DELETE", "UPDATE", "INSERT",
		"ALTER", "CREATE", "GRANT", "REVOKE", "EXECUTE",
		"CALL", "COPY", "VACUUM", "ANALYZE",
	}

	for _, keyword := range dangerous {
		if strings.Contains(queryUpper, " "+keyword+" ") || strings.Contains(queryUpper, "\n"+keyword+" ") {
			return &ValidationError{
				Field:   "query",
				Message: fmt.Sprintf("dangerous SQL operation detected: %s", keyword),
			}
		}
	}

	// Check for SQL injection patterns
	sqlInjectionPatterns := []string{
		";--", "';--", "'; DROP", "'; DELETE",
		"UNION SELECT", "OR 1=1", "OR '1'='1",
	}

	for _, pattern := range sqlInjectionPatterns {
		if strings.Contains(strings.ToUpper(query), pattern) {
			return &ValidationError{
				Field:   "query",
				Message: "potentially malicious SQL pattern detected",
			}
		}
	}

	return nil
}

// ValidateToolCall validates a tool call request
func ValidateToolCall(toolName string, arguments map[string]interface{}) []error {
	var errors []error

	if strings.TrimSpace(toolName) == "" {
		errors = append(errors, &ValidationError{
			Field:   "name",
			Message: "tool name is required",
		})
	}

	// Validate tool name format (alphanumeric, underscore, hyphen)
	toolNameRegex := regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)
	if !toolNameRegex.MatchString(toolName) {
		errors = append(errors, &ValidationError{
			Field:   "name",
			Message: "tool name contains invalid characters",
		})
	}

	return errors
}

// Helper functions

func isValidDSN(dsn string) bool {
	// Basic DSN validation - should contain host, user, dbname
	required := []string{"host=", "user=", "dbname="}
	dsnLower := strings.ToLower(dsn)
	for _, req := range required {
		if !strings.Contains(dsnLower, req) {
			return false
		}
	}
	return true
}

func getKeys(m map[string]bool) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

// SanitizeString sanitizes a string input
func SanitizeString(s string, maxLength int) string {
	// Remove control characters
	var builder strings.Builder
	for _, r := range s {
		if unicode.IsPrint(r) || unicode.IsSpace(r) {
			builder.WriteRune(r)
		}
	}
	result := builder.String()

	// Trim and limit length
	result = strings.TrimSpace(result)
	if maxLength > 0 && len(result) > maxLength {
		result = result[:maxLength]
	}

	return result
}

// ValidateEmail validates an email address (basic)
func ValidateEmail(email string) error {
	if email == "" {
		return &ValidationError{
			Field:   "email",
			Message: "email is required",
		}
	}

	emailRegex := regexp.MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`)
	if !emailRegex.MatchString(email) {
		return &ValidationError{
			Field:   "email",
			Message: "invalid email format",
		}
	}

	return nil
}

// ValidateUUID validates a UUID string
func ValidateUUID(uuid string) error {
	uuidRegex := regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)
	if !uuidRegex.MatchString(strings.ToLower(uuid)) {
		return &ValidationError{
			Field:   "uuid",
			Message: "invalid UUID format",
		}
	}
	return nil
}
