package validation

import (
	"fmt"
	"regexp"
	"strings"
)

var (
	dsnRegex = regexp.MustCompile(`^(host|user|dbname|password|port)=`)
)

/* DSNValidationResult represents the result of DSN validation */
type DSNValidationResult struct {
	Valid   bool
	Error   string
	Warnings []string
}

/* ValidateDSN validates a PostgreSQL DSN string and returns a result */
func ValidateDSN(dsn string) DSNValidationResult {
	result := DSNValidationResult{
		Valid:    true,
		Warnings: []string{},
	}

	if dsn == "" {
		result.Valid = false
		result.Error = "DSN cannot be empty"
		return result
	}

	dsn = strings.TrimSpace(dsn)

	/* Check for required components */
	required := []string{"host=", "user=", "dbname="}
	dsnLower := strings.ToLower(dsn)
	for _, req := range required {
		if !strings.Contains(dsnLower, req) {
			result.Valid = false
			result.Error = fmt.Sprintf("DSN is missing required component: %s", req[:len(req)-1])
			return result
		}
	}

	/* Basic format validation */
	parts := strings.Fields(dsn)
	if len(parts) == 0 {
		result.Valid = false
		result.Error = "DSN has invalid format: no components found"
		return result
	}

	/* Check for potentially malicious patterns */
	maliciousPatterns := []string{
		";", "&&", "||", "`", "$(", "${",
	}

	for _, pattern := range maliciousPatterns {
		if strings.Contains(dsn, pattern) {
			result.Valid = false
			result.Error = fmt.Sprintf("DSN contains potentially malicious pattern: %s", pattern)
			return result
		}
	}

	return result
}

/* ValidateDSNLegacy validates a PostgreSQL DSN string (legacy function for backward compatibility) */
func ValidateDSNLegacy(dsn, fieldName string) error {
	if dsn == "" {
		return fmt.Errorf("%s cannot be empty", fieldName)
	}
	
	dsn = strings.TrimSpace(dsn)
	
	/* Check for required components */
	required := []string{"host=", "user=", "dbname="}
	dsnLower := strings.ToLower(dsn)
	for _, req := range required {
		if !strings.Contains(dsnLower, req) {
			return fmt.Errorf("%s is missing required component: %s", fieldName, req[:len(req)-1])
		}
	}
	
	/* Basic format validation - should contain key=value pairs */
	parts := strings.Fields(dsn)
	if len(parts) == 0 {
		return fmt.Errorf("%s has invalid format: no components found", fieldName)
	}
	
	/* Check for potentially malicious patterns */
	maliciousPatterns := []string{
		";", "&&", "||", "`", "$(", "${",
	}
	
	for _, pattern := range maliciousPatterns {
		if strings.Contains(dsn, pattern) {
			return fmt.Errorf("%s contains potentially malicious pattern: %s", fieldName, pattern)
		}
	}
	
	result := ValidateDSN(dsn)
	if !result.Valid {
		return fmt.Errorf("%s: %s", fieldName, result.Error)
	}
	return nil
}

/* ValidateDSNRequired validates a DSN and ensures it's not empty */
func ValidateDSNRequired(dsn, fieldName string) error {
	if dsn == "" {
		return fmt.Errorf("%s is required and cannot be empty", fieldName)
	}
	return ValidateDSNLegacy(dsn, fieldName)
}

