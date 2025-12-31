package validation

import (
	"fmt"
	"regexp"
	"strings"
)

var (
	dsnRegex = regexp.MustCompile(`^(host|user|dbname|password|port)=`)
)

/* ValidateDSN validates a PostgreSQL DSN string */
func ValidateDSN(dsn, fieldName string) error {
	if dsn == "" {
		return fmt.Errorf("%s cannot be empty", fieldName)
	}
	
	dsn = strings.TrimSpace(dsn)
	
	// Check for required components
	required := []string{"host=", "user=", "dbname="}
	dsnLower := strings.ToLower(dsn)
	for _, req := range required {
		if !strings.Contains(dsnLower, req) {
			return fmt.Errorf("%s is missing required component: %s", fieldName, req[:len(req)-1])
		}
	}
	
	// Basic format validation - should contain key=value pairs
	parts := strings.Fields(dsn)
	if len(parts) == 0 {
		return fmt.Errorf("%s has invalid format: no components found", fieldName)
	}
	
	// Check for potentially malicious patterns
	maliciousPatterns := []string{
		";", "&&", "||", "`", "$(", "${",
	}
	
	for _, pattern := range maliciousPatterns {
		if strings.Contains(dsn, pattern) {
			return fmt.Errorf("%s contains potentially malicious pattern: %s", fieldName, pattern)
		}
	}
	
	return nil
}

/* ValidateDSNRequired validates a DSN and ensures it's not empty */
func ValidateDSNRequired(dsn, fieldName string) error {
	if dsn == "" {
		return fmt.Errorf("%s is required and cannot be empty", fieldName)
	}
	return ValidateDSN(dsn, fieldName)
}

