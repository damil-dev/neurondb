package validation

import (
	"fmt"
	"net/http"
	"time"
)

/* HealthCheckResult represents a health check result */
type HealthCheckResult struct {
	Service     string
	Status      string
	ResponseTime time.Duration
	Error       error
}

/* ValidateHealthCheck validates health check response */
func ValidateHealthCheck(result HealthCheckResult) error {
	if result.Error != nil {
		return fmt.Errorf("health check failed for %s: %w", result.Service, result.Error)
	}
	
	if result.Status != "healthy" && result.Status != "ok" {
		return fmt.Errorf("health check returned unhealthy status: %s", result.Status)
	}
	
	// Validate response time (should be under 5 seconds)
	if result.ResponseTime > 5*time.Second {
		return fmt.Errorf("health check response time too slow: %v", result.ResponseTime)
	}
	
	return nil
}

/* ValidateHTTPStatus validates HTTP status code */
func ValidateHTTPStatus(statusCode int, expectedCodes ...int) error {
	for _, expected := range expectedCodes {
		if statusCode == expected {
			return nil
		}
	}
	
	// Check if it's a client error (4xx)
	if statusCode >= 400 && statusCode < 500 {
		return fmt.Errorf("client error: HTTP %d", statusCode)
	}
	
	// Check if it's a server error (5xx)
	if statusCode >= 500 {
		return fmt.Errorf("server error: HTTP %d", statusCode)
	}
	
	return fmt.Errorf("unexpected HTTP status: %d (expected one of: %v)", statusCode, expectedCodes)
}

/* ValidateHTTPResponse validates HTTP response */
func ValidateHTTPResponse(resp *http.Response, expectedStatus int) error {
	if resp == nil {
		return fmt.Errorf("HTTP response is nil")
	}
	
	if err := ValidateHTTPStatus(resp.StatusCode, expectedStatus); err != nil {
		return err
	}
	
	return nil
}

