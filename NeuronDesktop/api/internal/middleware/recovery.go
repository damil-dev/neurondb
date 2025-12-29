package middleware

import (
	"encoding/json"
	"net/http"

	"github.com/neurondb/NeuronDesktop/api/internal/logging"
)

// ErrorResponse represents an error response (duplicated here to avoid circular import)
type ErrorResponse struct {
	Error     string                 `json:"error"`
	Message   string                 `json:"message,omitempty"`
	Code      string                 `json:"code,omitempty"`
	Details   map[string]interface{} `json:"details,omitempty"`
	RequestID string                 `json:"request_id,omitempty"`
}

// RecoveryMiddleware recovers from panics
func RecoveryMiddleware(logger *logging.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer func() {
				if err := recover(); err != nil {
					requestID := GetRequestID(r.Context())
					logger.Error("Panic recovered", nil, map[string]interface{}{
						"error":      err,
						"path":       r.URL.Path,
						"request_id": requestID,
					})
					
					// Use standardized error response (without importing handlers)
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusInternalServerError)
					
					response := ErrorResponse{
						Error:     "Internal Server Error",
						Message:   "An internal server error occurred",
						Code:      "INTERNAL_ERROR",
						RequestID: requestID,
					}
					
					json.NewEncoder(w).Encode(response)
				}
			}()
			
			next.ServeHTTP(w, r)
		})
	}
}

