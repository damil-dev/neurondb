package middleware

import (
	"encoding/json"
	"net/http"

	"github.com/neurondb/NeuronDesktop/api/internal/logging"
)

// RecoveryMiddleware recovers from panics
func RecoveryMiddleware(logger *logging.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer func() {
				if err := recover(); err != nil {
					logger.Error("Panic recovered", nil, map[string]interface{}{
						"error": err,
						"path":  r.URL.Path,
					})
					
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusInternalServerError)
					json.NewEncoder(w).Encode(map[string]interface{}{
						"error": "Internal server error",
					})
				}
			}()
			
			next.ServeHTTP(w, r)
		})
	}
}

