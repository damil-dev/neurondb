package middleware

import (
	"bytes"
	"io"
	"net/http"
	"time"

	"github.com/neurondb/NeuronDesktop/api/internal/logging"
)

// LoggingMiddleware logs HTTP requests and responses
func LoggingMiddleware(logger *logging.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			
			// Read request body
			var requestBody []byte
			if r.Body != nil {
				requestBody, _ = io.ReadAll(r.Body)
				r.Body = io.NopCloser(bytes.NewBuffer(requestBody))
			}
			
			// Wrap response writer to capture status code
			recorder := &responseRecorder{
				ResponseWriter: w,
				statusCode:     http.StatusOK,
			}
			
			// Process request
			next.ServeHTTP(recorder, r)
			
			// Log request
			duration := time.Since(start)
			logger.Info("HTTP request", map[string]interface{}{
				"method":      r.Method,
				"path":        r.URL.Path,
				"status_code": recorder.statusCode,
				"duration_ms": duration.Milliseconds(),
				"remote_addr": r.RemoteAddr,
				"user_agent":  r.UserAgent(),
			})
		})
	}
}

type responseRecorder struct {
	http.ResponseWriter
	statusCode int
}

func (r *responseRecorder) WriteHeader(code int) {
	r.statusCode = code
	r.ResponseWriter.WriteHeader(code)
}

