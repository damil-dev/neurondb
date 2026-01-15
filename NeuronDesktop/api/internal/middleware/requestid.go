package middleware

import (
	"context"
	"net/http"

	"github.com/google/uuid"
)

/* Context key type for type-safe context values */
type requestIDKeyType string

const RequestIDKey requestIDKeyType = "request_id"

/* RequestIDMiddleware adds a request ID to each request */
func RequestIDMiddleware() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			requestID := r.Header.Get("X-Request-Id")
			if requestID == "" {
				requestID = uuid.New().String()
			}

			w.Header().Set("X-Request-Id", requestID)

			ctx := context.WithValue(r.Context(), RequestIDKey, requestID)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

/* GetRequestID gets request ID from context */
func GetRequestID(ctx context.Context) string {
	if id, ok := ctx.Value(RequestIDKey).(string); ok {
		return id
	}
	return ""
}







