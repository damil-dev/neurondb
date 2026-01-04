package middleware

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/logging"
	"github.com/neurondb/NeuronDesktop/api/internal/utils"
)

/* LoggingMiddleware logs HTTP requests and responses */
func LoggingMiddleware(logger *logging.Logger, queries *db.Queries) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get("Upgrade") == "websocket" {
				next.ServeHTTP(w, r)
				return
			}

			start := time.Now()

			var requestBody []byte
			var requestBodyMap map[string]interface{}
			if r.Body != nil {
				requestBody, _ = io.ReadAll(r.Body)
				r.Body = io.NopCloser(bytes.NewBuffer(requestBody))
				
				/* Try to parse as JSON for logging */
				if len(requestBody) > 0 {
					json.Unmarshal(requestBody, &requestBodyMap)
				}
			}

			bodyBuffer := &bytes.Buffer{}
			recorder := &responseRecorder{
				ResponseWriter: w,
				statusCode:     http.StatusOK,
				body:           bodyBuffer,
			}

			/* Capture response body */
			responseWriter := &responseWriterWithBody{
				ResponseWriter: w,
				recorder:       recorder,
			}

			next.ServeHTTP(responseWriter, r)

			duration := time.Since(start)
			
			/* Extract profile ID from context or path */
			var profileID *string
			if profileIDStr := r.Context().Value("profile_id"); profileIDStr != nil {
				if pid, ok := profileIDStr.(string); ok {
					profileID = &pid
				}
			} else {
				/* Try to extract from path */
				pathParts := strings.Split(r.URL.Path, "/")
				for i, part := range pathParts {
					if part == "profiles" && i+1 < len(pathParts) {
						pid := pathParts[i+1]
						/* Validate it looks like a UUID */
						if len(pid) == 36 {
							profileID = &pid
						}
						break
					}
				}
			}

			/* Parse response body if available */
			var responseBodyMap map[string]interface{}
			if recorder.body.Len() > 0 {
				json.Unmarshal(recorder.body.Bytes(), &responseBodyMap)
			}

			/* Log to database asynchronously */
			if queries != nil {
				go func() {
					ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
					defer cancel()

					/* Sanitize sensitive data */
					sanitizedRequestBody := requestBodyMap
					sanitizedResponseBody := responseBodyMap
					if requestBodyMap != nil {
						sanitizedRequestBody = utils.SanitizeMap(requestBodyMap)
					}
					if responseBodyMap != nil {
						sanitizedResponseBody = utils.SanitizeMap(responseBodyMap)
					}

					requestLog := &db.RequestLog{
						ProfileID:    profileID,
						Endpoint:     r.URL.Path,
						Method:       r.Method,
						RequestBody:  sanitizedRequestBody,
						ResponseBody: sanitizedResponseBody,
						StatusCode:   recorder.statusCode,
						DurationMS:   int(duration.Milliseconds()),
					}

					/* Ignore errors - logging should not break requests */
					_ = queries.CreateRequestLog(ctx, requestLog)
				}()
			}

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
	body       *bytes.Buffer
}

func (r *responseRecorder) WriteHeader(code int) {
	r.statusCode = code
	r.ResponseWriter.WriteHeader(code)
}

func (r *responseRecorder) Write(b []byte) (int, error) {
	if r.body != nil {
		r.body.Write(b)
	}
	return r.ResponseWriter.Write(b)
}

/* Hijack preserves the http.Hijacker interface for WebSocket support */
func (r *responseRecorder) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	if hijacker, ok := r.ResponseWriter.(http.Hijacker); ok {
		return hijacker.Hijack()
	}
	return nil, nil, http.ErrHijacked
}

type responseWriterWithBody struct {
	http.ResponseWriter
	recorder *responseRecorder
}

func (w *responseWriterWithBody) Write(b []byte) (int, error) {
	if w.recorder.body != nil {
		w.recorder.body.Write(b)
	}
	return w.ResponseWriter.Write(b)
}

func (w *responseWriterWithBody) WriteHeader(code int) {
	w.recorder.statusCode = code
	w.ResponseWriter.WriteHeader(code)
}
