/*-------------------------------------------------------------------------
 *
 * middleware.go
 *    HTTP middleware for NeuronAgent API
 *
 * Provides authentication, CORS, logging, and request ID middleware
 * for the NeuronAgent HTTP API server.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/middleware.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/neurondb/NeuronAgent/internal/auth"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

type contextKey string

const apiKeyContextKey contextKey = "api_key"
const principalContextKey contextKey = "principal"

/* AuthMiddleware authenticates requests using API keys and resolves principals */
func AuthMiddleware(keyManager *auth.APIKeyManager, principalManager *auth.PrincipalManager, rateLimiter *auth.RateLimiter) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    /* Skip auth for health and metrics endpoints */
			if r.URL.Path == "/health" || r.URL.Path == "/metrics" {
				next.ServeHTTP(w, r)
				return
			}

    /* Get API key from header */
			authHeader := r.Header.Get("Authorization")
			if authHeader == "" {
				requestID := GetRequestID(r.Context())
				respondError(w, WrapError(ErrUnauthorized, requestID))
				return
			}

    /* Extract key (format: "Bearer <key>" or "ApiKey <key>") */
			parts := strings.Fields(authHeader)
			if len(parts) != 2 {
				requestID := GetRequestID(r.Context())
				respondError(w, WrapError(ErrUnauthorized, requestID))
				return
			}

			key := parts[1]
			keyPrefix := key
			if len(keyPrefix) > 8 {
				keyPrefix = keyPrefix[:8]
			}
			requestID := GetRequestID(r.Context())
			ctx := metrics.WithLogContext(r.Context(), requestID, "", "", "", "")
			metrics.DebugWithContext(ctx, "API key extracted from authorization header", map[string]interface{}{
				"key_prefix": keyPrefix,
				"key_length":  len(key),
			})

    /* Validate key */
			apiKey, err := keyManager.ValidateAPIKey(r.Context(), key)
			if err != nil {
				metrics.WarnWithContext(ctx, "API key validation failed", map[string]interface{}{
					"key_prefix": keyPrefix,
					"error":      err.Error(),
				})
				respondError(w, WrapError(ErrUnauthorized, requestID))
				return
			}
			metrics.DebugWithContext(ctx, "API key validation succeeded", map[string]interface{}{
				"key_prefix": apiKey.KeyPrefix,
				"key_id":     apiKey.ID.String(),
			})

    /* Check rate limit */
			if !rateLimiter.CheckLimit(apiKey.ID.String(), apiKey.RateLimitPerMin) {
				requestID := GetRequestID(r.Context())
				respondError(w, WrapError(NewError(http.StatusTooManyRequests, "rate limit exceeded", nil), requestID))
				return
			}

    /* Resolve principal */
			principal, err := principalManager.ResolvePrincipalFromAPIKey(r.Context(), apiKey)
			if err != nil {
				metrics.WarnWithContext(ctx, "Principal resolution failed, continuing with request", map[string]interface{}{
					"key_id": apiKey.ID.String(),
					"error":  err.Error(),
				})
				/* Continue anyway - principal resolution failure should not block requests */
			}

    /* Add API key and principal to context */
			ctx := context.WithValue(r.Context(), apiKeyContextKey, apiKey)
			if principal != nil {
				ctx = context.WithValue(ctx, principalContextKey, principal)
			}
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

/* SecurityHeadersMiddleware adds security headers to all HTTP responses */
func SecurityHeadersMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Prevent MIME type sniffing
		w.Header().Set("X-Content-Type-Options", "nosniff")
		// Enable XSS protection
		w.Header().Set("X-XSS-Protection", "1; mode=block")
		// Prevent clickjacking
		w.Header().Set("X-Frame-Options", "DENY")
		// Enforce HTTPS in production (Strict-Transport-Security)
		// Note: Only set if request is already HTTPS to avoid issues
		if r.TLS != nil {
			w.Header().Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
		}
		// Content Security Policy - restrictive by default
		w.Header().Set("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'")
		// Referrer Policy
		w.Header().Set("Referrer-Policy", "strict-origin-when-cross-origin")
		// Permissions Policy (formerly Feature-Policy)
		w.Header().Set("Permissions-Policy", "geolocation=(), microphone=(), camera=()")

		next.ServeHTTP(w, r)
	})
}

/* CORSMiddleware adds CORS headers */
func CORSMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

/* LoggingMiddleware logs requests with structured logging and metrics */
func LoggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		
   /* Wrap response writer to capture status code */
		wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}
		
		next.ServeHTTP(wrapped, r)
		
		duration := time.Since(start)
		
   /* Record metrics */
		endpoint := r.URL.Path
		metrics.RecordHTTPRequest(r.Method, endpoint, wrapped.statusCode, duration)
	})
}

type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

