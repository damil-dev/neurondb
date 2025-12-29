package middleware

import (
	"fmt"
	"net/http"
	"sync"
	"time"
)

// RateLimiter provides rate limiting functionality
type RateLimiter struct {
	requests map[string][]time.Time
	mu       sync.RWMutex
	limit    int
	window   time.Duration
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(limit int, window time.Duration) *RateLimiter {
	rl := &RateLimiter{
		requests: make(map[string][]time.Time),
		limit:    limit,
		window:   window,
	}

	// Cleanup old entries periodically
	go rl.cleanup()

	return rl
}

// Allow checks if a request is allowed
func (rl *RateLimiter) Allow(key string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	cutoff := now.Add(-rl.window)

	// Get or create request history
	history, exists := rl.requests[key]
	if !exists {
		rl.requests[key] = []time.Time{now}
		return true
	}

	// Remove old requests
	filtered := make([]time.Time, 0)
	for _, t := range history {
		if t.After(cutoff) {
			filtered = append(filtered, t)
		}
	}

	// Check if limit exceeded
	if len(filtered) >= rl.limit {
		return false
	}

	// Add current request
	filtered = append(filtered, now)
	rl.requests[key] = filtered

	return true
}

// Remaining returns the number of remaining requests
func (rl *RateLimiter) Remaining(key string) int {
	rl.mu.RLock()
	defer rl.mu.RUnlock()

	now := time.Now()
	cutoff := now.Add(-rl.window)

	history, exists := rl.requests[key]
	if !exists {
		return rl.limit
	}

	count := 0
	for _, t := range history {
		if t.After(cutoff) {
			count++
		}
	}

	return rl.limit - count
}

// Reset resets the rate limit for a key
func (rl *RateLimiter) Reset(key string) {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	delete(rl.requests, key)
}

// cleanup removes old entries periodically
func (rl *RateLimiter) cleanup() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		rl.mu.Lock()
		now := time.Now()
		cutoff := now.Add(-rl.window * 2) // Keep entries for 2x window

		for key, history := range rl.requests {
			allOld := true
			for _, t := range history {
				if t.After(cutoff) {
					allOld = false
					break
				}
			}
			if allOld {
				delete(rl.requests, key)
			}
		}
		rl.mu.Unlock()
	}
}

// RateLimitMiddleware provides rate limiting middleware
func RateLimitMiddleware(limiter *RateLimiter) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Skip rate limiting for lightweight auth verification endpoint
			// This endpoint is called immediately after login to verify the token works
			if r.URL.Path == "/api/v1/auth/me" {
				next.ServeHTTP(w, r)
				return
			}

			// Get rate limit key (user ID from context, session ID, or IP)
			key := r.RemoteAddr
			
			// Try to get user ID from context (set by session or JWT middleware)
			// This is set by both session middleware and JWT middleware
			if userID, ok := r.Context().Value("user_id").(string); ok && userID != "" {
				key = "user:" + userID
			} else if authHeader := r.Header.Get("Authorization"); authHeader != "" {
				// For JWT/API key, use a hash of the token (not the full token)
				// Extract just the token part and hash it
				if len(authHeader) > 7 && authHeader[:7] == "Bearer " {
					token := authHeader[7:]
					// Use first 16 chars as key (safe for rate limiting)
					if len(token) > 16 {
						key = "token:" + token[:16]
					} else {
						key = "token:" + token
					}
				} else {
					key = "auth:" + authHeader[:min(32, len(authHeader))]
				}
			}

			if !limiter.Allow(key) {
				w.Header().Set("X-RateLimit-Limit", "100")
				w.Header().Set("X-RateLimit-Remaining", "0")
				w.Header().Set("X-RateLimit-Reset", time.Now().Add(1*time.Minute).Format(time.RFC1123))
				w.Header().Set("Retry-After", "60")
				http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
				return
			}

			remaining := limiter.Remaining(key)
			w.Header().Set("X-RateLimit-Limit", "100")
			w.Header().Set("X-RateLimit-Remaining", fmt.Sprintf("%d", remaining))
			w.Header().Set("X-RateLimit-Reset", time.Now().Add(1*time.Minute).Format(time.RFC1123))

			next.ServeHTTP(w, r)
		})
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

