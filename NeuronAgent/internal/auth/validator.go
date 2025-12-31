/*-------------------------------------------------------------------------
 *
 * validator.go
 *    Rate limiting and request validation for NeuronAgent
 *
 * Provides thread-safe rate limiting with per-key tracking and
 * automatic reset based on time windows.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/auth/validator.go
 *
 *-------------------------------------------------------------------------
 */

package auth

import (
	"sync"
	"time"
)

type RateLimiter struct {
	limits map[string]*rateLimit
	mu     sync.RWMutex
}

type rateLimit struct {
	count     int
	resetTime time.Time
}

func NewRateLimiter() *RateLimiter {
	return &RateLimiter{
		limits: make(map[string]*rateLimit),
	}
}

func (r *RateLimiter) CheckLimit(keyID string, limitPerMin int) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	now := time.Now()
	rl, exists := r.limits[keyID]

	if !exists || now.After(rl.resetTime) {
		/* Reset or create */
		r.limits[keyID] = &rateLimit{
			count:     1,
			resetTime: now.Add(1 * time.Minute),
		}
		return true
	}

	if rl.count >= limitPerMin {
		return false
	}

	rl.count++
	return true
}

/* HasRole and RequireRole are now in roles.go */
