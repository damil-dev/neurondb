package initialization

import (
	"context"
	"fmt"
	"time"

	"github.com/neurondb/NeuronDesktop/api/internal/logging"
)

// RetryConfig defines retry behavior
type RetryConfig struct {
	MaxAttempts  int
	InitialDelay time.Duration
	MaxDelay     time.Duration
	Multiplier   float64
}

// DefaultRetryConfig returns a sensible default retry configuration
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts:  3,
		InitialDelay: 1 * time.Second,
		MaxDelay:     10 * time.Second,
		Multiplier:   2.0,
	}
}

// RetryableFunc is a function that can be retried
type RetryableFunc func(ctx context.Context) error

// Retry executes a function with retry logic
func Retry(ctx context.Context, logger *logging.Logger, config RetryConfig, operation string, fn RetryableFunc) error {
	var lastErr error
	delay := config.InitialDelay

	for attempt := 1; attempt <= config.MaxAttempts; attempt++ {
		logger.Info(fmt.Sprintf("Attempting %s (attempt %d/%d)", operation, attempt, config.MaxAttempts), nil)

		attemptErr := fn(ctx)
		if attemptErr == nil {
			if attempt > 1 {
				logger.Info(fmt.Sprintf("%s succeeded after %d attempts", operation, attempt), nil)
			}
			return nil
		}

		lastErr = attemptErr
		logger.Info(fmt.Sprintf("%s failed (attempt %d/%d): %v", operation, attempt, config.MaxAttempts, attemptErr), nil)

		// Don't wait after the last attempt
		if attempt < config.MaxAttempts {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(delay):
				// Calculate next delay with exponential backoff
				delay = time.Duration(float64(delay) * config.Multiplier)
				if delay > config.MaxDelay {
					delay = config.MaxDelay
				}
			}
		}
	}

	return fmt.Errorf("%s failed after %d attempts: %w", operation, config.MaxAttempts, lastErr)
}

// RetryWithBackoff executes a function with exponential backoff retry
func RetryWithBackoff(ctx context.Context, logger *logging.Logger, operation string, fn RetryableFunc) error {
	return Retry(ctx, logger, DefaultRetryConfig(), operation, fn)
}
