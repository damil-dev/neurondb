package initialization

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"time"

	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/logging"
	"github.com/neurondb/NeuronDesktop/api/internal/utils"
	"golang.org/x/crypto/bcrypt"
)

// Bootstrap handles all application initialization tasks
type Bootstrap struct {
	queries   *db.Queries
	logger    *logging.Logger
	validator *Validator
}

// NewBootstrap creates a new bootstrap instance
func NewBootstrap(queries *db.Queries, logger *logging.Logger) *Bootstrap {
	return &Bootstrap{
		queries:   queries,
		logger:    logger,
		validator: NewValidator(logger),
	}
}

// Initialize performs all initialization tasks in the correct order
func (b *Bootstrap) Initialize(ctx context.Context) error {
	metrics := NewBootstrapMetrics()
	defer metrics.Finish()
	defer metrics.LogMetrics(b.logger)

	b.logger.Info("Starting application bootstrap sequence", nil)

	// Step 1: Ensure default admin user exists (with retry)
	stepStart := time.Now()
	adminUser, err := b.ensureAdminUserWithRetry(ctx)
	if err != nil {
		metrics.TrackStep("admin_user", time.Since(stepStart), false)
		return fmt.Errorf("failed to ensure admin user: %w", err)
	}
	metrics.TrackStep("admin_user", time.Since(stepStart), true)

	// Step 2: Validate admin user
	validationStart := time.Now()
	if validation := b.validator.ValidateAdminUser(ctx, b.queries); !validation.Valid {
		b.logger.Info("Admin user validation failed", map[string]interface{}{
			"errors": validation.Errors,
		})
		for _, errMsg := range validation.Errors {
			b.logger.Error("Validation error", fmt.Errorf(errMsg), nil)
		}
		metrics.TrackStep("validation", time.Since(validationStart), false)
	} else {
		metrics.TrackStep("validation", time.Since(validationStart), true)
	}

	// Step 3: Ensure default profile exists (with retry)
	profileStart := time.Now()
	defaultProfile, err := b.ensureDefaultProfileWithRetry(ctx, adminUser)
	if err != nil {
		metrics.TrackStep("profile", time.Since(profileStart), false)
		return fmt.Errorf("failed to ensure default profile: %w", err)
	}
	metrics.TrackStep("profile", time.Since(profileStart), true)

	// Step 4: Initialize profile database schema (with retry)
	if defaultProfile != nil {
		schemaStart := time.Now()
		if err := b.initializeProfileSchemaWithRetry(ctx, defaultProfile); err != nil {
			b.logger.Info("Warning: Failed to initialize profile schema", map[string]interface{}{
				"error": err.Error(),
			})
			metrics.TrackStep("schema", time.Since(schemaStart), false)
		} else {
			metrics.TrackStep("schema", time.Since(schemaStart), true)
		}
	}

	// Step 5: Validate profile
	if defaultProfile != nil {
		if validation := b.validator.ValidateProfile(ctx, b.queries); !validation.Valid {
			b.logger.Info("Profile validation failed", map[string]interface{}{
				"errors": validation.Errors,
			})
		}
	}

	// Step 6: Verify connections
	if defaultProfile != nil {
		b.verifyConnections(ctx, defaultProfile)
	}

	// Step 7: Perform health check
	healthStart := time.Now()
	healthChecker := NewHealthChecker(b.queries, b.logger)
	healthStatus := healthChecker.CheckAll(ctx)
	metrics.TrackStep("health_check", time.Since(healthStart), healthStatus.Overall)
	if !healthStatus.Overall {
		b.logger.Info("Health check completed with issues", map[string]interface{}{
			"status": healthStatus.Status,
			"checks": healthStatus.Checks,
		})
	} else {
		b.logger.Info("Health check passed", map[string]interface{}{
			"status": healthStatus.Status,
		})
	}

	b.logger.Info("Application bootstrap completed successfully", map[string]interface{}{
		"total_duration": metrics.Duration.String(),
	})
	return nil
}

// ensureAdminUserWithRetry ensures admin user exists with retry logic
func (b *Bootstrap) ensureAdminUserWithRetry(ctx context.Context) (*db.User, error) {
	var adminUser *db.User
	var err error

	retryFunc := func(ctx context.Context) error {
		adminUser, err = b.ensureAdminUser(ctx)
		return err
	}

	if err := RetryWithBackoff(ctx, b.logger, "ensure admin user", retryFunc); err != nil {
		return nil, err
	}

	return adminUser, nil
}

// ensureDefaultProfileWithRetry ensures default profile exists with retry logic
func (b *Bootstrap) ensureDefaultProfileWithRetry(ctx context.Context, adminUser *db.User) (*db.Profile, error) {
	var defaultProfile *db.Profile
	var err error

	retryFunc := func(ctx context.Context) error {
		defaultProfile, err = b.ensureDefaultProfile(ctx, adminUser)
		return err
	}

	if err := RetryWithBackoff(ctx, b.logger, "ensure default profile", retryFunc); err != nil {
		return nil, err
	}

	return defaultProfile, nil
}

// initializeProfileSchemaWithRetry initializes schema with retry logic
func (b *Bootstrap) initializeProfileSchemaWithRetry(ctx context.Context, profile *db.Profile) error {
	retryFunc := func(ctx context.Context) error {
		return b.initializeProfileSchema(ctx, profile)
	}

	return RetryWithBackoff(ctx, b.logger, "initialize profile schema", retryFunc)
}

// ensureAdminUser ensures the default admin user exists
func (b *Bootstrap) ensureAdminUser(ctx context.Context) (*db.User, error) {
	b.logger.Info("Checking for default admin user", nil)

	adminUser, err := b.queries.GetUserByUsername(ctx, "admin")
	if err != nil {
		// Admin user doesn't exist, create it
		b.logger.Info("Creating default admin user", nil)

		// Get admin password from environment or generate a random one
		adminPassword := os.Getenv("ADMIN_PASSWORD")
		if adminPassword == "" {
			// Generate a random password and log it (one-time setup)
			adminPassword = fmt.Sprintf("admin-%d", time.Now().Unix())
			b.logger.Info("⚠️  ADMIN_PASSWORD not set - using temporary password", map[string]interface{}{
				"password": adminPassword,
				"warning":  "Please set ADMIN_PASSWORD environment variable and change this password immediately",
			})
		}

		passwordHash, hashErr := bcrypt.GenerateFromPassword([]byte(adminPassword), bcrypt.DefaultCost)
		if hashErr != nil {
			b.logger.Error("Failed to hash admin password", hashErr, nil)
			return nil, fmt.Errorf("failed to hash admin password: %w", hashErr)
		}

		adminUser = &db.User{
			Username:     "admin",
			IsAdmin:      true,
			PasswordHash: string(passwordHash),
		}

		if createErr := b.queries.CreateUser(ctx, adminUser); createErr != nil {
			b.logger.Error("Failed to create admin user", createErr, nil)
			return nil, fmt.Errorf("failed to create admin user: %w", createErr)
		}

		b.logger.Info("Default admin user created successfully", map[string]interface{}{
			"username": "admin",
			"user_id":  adminUser.ID,
		})
		return adminUser, nil
	}

	b.logger.Info("Admin user already exists", map[string]interface{}{
		"username": "admin",
		"user_id":  adminUser.ID,
	})
	return adminUser, nil
}

// ensureDefaultProfile ensures the default profile exists
func (b *Bootstrap) ensureDefaultProfile(ctx context.Context, adminUser *db.User) (*db.Profile, error) {
	b.logger.Info("Checking for default profile", nil)

	defaultProfile, err := b.queries.GetDefaultProfile(ctx)
	if err != nil || defaultProfile == nil {
		b.logger.Info("Creating default profile", nil)

		// Determine user ID for profile
		userID := "default"
		if adminUser != nil {
			userID = adminUser.ID
		}

		// Get default configuration
		neurondbDSN := utils.GetDefaultNeuronDBDSN()
		mcpConfig := utils.GetDefaultMCPConfig()

		// Hash password for admin profile (use same as admin user password)
		adminPassword := os.Getenv("ADMIN_PASSWORD")
		if adminPassword == "" {
			adminPassword = fmt.Sprintf("admin-%d", time.Now().Unix())
		}
		profilePasswordHash, hashErr := bcrypt.GenerateFromPassword([]byte(adminPassword), bcrypt.DefaultCost)
		if hashErr != nil {
			b.logger.Error("Failed to hash admin profile password", hashErr, nil)
			return nil, fmt.Errorf("failed to hash admin profile password: %w", hashErr)
		}

		defaultProfile = &db.Profile{
			UserID:          userID,
			Name:            "admin",
			ProfileUsername: "admin",
			ProfilePassword: string(profilePasswordHash),
			NeuronDBDSN:     neurondbDSN,
			MCPConfig:       mcpConfig,
			IsDefault:       true,
		}

		if err := b.queries.CreateProfile(ctx, defaultProfile); err != nil {
			b.logger.Error("Failed to create default profile", err, nil)
			return nil, fmt.Errorf("failed to create default profile: %w", err)
		}

		if err := b.queries.SetDefaultProfile(ctx, defaultProfile.ID); err != nil {
			b.logger.Error("Failed to set default profile", err, nil)
			return nil, fmt.Errorf("failed to set default profile: %w", err)
		}

		// Create default model configurations
		if err := utils.CreateDefaultModelsForProfile(ctx, b.queries, defaultProfile.ID); err != nil {
			b.logger.Error("Failed to create default models", err, map[string]interface{}{
				"profile_id": defaultProfile.ID,
			})
		} else {
			b.logger.Info("Default models created successfully", map[string]interface{}{
				"profile_id": defaultProfile.ID,
			})
		}

		b.logger.Info("Default profile created successfully", map[string]interface{}{
			"profile_id": defaultProfile.ID,
			"name":       defaultProfile.Name,
			"user_id":    defaultProfile.UserID,
		})

		// Reload to get full profile data
		defaultProfile, _ = b.queries.GetDefaultProfile(ctx)
		return defaultProfile, nil
	}

	b.logger.Info("Default profile already exists", map[string]interface{}{
		"profile_id": defaultProfile.ID,
		"name":       defaultProfile.Name,
	})
	return defaultProfile, nil
}

// initializeProfileSchema initializes the database schema for a profile
func (b *Bootstrap) initializeProfileSchema(ctx context.Context, profile *db.Profile) error {
	b.logger.Info("Initializing database schema for profile", map[string]interface{}{
		"profile_id": profile.ID,
		"dsn":        profile.NeuronDBDSN,
	})

	if err := utils.InitSchema(ctx, profile.NeuronDBDSN); err != nil {
		return fmt.Errorf("failed to initialize schema: %w", err)
	}

	b.logger.Info("Schema initialized successfully for profile database", map[string]interface{}{
		"profile_id": profile.ID,
	})
	return nil
}

// verifyConnections verifies connections for a profile
func (b *Bootstrap) verifyConnections(ctx context.Context, profile *db.Profile) {
	b.logger.Info("Verifying connections for default profile", map[string]interface{}{
		"profile_id": profile.ID,
	})

	// Verify PostgreSQL (NeuronDB) connection
	b.verifyPostgreSQLConnection(ctx, profile)

	// Verify MCP server connection
	if profile.MCPConfig != nil {
		b.verifyMCPConnection(ctx, profile)
	}
}

// verifyPostgreSQLConnection verifies the PostgreSQL connection
func (b *Bootstrap) verifyPostgreSQLConnection(ctx context.Context, profile *db.Profile) {
	b.logger.Info("Verifying PostgreSQL (NeuronDB) connection", map[string]interface{}{
		"dsn": profile.NeuronDBDSN,
	})

	conn, err := sql.Open("pgx", profile.NeuronDBDSN)
	if err != nil {
		b.logger.Info("⚠ Failed to open PostgreSQL (NeuronDB) connection", map[string]interface{}{
			"error": err.Error(),
		})
		return
	}
	defer conn.Close()

	testCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	if err := conn.PingContext(testCtx); err != nil {
		b.logger.Info("⚠ PostgreSQL (NeuronDB) connection failed", map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	b.logger.Info("✓ PostgreSQL (NeuronDB) connection verified", map[string]interface{}{
		"dsn": profile.NeuronDBDSN,
	})

	// Test NeuronDB extension (optional - may not be installed)
	var version string
	if err := conn.QueryRowContext(testCtx, "SELECT neurondb.version()").Scan(&version); err == nil {
		b.logger.Info("✓ NeuronDB extension verified", map[string]interface{}{
			"version": version,
		})
	} else {
		b.logger.Info("⚠ NeuronDB extension not found (database may not have extension installed)", nil)
	}
}

// verifyMCPConnection verifies the MCP server connection
func (b *Bootstrap) verifyMCPConnection(ctx context.Context, profile *db.Profile) {
	// This is a placeholder - MCP verification would require MCPManager
	// For now, just log that we would verify it
	b.logger.Info("MCP connection verification", map[string]interface{}{
		"command": profile.MCPConfig["command"],
	})
	// TODO: Implement MCP connection verification using MCPManager
}
