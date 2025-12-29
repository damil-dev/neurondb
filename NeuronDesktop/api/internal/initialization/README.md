# Initialization Package

This package handles all application initialization and bootstrap logic in a modular, organized manner.

## Overview

The `initialization` package provides a clean, modular approach to application startup. It encapsulates all bootstrap logic into a single, well-structured component that can be easily tested, maintained, and extended.

## Architecture

### Bootstrap

The `Bootstrap` struct is the main entry point for application initialization. It orchestrates all startup tasks in the correct order:

1. **User Initialization**: Ensures the default admin user exists
2. **Profile Initialization**: Creates and configures the default profile
3. **Schema Initialization**: Sets up database schemas
4. **Connection Verification**: Validates all external connections

### Key Features

- **Modular Design**: Each initialization step is separated into its own method
- **Error Handling**: Comprehensive error handling with detailed logging
- **Idempotent**: Safe to run multiple times - won't recreate existing resources
- **Extensible**: Easy to add new initialization steps
- **Testable**: Each component can be tested independently

## Usage

```go
// Create bootstrap instance
bootstrap := initialization.NewBootstrap(queries, logger)

// Initialize application
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

if err := bootstrap.Initialize(ctx); err != nil {
    log.Fatal("Failed to initialize application:", err)
}
```

## Initialization Steps

### 1. Admin User Creation

Ensures a default admin user exists with:
- Username: `admin`
- Password: `neurondb` (bcrypt hashed)
- Automatic creation on first startup

### 2. Default Profile Creation

Creates a default profile with:
- Linked to admin user
- Auto-detected NeuronDB DSN
- Auto-detected MCP configuration
- Default model configurations

### 3. Schema Initialization

Initializes complete database schema:
- Users, Profiles, API Keys
- Model Configurations
- Request Logs
- App Settings

### 4. Connection Verification

Verifies all external connections:
- PostgreSQL/NeuronDB connection
- NeuronDB extension availability
- MCP server connectivity

## Extending

To add new initialization steps:

1. Add a new method to `Bootstrap`
2. Call it from `Initialize()` in the appropriate order
3. Follow the existing patterns for error handling and logging

Example:

```go
func (b *Bootstrap) initializeCustomFeature(ctx context.Context) error {
    b.logger.Info("Initializing custom feature", nil)
    // Your initialization logic
    return nil
}
```

## Testing

Each initialization method can be tested independently by mocking the `queries` and `logger` dependencies.

