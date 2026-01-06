# Utils Package

This package contains utility functions and helpers used throughout the NeuronDesktop API.

## Packages

### Schema Manager (`schema_manager.go`)

Manages database schema initialization and execution.

**Features:**
- Finds schema file from multiple possible locations
- Parses SQL into individual statements
- Executes statements with error handling
- Handles "already exists" errors gracefully
- Provides both functional and object-oriented interfaces

**Usage:**
```go
// Functional interface (backwards compatible)
err := utils.InitSchema(ctx, dsn)

// Object-oriented interface
manager := utils.NewSchemaManager()
err := manager.Initialize(ctx, dsn)
```

**Key Methods:**
- `NewSchemaManager()`: Creates a new schema manager instance
- `InitSchema(ctx, dsn)`: Functional interface for schema initialization
- `Initialize(ctx, dsn)`: Object-oriented interface
- `executeSchema(ctx, db, sql)`: Executes SQL statements
- `splitSQL(sql)`: Splits SQL into individual statements

### Default Models (`default_models.go`)

Manages default AI model configurations for profiles.

**Features:**
- Pre-configured list of popular AI models
- Automatic model creation for new profiles
- Support for multiple providers (OpenAI, Anthropic, Google, Ollama)

**Usage:**
```go
err := utils.CreateDefaultModelsForProfile(ctx, queries, profileID)
```

**Default Models:**
- GPT-4o (OpenAI) - Default
- GPT-4 Turbo (OpenAI)
- GPT-3.5 Turbo (OpenAI)
- Claude 3.5 Sonnet (Anthropic)
- Claude 3 Opus (Anthropic)
- Gemini Pro (Google)
- Llama 3 (Ollama)
- Mistral (Ollama)

### MCP Detection (`mcp_detection.go`)

Detects and configures NeuronMCP binary locations.

**Features:**
- Cross-platform detection (Linux, macOS, Windows)
- Multiple search paths
- Environment variable support
- Default configuration generation

**Usage:**
```go
mcpConfig := utils.GetDefaultMCPConfig()
dsn := utils.GetDefaultNeuronDBDSN()
```

### Validation (`validation.go`)

Provides validation utilities for various data types.

**Features:**
- Profile validation
- DSN validation
- Configuration validation

## Design Principles

1. **Modularity**: Each utility is self-contained and focused
2. **Reusability**: Functions can be used across the codebase
3. **Error Handling**: Comprehensive error handling and reporting
4. **Documentation**: Well-documented with examples
5. **Testing**: Utilities are easily testable








