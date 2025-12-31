# Build and Architecture Guide

## Package Structure

```
internal/initialization/
├── bootstrap.go      # Main bootstrap orchestration
├── readme.md         # Package documentation
└── BUILD.md          # This file - architecture guide
```

## Design Principles

### 1. Single Responsibility
Each method in the `Bootstrap` struct has a single, well-defined responsibility:
- `ensureAdminUser`: Only handles admin user creation
- `ensureDefaultProfile`: Only handles profile creation
- `initializeProfileSchema`: Only handles schema initialization
- `verifyConnections`: Only handles connection verification

### 2. Dependency Injection
The `Bootstrap` struct accepts its dependencies (queries, logger) through the constructor, making it:
- Testable: Easy to mock dependencies
- Flexible: Can be configured with different implementations
- Maintainable: Dependencies are explicit

### 3. Error Handling
Comprehensive error handling at each level:
- Methods return errors that can be handled by callers
- Detailed logging at each step
- Graceful degradation when possible

### 4. Idempotency
All operations are idempotent - safe to run multiple times:
- Checks for existence before creating
- Uses `IF NOT EXISTS` in SQL where possible
- Handles "already exists" errors gracefully

### 5. Extensibility
Easy to extend with new initialization steps:
1. Add a new method following existing patterns
2. Call it from `Initialize()` in the correct order
3. Update documentation

## Initialization Flow

```
Initialize()
    │
    ├──> ensureAdminUser()
    │       ├── Check if admin user exists
    │       ├── Create if missing (bcrypt hash password)
    │       └── Return admin user
    │
    ├──> ensureDefaultProfile(adminUser)
    │       ├── Check if default profile exists
    │       ├── Create if missing
    │       │   ├── Get default DSN and MCP config
    │       │   ├── Link to admin user
    │       │   └── Create default models
    │       └── Return default profile
    │
    ├──> initializeProfileSchema(profile)
    │       ├── Find schema file
    │       ├── Parse SQL statements
    │       └── Execute statements (handle already exists)
    │
    └──> verifyConnections(profile)
            ├── verifyPostgreSQLConnection()
            │   ├── Ping database
            │   └── Check NeuronDB extension
            └── verifyMCPConnection()
                └── Log MCP verification (deferred to avoid circular deps)
```

## Testing Strategy

### Unit Tests
Each method can be tested independently:
```go
func TestEnsureAdminUser(t *testing.T) {
    // Mock queries
    mockQueries := &MockQueries{}
    mockLogger := &MockLogger{}
    
    bootstrap := NewBootstrap(mockQueries, mockLogger)
    
    // Test admin user creation
    user, err := bootstrap.ensureAdminUser(context.Background())
    // Assertions...
}
```

### Integration Tests
Test the full initialization flow:
```go
func TestBootstrap_Initialize(t *testing.T) {
    // Setup test database
    db := setupTestDB(t)
    queries := db.NewQueries(db)
    logger := logging.NewTestLogger()
    
    bootstrap := NewBootstrap(queries, logger)
    err := bootstrap.Initialize(context.Background())
    
    // Verify admin user exists
    // Verify default profile exists
    // Verify schema is initialized
}
```

## Extension Points

### Adding New Initialization Steps

1. **Add Method**:
```go
func (b *Bootstrap) initializeCustomFeature(ctx context.Context) error {
    b.logger.Info("Initializing custom feature", nil)
    // Implementation
    return nil
}
```

2. **Integrate into Flow**:
```go
func (b *Bootstrap) Initialize(ctx context.Context) error {
    // ... existing steps ...
    
    // New step
    if err := b.initializeCustomFeature(ctx); err != nil {
        return fmt.Errorf("failed to initialize custom feature: %w", err)
    }
    
    // ... rest of steps ...
}
```

### Adding New Verification Steps

1. **Add Verification Method**:
```go
func (b *Bootstrap) verifyCustomConnection(ctx context.Context, profile *db.Profile) {
    b.logger.Info("Verifying custom connection", nil)
    // Implementation
}
```

2. **Call from verifyConnections**:
```go
func (b *Bootstrap) verifyConnections(ctx context.Context, profile *db.Profile) {
    // ... existing verifications ...
    b.verifyCustomConnection(ctx, profile)
}
```

## Performance Considerations

- **Timeouts**: All operations use context with timeouts
- **Lazy Initialization**: Only initializes what's needed
- **Error Recovery**: Continues when non-critical errors occur
- **Logging**: Detailed logging without impacting performance

## Future Improvements

1. **Parallel Initialization**: Some independent steps could run in parallel
2. **Progress Reporting**: Add progress callbacks for long-running operations
3. **Rollback Support**: Add rollback capability for failed initializations
4. **Configuration**: Make initialization steps configurable
5. **Health Checks**: Periodic health checks after initialization






