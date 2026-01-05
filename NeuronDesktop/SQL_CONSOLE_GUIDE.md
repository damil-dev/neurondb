# Enabling SQL Console in NeuronDesktop

## Issue
When trying to use the SQL Query Editor in NeuronDesktop, you see:
```
Error
SQL console is disabled
```

## Why is it Disabled?

The SQL console is **disabled by default** for security reasons because it allows **arbitrary SQL execution** with full database access. This could be dangerous in production environments.

## How to Enable SQL Console

### Method 1: Environment Variable (Recommended)

Start the NeuronDesktop API with the `ENABLE_SQL_CONSOLE` environment variable:

```bash
cd /home/pge/pge/neurondb/NeuronDesktop/api

# Build if needed
go build -o bin/neurondesk-api ./cmd/server

# Start with SQL console enabled
ENABLE_SQL_CONSOLE=true \
JWT_SECRET="neurondesktop-super-secret-jwt-key-change-in-production" \
DB_HOST=localhost \
DB_PORT=5432 \
DB_USER=pge \
DB_PASSWORD=test \
DB_NAME=neurondb \
./bin/neurondesk-api
```

### Method 2: Using nohup (Background Process)

```bash
cd /home/pge/pge/neurondb/NeuronDesktop/api

# Kill existing process if running
pkill -f neurondesk-api

# Start in background with SQL console enabled
ENABLE_SQL_CONSOLE=true \
JWT_SECRET="neurondesktop-super-secret-jwt-key-change-in-production" \
DB_HOST=localhost \
DB_PORT=5432 \
DB_USER=pge \
DB_PASSWORD=test \
DB_NAME=neurondb \
nohup ./bin/neurondesk-api > /tmp/neurondesk-api.log 2>&1 &

# Check if it's running
ps aux | grep neurondesk-api | grep -v grep

# Check logs
tail -f /tmp/neurondesk-api.log
```

### Method 3: Docker Compose

If using Docker, edit `docker-compose.yml`:

```yaml
services:
  api:
    environment:
      - ENABLE_SQL_CONSOLE=true  # Add this line
      - JWT_SECRET=...
      - DB_HOST=...
      # ... other env vars
```

Then restart:
```bash
cd /home/pge/pge/neurondb/NeuronDesktop
docker-compose down
docker-compose up -d
```

### Method 4: Create .env File

Create a `.env` file in the API directory:

```bash
cat > /home/pge/pge/neurondb/NeuronDesktop/api/.env << 'EOF'
ENABLE_SQL_CONSOLE=true
JWT_SECRET=neurondesktop-super-secret-jwt-key-change-in-production
DB_HOST=localhost
DB_PORT=5432
DB_USER=pge
DB_PASSWORD=test
DB_NAME=neurondb
EOF
```

Then modify `cmd/server/main.go` to load from `.env` file (if not already implemented).

## Current Status

✅ **SQL Console is NOW ENABLED**

The API has been restarted with `ENABLE_SQL_CONSOLE=true` and is running in the background.

You can now:
1. Refresh your NeuronDesktop web page
2. Go to the NeuronDB page
3. Use the SQL Query Editor
4. Execute queries like `CREATE TABLE test (a int);`

## Verification

To verify SQL console is enabled:

```bash
# Check API health
curl http://localhost:8081/health

# Try to execute SQL (requires authentication token)
curl -X POST http://localhost:8081/api/v1/factory/sql \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"sql": "SELECT version();"}'
```

## Security Considerations

⚠️ **Important Security Notes:**

1. **Development Only** - Only enable SQL console in development/testing environments
2. **Production Risk** - Never enable in production without proper access controls
3. **Admin Only** - Even when enabled, SQL console requires admin user privileges
4. **Audit Logging** - All SQL queries are logged for security audit trails
5. **Input Validation** - The API still performs input validation and sanitization

## Troubleshooting

### 1. Still seeing "SQL console is disabled"?

- **Solution**: Restart the API server with the environment variable set
- **Check**: The API must be restarted for the change to take effect

### 2. Getting permission errors?

- **Solution**: Make sure you're logged in as an admin user
- **Check**: SQL console requires both:
  1. `ENABLE_SQL_CONSOLE=true` environment variable
  2. Admin user authentication

### 3. API not starting?

- **Check logs**: `tail -f /tmp/neurondesk-api.log`
- **Check database**: Ensure PostgreSQL is running
- **Check port**: Make sure port 8081 is not in use

```bash
# Check if port is in use
lsof -i :8081

# Kill process using the port
pkill -f neurondesk-api
```

## Code References

The SQL console functionality is implemented in:

- **Config**: `NeuronDesktop/api/internal/config/config.go:83`
- **Handler**: `NeuronDesktop/api/internal/handlers/neurondb.go:155`
- **Main**: `NeuronDesktop/api/cmd/server/main.go:125`

The check happens in the `ExecuteSQLFull` handler:

```go
if !h.enableSQLConsole {
    WriteError(w, r, http.StatusForbidden, 
        fmt.Errorf("SQL console is disabled"), nil)
    return
}
```

## Usage Examples

Once enabled, you can execute any SQL query:

### Create Table
```sql
CREATE TABLE test (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100)
);
```

### Insert Data
```sql
INSERT INTO test (name) VALUES ('Alice'), ('Bob');
```

### Query Data
```sql
SELECT * FROM test;
```

### Vector Operations (NeuronDB)
```sql
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    embedding vector(1536)
);

INSERT INTO embeddings (embedding) 
VALUES ('[0.1, 0.2, ...]'::vector);
```

## Quick Restart Command

For easy copy-paste to restart with SQL console enabled:

```bash
pkill -f neurondesk-api && sleep 1 && \
cd /home/pge/pge/neurondb/NeuronDesktop/api && \
ENABLE_SQL_CONSOLE=true \
JWT_SECRET="neurondesktop-super-secret-jwt-key-change-in-production" \
DB_HOST=localhost DB_PORT=5432 DB_USER=pge DB_PASSWORD=test DB_NAME=neurondb \
nohup ./bin/neurondesk-api > /tmp/neurondesk-api.log 2>&1 & \
echo "✅ API restarted with SQL console ENABLED"
```

---

*Last updated: December 30, 2025*







