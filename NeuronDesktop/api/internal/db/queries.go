package db

import (
	"context"
	"database/sql"
	"encoding/json"
	"time"

	"github.com/google/uuid"
	_ "github.com/jackc/pgx/v5/stdlib"
)

// Queries provides database operations
type Queries struct {
	db *sql.DB
}

// NewQueries creates a new Queries instance
func NewQueries(db *sql.DB) *Queries {
	return &Queries{db: db}
}

// Profile operations

// CreateProfile creates a new profile
func (q *Queries) CreateProfile(ctx context.Context, profile *Profile) error {
	if profile.ID == "" {
		profile.ID = uuid.New().String()
	}
	if profile.CreatedAt.IsZero() {
		profile.CreatedAt = time.Now()
	}
	profile.UpdatedAt = time.Now()

	mcpConfigJSON, _ := json.Marshal(profile.MCPConfig)

	query := `
		INSERT INTO profiles (id, name, user_id, mcp_config, neurondb_dsn, agent_endpoint, agent_api_key, default_collection, is_default, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
	`
	_, err := q.db.ExecContext(ctx, query,
		profile.ID, profile.Name, profile.UserID, mcpConfigJSON,
		profile.NeuronDBDSN, profile.AgentEndpoint, profile.AgentAPIKey,
		profile.DefaultCollection, profile.IsDefault, profile.CreatedAt, profile.UpdatedAt)
	return err
}

// GetProfile gets a profile by ID
func (q *Queries) GetProfile(ctx context.Context, id string) (*Profile, error) {
	var profile Profile
	var mcpConfigJSON []byte
	var agentEndpoint, agentAPIKey, defaultCollection sql.NullString

	query := `
		SELECT id, name, user_id, mcp_config, neurondb_dsn, agent_endpoint, agent_api_key, default_collection, is_default, created_at, updated_at
		FROM profiles
		WHERE id = $1
	`
	err := q.db.QueryRowContext(ctx, query, id).Scan(
		&profile.ID, &profile.Name, &profile.UserID, &mcpConfigJSON,
		&profile.NeuronDBDSN, &agentEndpoint, &agentAPIKey,
		&defaultCollection, &profile.IsDefault, &profile.CreatedAt, &profile.UpdatedAt)
	if err != nil {
		return nil, err
	}

	if agentEndpoint.Valid {
		profile.AgentEndpoint = agentEndpoint.String
	}
	if agentAPIKey.Valid {
		profile.AgentAPIKey = agentAPIKey.String
	}
	if defaultCollection.Valid {
		profile.DefaultCollection = defaultCollection.String
	}

	if len(mcpConfigJSON) > 0 {
		json.Unmarshal(mcpConfigJSON, &profile.MCPConfig)
	}

	return &profile, nil
}

// ListProfiles lists profiles for a user
func (q *Queries) ListProfiles(ctx context.Context, userID string) ([]Profile, error) {
	query := `
		SELECT id, name, user_id, mcp_config, neurondb_dsn, agent_endpoint, agent_api_key, default_collection, is_default, created_at, updated_at
		FROM profiles
		WHERE user_id = $1
		ORDER BY is_default DESC, created_at DESC
	`
	rows, err := q.db.QueryContext(ctx, query, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var profiles []Profile
	for rows.Next() {
		var profile Profile
		var mcpConfigJSON []byte
		var agentEndpoint, agentAPIKey, defaultCollection sql.NullString

		if err := rows.Scan(
			&profile.ID, &profile.Name, &profile.UserID, &mcpConfigJSON,
			&profile.NeuronDBDSN, &agentEndpoint, &agentAPIKey,
			&defaultCollection, &profile.IsDefault, &profile.CreatedAt, &profile.UpdatedAt); err != nil {
			continue
		}

		if agentEndpoint.Valid {
			profile.AgentEndpoint = agentEndpoint.String
		}
		if agentAPIKey.Valid {
			profile.AgentAPIKey = agentAPIKey.String
		}
		if defaultCollection.Valid {
			profile.DefaultCollection = defaultCollection.String
		}

		if len(mcpConfigJSON) > 0 {
			json.Unmarshal(mcpConfigJSON, &profile.MCPConfig)
		}

		profiles = append(profiles, profile)
	}

	return profiles, nil
}

// UpdateProfile updates a profile
func (q *Queries) UpdateProfile(ctx context.Context, profile *Profile) error {
	profile.UpdatedAt = time.Now()
	mcpConfigJSON, _ := json.Marshal(profile.MCPConfig)

	query := `
		UPDATE profiles
		SET name = $2, mcp_config = $3, neurondb_dsn = $4, agent_endpoint = $5, agent_api_key = $6, default_collection = $7, is_default = $8, updated_at = $9
		WHERE id = $1
	`
	_, err := q.db.ExecContext(ctx, query,
		profile.ID, profile.Name, mcpConfigJSON,
		profile.NeuronDBDSN, profile.AgentEndpoint, profile.AgentAPIKey,
		profile.DefaultCollection, profile.IsDefault, profile.UpdatedAt)
	return err
}

// DeleteProfile deletes a profile
func (q *Queries) DeleteProfile(ctx context.Context, id string) error {
	query := `DELETE FROM profiles WHERE id = $1`
	_, err := q.db.ExecContext(ctx, query, id)
	return err
}

// API Key operations

// CreateAPIKey creates a new API key
func (q *Queries) CreateAPIKey(ctx context.Context, apiKey *APIKey) error {
	if apiKey.ID == "" {
		apiKey.ID = uuid.New().String()
	}
	if apiKey.CreatedAt.IsZero() {
		apiKey.CreatedAt = time.Now()
	}

	query := `
		INSERT INTO api_keys (id, key_hash, key_prefix, user_id, profile_id, rate_limit, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
	`
	_, err := q.db.ExecContext(ctx, query,
		apiKey.ID, apiKey.KeyHash, apiKey.KeyPrefix, apiKey.UserID,
		apiKey.ProfileID, apiKey.RateLimit, apiKey.CreatedAt)
	return err
}

// GetAPIKeyByPrefix gets an API key by prefix
func (q *Queries) GetAPIKeyByPrefix(ctx context.Context, prefix string) (*APIKey, error) {
	var apiKey APIKey

	query := `
		SELECT id, key_hash, key_prefix, user_id, profile_id, rate_limit, last_used_at, created_at
		FROM api_keys
		WHERE key_prefix = $1
	`
	err := q.db.QueryRowContext(ctx, query, prefix).Scan(
		&apiKey.ID, &apiKey.KeyHash, &apiKey.KeyPrefix, &apiKey.UserID,
		&apiKey.ProfileID, &apiKey.RateLimit, &apiKey.LastUsedAt, &apiKey.CreatedAt)
	if err != nil {
		return nil, err
	}

	return &apiKey, nil
}

// UpdateAPIKeyLastUsed updates the last used timestamp
func (q *Queries) UpdateAPIKeyLastUsed(ctx context.Context, id string) error {
	query := `UPDATE api_keys SET last_used_at = $1 WHERE id = $2`
	_, err := q.db.ExecContext(ctx, query, time.Now(), id)
	return err
}

// DeleteAPIKey deletes an API key
func (q *Queries) DeleteAPIKey(ctx context.Context, id string) error {
	query := `DELETE FROM api_keys WHERE id = $1`
	_, err := q.db.ExecContext(ctx, query, id)
	return err
}

// GetAllAPIKeys gets all API keys (for admin purposes)
func (q *Queries) GetAllAPIKeys(ctx context.Context) ([]APIKey, error) {
	query := `
		SELECT id, key_hash, key_prefix, user_id, profile_id, rate_limit, last_used_at, created_at
		FROM api_keys
		ORDER BY created_at DESC
	`
	rows, err := q.db.QueryContext(ctx, query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var keys []APIKey
	for rows.Next() {
		var key APIKey
		var profileID sql.NullString
		var lastUsedAt sql.NullTime

		if err := rows.Scan(
			&key.ID, &key.KeyHash, &key.KeyPrefix, &key.UserID,
			&profileID, &key.RateLimit, &lastUsedAt, &key.CreatedAt); err != nil {
			continue
		}

		if profileID.Valid {
			key.ProfileID = &profileID.String
		}
		if lastUsedAt.Valid {
			key.LastUsedAt = &lastUsedAt.Time
		}

		keys = append(keys, key)
	}

	return keys, nil
}

// RequestLog operations

// CreateRequestLog creates a new request log
func (q *Queries) CreateRequestLog(ctx context.Context, log *RequestLog) error {
	if log.ID == "" {
		log.ID = uuid.New().String()
	}
	if log.CreatedAt.IsZero() {
		log.CreatedAt = time.Now()
	}

	requestBodyJSON, _ := json.Marshal(log.RequestBody)
	responseBodyJSON, _ := json.Marshal(log.ResponseBody)

	query := `
		INSERT INTO request_logs (id, profile_id, endpoint, method, request_body, response_body, status_code, duration_ms, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
	`
	_, err := q.db.ExecContext(ctx, query,
		log.ID, log.ProfileID, log.Endpoint, log.Method,
		requestBodyJSON, responseBodyJSON, log.StatusCode, log.DurationMS, log.CreatedAt)
	return err
}

// ListRequestLogs lists request logs
func (q *Queries) ListRequestLogs(ctx context.Context, profileID *string, limit int) ([]RequestLog, error) {
	var query string
	var rows *sql.Rows
	var err error

	if profileID != nil {
		query = `
			SELECT id, profile_id, endpoint, method, request_body, response_body, status_code, duration_ms, created_at
			FROM request_logs
			WHERE profile_id = $1
			ORDER BY created_at DESC
			LIMIT $2
		`
		rows, err = q.db.QueryContext(ctx, query, *profileID, limit)
	} else {
		query = `
			SELECT id, profile_id, endpoint, method, request_body, response_body, status_code, duration_ms, created_at
			FROM request_logs
			ORDER BY created_at DESC
			LIMIT $1
		`
		rows, err = q.db.QueryContext(ctx, query, limit)
	}

	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var logs []RequestLog
	for rows.Next() {
		var log RequestLog
		var requestBodyJSON, responseBodyJSON []byte
		var profileID sql.NullString

		if err := rows.Scan(
			&log.ID, &profileID, &log.Endpoint, &log.Method,
			&requestBodyJSON, &responseBodyJSON, &log.StatusCode, &log.DurationMS, &log.CreatedAt); err != nil {
			continue
		}

		if profileID.Valid {
			log.ProfileID = &profileID.String
		}

		if len(requestBodyJSON) > 0 {
			json.Unmarshal(requestBodyJSON, &log.RequestBody)
		}
		if len(responseBodyJSON) > 0 {
			json.Unmarshal(responseBodyJSON, &log.ResponseBody)
		}

		logs = append(logs, log)
	}

	return logs, nil
}

// ModelConfig operations

// CreateModelConfig creates a new model configuration
func (q *Queries) CreateModelConfig(ctx context.Context, config *ModelConfig) error {
	if config.ID == "" {
		config.ID = uuid.New().String()
	}
	if config.CreatedAt.IsZero() {
		config.CreatedAt = time.Now()
	}
	config.UpdatedAt = time.Now()

	metadataJSON, _ := json.Marshal(config.Metadata)

	query := `
		INSERT INTO model_configs (id, profile_id, model_provider, model_name, api_key, base_url, is_default, is_free, metadata, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
	`
	_, err := q.db.ExecContext(ctx, query,
		config.ID, config.ProfileID, config.ModelProvider, config.ModelName,
		config.APIKey, config.BaseURL, config.IsDefault, config.IsFree,
		metadataJSON, config.CreatedAt, config.UpdatedAt)
	return err
}

// GetModelConfig gets a model configuration by ID
func (q *Queries) GetModelConfig(ctx context.Context, id string, includeAPIKey bool) (*ModelConfig, error) {
	var config ModelConfig
	var metadataJSON []byte
	var apiKey sql.NullString

	var query string
	if includeAPIKey {
		query = `
			SELECT id, profile_id, model_provider, model_name, api_key, base_url, is_default, is_free, metadata, created_at, updated_at
			FROM model_configs
			WHERE id = $1
		`
	} else {
		query = `
			SELECT id, profile_id, model_provider, model_name, NULL as api_key, base_url, is_default, is_free, metadata, created_at, updated_at
			FROM model_configs
			WHERE id = $1
		`
	}

	err := q.db.QueryRowContext(ctx, query, id).Scan(
		&config.ID, &config.ProfileID, &config.ModelProvider, &config.ModelName,
		&apiKey, &config.BaseURL, &config.IsDefault, &config.IsFree,
		&metadataJSON, &config.CreatedAt, &config.UpdatedAt)
	if err != nil {
		return nil, err
	}

	if apiKey.Valid {
		config.APIKey = apiKey.String
	}

	if len(metadataJSON) > 0 {
		json.Unmarshal(metadataJSON, &config.Metadata)
	}

	return &config, nil
}

// ListModelConfigs lists model configurations for a profile
func (q *Queries) ListModelConfigs(ctx context.Context, profileID string, includeAPIKey bool) ([]ModelConfig, error) {
	var query string
	if includeAPIKey {
		query = `
			SELECT id, profile_id, model_provider, model_name, api_key, base_url, is_default, is_free, metadata, created_at, updated_at
			FROM model_configs
			WHERE profile_id = $1
			ORDER BY is_default DESC, created_at DESC
		`
	} else {
		query = `
			SELECT id, profile_id, model_provider, model_name, NULL as api_key, base_url, is_default, is_free, metadata, created_at, updated_at
			FROM model_configs
			WHERE profile_id = $1
			ORDER BY is_default DESC, created_at DESC
		`
	}

	rows, err := q.db.QueryContext(ctx, query, profileID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var configs []ModelConfig
	for rows.Next() {
		var config ModelConfig
		var metadataJSON []byte
		var apiKey sql.NullString

		if err := rows.Scan(
			&config.ID, &config.ProfileID, &config.ModelProvider, &config.ModelName,
			&apiKey, &config.BaseURL, &config.IsDefault, &config.IsFree,
			&metadataJSON, &config.CreatedAt, &config.UpdatedAt); err != nil {
			continue
		}

		if apiKey.Valid {
			config.APIKey = apiKey.String
		}

		if len(metadataJSON) > 0 {
			json.Unmarshal(metadataJSON, &config.Metadata)
		}

		configs = append(configs, config)
	}

	return configs, nil
}

// UpdateModelConfig updates a model configuration
func (q *Queries) UpdateModelConfig(ctx context.Context, config *ModelConfig) error {
	config.UpdatedAt = time.Now()
	metadataJSON, _ := json.Marshal(config.Metadata)

	query := `
		UPDATE model_configs
		SET model_provider = $2, model_name = $3, api_key = $4, base_url = $5, is_default = $6, is_free = $7, metadata = $8, updated_at = $9
		WHERE id = $1
	`
	_, err := q.db.ExecContext(ctx, query,
		config.ID, config.ModelProvider, config.ModelName, config.APIKey,
		config.BaseURL, config.IsDefault, config.IsFree, metadataJSON, config.UpdatedAt)
	return err
}

// DeleteModelConfig deletes a model configuration
func (q *Queries) DeleteModelConfig(ctx context.Context, id string) error {
	query := `DELETE FROM model_configs WHERE id = $1`
	_, err := q.db.ExecContext(ctx, query, id)
	return err
}

// SetDefaultModelConfig sets a model as default (unsetting others)
func (q *Queries) SetDefaultModelConfig(ctx context.Context, profileID string, modelID string) error {
	tx, err := q.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()

	// Unset all defaults for this profile
	_, err = tx.ExecContext(ctx, `UPDATE model_configs SET is_default = false WHERE profile_id = $1`, profileID)
	if err != nil {
		return err
	}

	// Set the specified model as default
	_, err = tx.ExecContext(ctx, `UPDATE model_configs SET is_default = true WHERE id = $1`, modelID)
	if err != nil {
		return err
	}

	return tx.Commit()
}

// GetDefaultModelConfig gets the default model for a profile
func (q *Queries) GetDefaultModelConfig(ctx context.Context, profileID string) (*ModelConfig, error) {
	var config ModelConfig
	var metadataJSON []byte

	query := `
		SELECT id, profile_id, model_provider, model_name, NULL as api_key, base_url, is_default, is_free, metadata, created_at, updated_at
		FROM model_configs
		WHERE profile_id = $1 AND is_default = true
		LIMIT 1
	`
	err := q.db.QueryRowContext(ctx, query, profileID).Scan(
		&config.ID, &config.ProfileID, &config.ModelProvider, &config.ModelName,
		&config.APIKey, &config.BaseURL, &config.IsDefault, &config.IsFree,
		&metadataJSON, &config.CreatedAt, &config.UpdatedAt)
	if err != nil {
		return nil, err
	}

	if len(metadataJSON) > 0 {
		json.Unmarshal(metadataJSON, &config.Metadata)
	}

	return &config, nil
}

// GetDefaultProfile gets the default profile
func (q *Queries) GetDefaultProfile(ctx context.Context) (*Profile, error) {
	var profile Profile
	var mcpConfigJSON []byte
	var agentEndpoint, agentAPIKey, defaultCollection sql.NullString

	query := `
		SELECT id, name, user_id, mcp_config, neurondb_dsn, agent_endpoint, agent_api_key, default_collection, is_default, created_at, updated_at
		FROM profiles
		WHERE is_default = true
		LIMIT 1
	`
	err := q.db.QueryRowContext(ctx, query).Scan(
		&profile.ID, &profile.Name, &profile.UserID, &mcpConfigJSON,
		&profile.NeuronDBDSN, &agentEndpoint, &agentAPIKey,
		&defaultCollection, &profile.IsDefault, &profile.CreatedAt, &profile.UpdatedAt)
	if err != nil {
		return nil, err
	}

	if agentEndpoint.Valid {
		profile.AgentEndpoint = agentEndpoint.String
	}
	if agentAPIKey.Valid {
		profile.AgentAPIKey = agentAPIKey.String
	}
	if defaultCollection.Valid {
		profile.DefaultCollection = defaultCollection.String
	}

	if len(mcpConfigJSON) > 0 {
		json.Unmarshal(mcpConfigJSON, &profile.MCPConfig)
	}

	return &profile, nil
}

// SetDefaultProfile sets a profile as default (unsetting others)
func (q *Queries) SetDefaultProfile(ctx context.Context, profileID string) error {
	tx, err := q.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()

	// Unset all defaults
	_, err = tx.ExecContext(ctx, `UPDATE profiles SET is_default = false`)
	if err != nil {
		return err
	}

	// Set the specified profile as default
	_, err = tx.ExecContext(ctx, `UPDATE profiles SET is_default = true WHERE id = $1`, profileID)
	if err != nil {
		return err
	}

	return tx.Commit()
}

