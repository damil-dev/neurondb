package db

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
	_ "github.com/jackc/pgx/v5/stdlib"
	"golang.org/x/crypto/bcrypt"
)

// Queries provides database operations
type Queries struct {
	db *sql.DB
}

// NewQueries creates a new Queries instance
func NewQueries(db *sql.DB) *Queries {
	return &Queries{db: db}
}

// GetDB returns the underlying database connection
func (q *Queries) GetDB() *sql.DB {
	return q.db
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
		INSERT INTO profiles (id, name, user_id, profile_username, profile_password_hash, mcp_config, neurondb_dsn, agent_endpoint, agent_api_key, default_collection, is_default, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
	`
	_, err := q.db.ExecContext(ctx, query,
		profile.ID, profile.Name, profile.UserID, profile.ProfileUsername, profile.ProfilePassword, mcpConfigJSON,
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
		SELECT id, name, user_id, profile_username, mcp_config, neurondb_dsn, agent_endpoint, agent_api_key, default_collection, is_default, created_at, updated_at
		FROM profiles
		WHERE id = $1
	`
	var profileUsername sql.NullString
	err := q.db.QueryRowContext(ctx, query, id).Scan(
		&profile.ID, &profile.Name, &profile.UserID, &profileUsername, &mcpConfigJSON,
		&profile.NeuronDBDSN, &agentEndpoint, &agentAPIKey,
		&defaultCollection, &profile.IsDefault, &profile.CreatedAt, &profile.UpdatedAt)
	if profileUsername.Valid {
		profile.ProfileUsername = profileUsername.String
	}
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
		SELECT id, name, user_id, profile_username, mcp_config, neurondb_dsn, agent_endpoint, agent_api_key, default_collection, is_default, created_at, updated_at
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
		var profileUsername sql.NullString

		if err := rows.Scan(
			&profile.ID, &profile.Name, &profile.UserID, &profileUsername, &mcpConfigJSON,
			&profile.NeuronDBDSN, &agentEndpoint, &agentAPIKey,
			&defaultCollection, &profile.IsDefault, &profile.CreatedAt, &profile.UpdatedAt); err != nil {
			continue
		}

		if profileUsername.Valid {
			profile.ProfileUsername = profileUsername.String
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

// ListAllProfiles lists all profiles (admin only)
func (q *Queries) ListAllProfiles(ctx context.Context) ([]Profile, error) {
	query := `
		SELECT id, name, user_id, profile_username, mcp_config, neurondb_dsn, agent_endpoint, agent_api_key, default_collection, is_default, created_at, updated_at
		FROM profiles
		ORDER BY is_default DESC, created_at DESC
	`
	rows, err := q.db.QueryContext(ctx, query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var profiles []Profile
	for rows.Next() {
		var profile Profile
		var mcpConfigJSON []byte
		var agentEndpoint, agentAPIKey, defaultCollection sql.NullString
		var profileUsername sql.NullString

		if err := rows.Scan(
			&profile.ID, &profile.Name, &profile.UserID, &profileUsername, &mcpConfigJSON,
			&profile.NeuronDBDSN, &agentEndpoint, &agentAPIKey,
			&defaultCollection, &profile.IsDefault, &profile.CreatedAt, &profile.UpdatedAt); err != nil {
			continue
		}

		if profileUsername.Valid {
			profile.ProfileUsername = profileUsername.String
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

// GetProfileByUsernameAndPassword gets a profile by username and password
func (q *Queries) GetProfileByUsernameAndPassword(ctx context.Context, username, password string) (*Profile, error) {
	var profile Profile
	var mcpConfigJSON []byte
	var agentEndpoint, agentAPIKey, defaultCollection sql.NullString
	var profileUsername sql.NullString
	var profilePasswordHash sql.NullString

	query := `
		SELECT id, name, user_id, profile_username, profile_password_hash, mcp_config, neurondb_dsn, agent_endpoint, agent_api_key, default_collection, is_default, created_at, updated_at
		FROM profiles
		WHERE profile_username = $1
	`
	err := q.db.QueryRowContext(ctx, query, username).Scan(
		&profile.ID, &profile.Name, &profile.UserID, &profileUsername, &profilePasswordHash, &mcpConfigJSON,
		&profile.NeuronDBDSN, &agentEndpoint, &agentAPIKey,
		&defaultCollection, &profile.IsDefault, &profile.CreatedAt, &profile.UpdatedAt)
	if err != nil {
		return nil, err
	}

	if profileUsername.Valid {
		profile.ProfileUsername = profileUsername.String
	}

	// Verify password
	if !profilePasswordHash.Valid {
		return nil, fmt.Errorf("profile has no password set")
	}

	if err := bcrypt.CompareHashAndPassword([]byte(profilePasswordHash.String), []byte(password)); err != nil {
		return nil, fmt.Errorf("invalid password")
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

// UpdateProfile updates a profile
func (q *Queries) UpdateProfile(ctx context.Context, profile *Profile) error {
	profile.UpdatedAt = time.Now()
	mcpConfigJSON, _ := json.Marshal(profile.MCPConfig)

	query := `
		UPDATE profiles
		SET name = $2, profile_username = $3, profile_password_hash = $4, mcp_config = $5, neurondb_dsn = $6, agent_endpoint = $7, agent_api_key = $8, default_collection = $9, is_default = $10, updated_at = $11
		WHERE id = $1
	`
	_, err := q.db.ExecContext(ctx, query,
		profile.ID, profile.Name, profile.ProfileUsername, profile.ProfilePassword, mcpConfigJSON,
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

// GetDefaultProfileForUser gets the "current" profile for a user (default if set, otherwise most recent)
func (q *Queries) GetDefaultProfileForUser(ctx context.Context, userID string) (*Profile, error) {
	var profile Profile
	var mcpConfigJSON []byte
	var agentEndpoint, agentAPIKey, defaultCollection sql.NullString
	var profileUsername sql.NullString

	query := `
		SELECT id, name, user_id, profile_username, mcp_config, neurondb_dsn, agent_endpoint, agent_api_key, default_collection, is_default, created_at, updated_at
		FROM profiles
		WHERE user_id = $1
		ORDER BY is_default DESC, created_at DESC
		LIMIT 1
	`
	err := q.db.QueryRowContext(ctx, query, userID).Scan(
		&profile.ID, &profile.Name, &profile.UserID, &profileUsername, &mcpConfigJSON,
		&profile.NeuronDBDSN, &agentEndpoint, &agentAPIKey,
		&defaultCollection, &profile.IsDefault, &profile.CreatedAt, &profile.UpdatedAt)
	if err != nil {
		return nil, err
	}

	if profileUsername.Valid {
		profile.ProfileUsername = profileUsername.String
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

// App Setting operations

// GetSetting gets an app setting by key
func (q *Queries) GetSetting(ctx context.Context, key string) (*AppSetting, error) {
	var setting AppSetting
	var valueJSON []byte

	query := `
		SELECT key, value, updated_at
		FROM app_settings
		WHERE key = $1
	`
	err := q.db.QueryRowContext(ctx, query, key).Scan(
		&setting.Key, &valueJSON, &setting.UpdatedAt)
	if err != nil {
		return nil, err
	}

	if len(valueJSON) > 0 {
		json.Unmarshal(valueJSON, &setting.Value)
	}

	return &setting, nil
}

// UpsertSetting upserts an app setting
func (q *Queries) UpsertSetting(ctx context.Context, key string, value map[string]interface{}) error {
	valueJSON, _ := json.Marshal(value)

	query := `
		INSERT INTO app_settings (key, value, updated_at)
		VALUES ($1, $2, NOW())
		ON CONFLICT (key) 
		DO UPDATE SET 
			value = EXCLUDED.value,
			updated_at = NOW()
	`
	_, err := q.db.ExecContext(ctx, query, key, valueJSON)
	return err
}

// User operations

// CreateUser creates a new user
func (q *Queries) CreateUser(ctx context.Context, user *User) error {
	if user.ID == "" {
		user.ID = uuid.New().String()
	}
	if user.CreatedAt.IsZero() {
		user.CreatedAt = time.Now()
	}
	user.UpdatedAt = time.Now()

	query := `
		INSERT INTO users (id, username, password_hash, is_admin, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6)
	`
	_, err := q.db.ExecContext(ctx, query,
		user.ID, user.Username, user.PasswordHash, user.IsAdmin, user.CreatedAt, user.UpdatedAt)
	return err
}

// GetUserByUsername gets a user by username
func (q *Queries) GetUserByUsername(ctx context.Context, username string) (*User, error) {
	var user User

	query := `
		SELECT id, username, password_hash, is_admin, created_at, updated_at
		FROM users
		WHERE username = $1
	`
	err := q.db.QueryRowContext(ctx, query, username).Scan(
		&user.ID, &user.Username, &user.PasswordHash, &user.IsAdmin,
		&user.CreatedAt, &user.UpdatedAt)
	if err != nil {
		return nil, err
	}

	return &user, nil
}

// GetUserByID gets a user by ID
func (q *Queries) GetUserByID(ctx context.Context, id string) (*User, error) {
	var user User

	query := `
		SELECT id, username, password_hash, is_admin, created_at, updated_at
		FROM users
		WHERE id = $1
	`
	err := q.db.QueryRowContext(ctx, query, id).Scan(
		&user.ID, &user.Username, &user.PasswordHash, &user.IsAdmin,
		&user.CreatedAt, &user.UpdatedAt)
	if err != nil {
		return nil, err
	}

	return &user, nil
}

// UpdateUserPassword updates a user's password
func (q *Queries) UpdateUserPassword(ctx context.Context, userID string, passwordHash string) error {
	query := `
		UPDATE users
		SET password_hash = $1, updated_at = NOW()
		WHERE id = $2
	`
	_, err := q.db.ExecContext(ctx, query, passwordHash, userID)
	return err
}

// MCP Chat Thread operations

// MCPThread represents a chat thread
type MCPThread struct {
	ID        string    `json:"id"`
	ProfileID string    `json:"profile_id"`
	Title     string    `json:"title"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// MCPMessage represents a chat message
type MCPMessage struct {
	ID        string                 `json:"id"`
	ThreadID  string                 `json:"thread_id"`
	Role      string                 `json:"role"`
	Content   string                 `json:"content"`
	ToolName  sql.NullString         `json:"tool_name,omitempty"`
	Data      map[string]interface{} `json:"data,omitempty"`
	CreatedAt time.Time              `json:"created_at"`
}

// MarshalJSON customizes JSON marshaling for MCPMessage to handle sql.NullString properly
func (m MCPMessage) MarshalJSON() ([]byte, error) {
	aux := struct {
		ID        string                 `json:"id"`
		ThreadID  string                 `json:"thread_id"`
		Role      string                 `json:"role"`
		Content   string                 `json:"content"`
		ToolName  *string                `json:"tool_name,omitempty"`
		Data      map[string]interface{} `json:"data,omitempty"`
		CreatedAt time.Time              `json:"created_at"`
	}{
		ID:        m.ID,
		ThreadID:  m.ThreadID,
		Role:      m.Role,
		Content:   m.Content,
		Data:      m.Data,
		CreatedAt: m.CreatedAt,
	}
	if m.ToolName.Valid {
		aux.ToolName = &m.ToolName.String
	}
	return json.Marshal(aux)
}

// ListMCPThreads lists all threads for a profile
func (q *Queries) ListMCPThreads(ctx context.Context, profileID string) ([]MCPThread, error) {
	query := `
		SELECT id, profile_id, title, created_at, updated_at
		FROM mcp_chat_threads
		WHERE profile_id = $1
		ORDER BY updated_at DESC
	`
	rows, err := q.db.QueryContext(ctx, query, profileID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var threads []MCPThread
	for rows.Next() {
		var thread MCPThread
		if err := rows.Scan(&thread.ID, &thread.ProfileID, &thread.Title, &thread.CreatedAt, &thread.UpdatedAt); err != nil {
			return nil, err
		}
		threads = append(threads, thread)
	}

	return threads, rows.Err()
}

// CreateMCPThread creates a new thread
func (q *Queries) CreateMCPThread(ctx context.Context, profileID, title string) (*MCPThread, error) {
	threadID := uuid.New().String()
	query := `
		INSERT INTO mcp_chat_threads (id, profile_id, title, created_at, updated_at)
		VALUES ($1, $2, $3, NOW(), NOW())
		RETURNING id, profile_id, title, created_at, updated_at
	`
	var thread MCPThread
	err := q.db.QueryRowContext(ctx, query, threadID, profileID, title).Scan(
		&thread.ID, &thread.ProfileID, &thread.Title, &thread.CreatedAt, &thread.UpdatedAt)
	if err != nil {
		return nil, err
	}
	return &thread, nil
}

// GetMCPThread gets a thread by ID
func (q *Queries) GetMCPThread(ctx context.Context, threadID string) (*MCPThread, error) {
	query := `
		SELECT id, profile_id, title, created_at, updated_at
		FROM mcp_chat_threads
		WHERE id = $1
	`
	var thread MCPThread
	err := q.db.QueryRowContext(ctx, query, threadID).Scan(
		&thread.ID, &thread.ProfileID, &thread.Title, &thread.CreatedAt, &thread.UpdatedAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &thread, nil
}

// UpdateMCPThread updates a thread's title and updated_at
func (q *Queries) UpdateMCPThread(ctx context.Context, threadID, title string) (*MCPThread, error) {
	query := `
		UPDATE mcp_chat_threads
		SET title = $1, updated_at = NOW()
		WHERE id = $2
		RETURNING id, profile_id, title, created_at, updated_at
	`
	var thread MCPThread
	err := q.db.QueryRowContext(ctx, query, title, threadID).Scan(
		&thread.ID, &thread.ProfileID, &thread.Title, &thread.CreatedAt, &thread.UpdatedAt)
	if err != nil {
		return nil, err
	}
	return &thread, nil
}

// DeleteMCPThread deletes a thread (cascade deletes messages)
func (q *Queries) DeleteMCPThread(ctx context.Context, threadID string) error {
	query := `DELETE FROM mcp_chat_threads WHERE id = $1`
	_, err := q.db.ExecContext(ctx, query, threadID)
	return err
}

// ListMCPMessages lists all messages for a thread
func (q *Queries) ListMCPMessages(ctx context.Context, threadID string) ([]MCPMessage, error) {
	query := `
		SELECT id, thread_id, role, content, tool_name, data, created_at
		FROM mcp_chat_messages
		WHERE thread_id = $1
		ORDER BY created_at ASC
	`
	rows, err := q.db.QueryContext(ctx, query, threadID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var messages []MCPMessage
	for rows.Next() {
		var msg MCPMessage
		var toolName sql.NullString
		var dataJSON []byte

		if err := rows.Scan(&msg.ID, &msg.ThreadID, &msg.Role, &msg.Content, &toolName, &dataJSON, &msg.CreatedAt); err != nil {
			return nil, err
		}

		if toolName.Valid {
			msg.ToolName = toolName
		}

		if len(dataJSON) > 0 {
			json.Unmarshal(dataJSON, &msg.Data)
		}

		messages = append(messages, msg)
	}

	return messages, rows.Err()
}

// CreateMCPMessage creates a new message and updates thread's updated_at
func (q *Queries) CreateMCPMessage(ctx context.Context, threadID, role, content, toolName string, data map[string]interface{}) (*MCPMessage, error) {
	messageID := uuid.New().String()

	var dataJSON []byte
	if data != nil {
		dataJSON, _ = json.Marshal(data)
	}

	// Update thread's updated_at when adding a message
	_, err := q.db.ExecContext(ctx, `UPDATE mcp_chat_threads SET updated_at = NOW() WHERE id = $1`, threadID)
	if err != nil {
		return nil, fmt.Errorf("failed to update thread: %w", err)
	}

	var toolNameNull sql.NullString
	if toolName != "" {
		toolNameNull = sql.NullString{String: toolName, Valid: true}
	}

	var dataJSONNull interface{}
	if len(dataJSON) > 0 {
		// Pass as string for JSONB casting - PostgreSQL will parse it
		dataJSONNull = string(dataJSON)
	} else {
		dataJSONNull = nil
	}

	query := `
		INSERT INTO mcp_chat_messages (id, thread_id, role, content, tool_name, data, created_at)
		VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW())
		RETURNING id, thread_id, role, content, tool_name, data, created_at
	`
	var msg MCPMessage
	var toolNameResult sql.NullString
	var dataJSONResult []byte

	err = q.db.QueryRowContext(ctx, query, messageID, threadID, role, content, toolNameNull, dataJSONNull).Scan(
		&msg.ID, &msg.ThreadID, &msg.Role, &msg.Content, &toolNameResult, &dataJSONResult, &msg.CreatedAt)
	if err != nil {
		return nil, err
	}

	if toolNameResult.Valid {
		msg.ToolName = toolNameResult
	}

	if len(dataJSONResult) > 0 {
		json.Unmarshal(dataJSONResult, &msg.Data)
	}

	return &msg, nil
}
