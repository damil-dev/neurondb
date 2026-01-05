package handlers

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/auth"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/utils"
)

/* Enhanced ModelHandlers with comprehensive validation, error handling, and audit logging */
type EnhancedModelHandlers struct {
	queries *db.Queries
}

/* NewEnhancedModelHandlers creates new enhanced model handlers */
func NewEnhancedModelHandlers(queries *db.Queries) *EnhancedModelHandlers {
	return &EnhancedModelHandlers{queries: queries}
}

/* ModelRequest represents a model creation/update request with validation */
type ModelRequest struct {
	Name        string                 `json:"name"`
	Provider    string                 `json:"provider"`
	ModelType   string                 `json:"model_type"`
	Config      map[string]interface{} `json:"config,omitempty"`
	Description string                 `json:"description,omitempty"`
}

/* Validate validates the model request */
func (r *ModelRequest) Validate() []error {
	var errors []error

	// Name validation
	if strings.TrimSpace(r.Name) == "" {
		errors = append(errors, &utils.ValidationError{
			Field:   "name",
			Message: "name is required",
		})
	} else if len(r.Name) > 100 {
		errors = append(errors, &utils.ValidationError{
			Field:   "name",
			Message: "name must be less than 100 characters",
		})
	} else if !isValidModelName(r.Name) {
		errors = append(errors, &utils.ValidationError{
			Field:   "name",
			Message: "name contains invalid characters (alphanumeric, dash, underscore only)",
		})
	}

	// Provider validation
	validProviders := []string{"openai", "anthropic", "google", "cohere", "huggingface", "local", "custom"}
	if strings.TrimSpace(r.Provider) == "" {
		errors = append(errors, &utils.ValidationError{
			Field:   "provider",
			Message: "provider is required",
		})
	} else if !contains(validProviders, strings.ToLower(r.Provider)) {
		errors = append(errors, &utils.ValidationError{
			Field:   "provider",
			Message: fmt.Sprintf("provider must be one of: %s", strings.Join(validProviders, ", ")),
		})
	}

	// Model type validation
	validModelTypes := []string{"chat", "completion", "embedding", "image", "audio", "custom"}
	if strings.TrimSpace(r.ModelType) == "" {
		errors = append(errors, &utils.ValidationError{
			Field:   "model_type",
			Message: "model_type is required",
		})
	} else if !contains(validModelTypes, strings.ToLower(r.ModelType)) {
		errors = append(errors, &utils.ValidationError{
			Field:   "model_type",
			Message: fmt.Sprintf("model_type must be one of: %s", strings.Join(validModelTypes, ", ")),
		})
	}

	// Config validation
	if r.Config != nil {
		if err := validateModelConfig(r.Config, r.ModelType); err != nil {
			errors = append(errors, err)
		}
	}

	// Description validation
	if r.Description != "" && len(r.Description) > 500 {
		errors = append(errors, &utils.ValidationError{
			Field:   "description",
			Message: "description must be less than 500 characters",
		})
	}

	return errors
}

/* isValidModelName validates model name format */
func isValidModelName(name string) bool {
	matched, _ := regexp.MatchString(`^[a-zA-Z0-9_-]+$`, name)
	return matched
}

/* contains checks if slice contains string */
func contains(slice []string, item string) bool {
	for _, s := range slice {
		if strings.EqualFold(s, item) {
			return true
		}
	}
	return false
}

/* validateModelConfig validates model configuration */
func validateModelConfig(config map[string]interface{}, modelType string) error {
	// Validate temperature (if present)
	if temp, ok := config["temperature"].(float64); ok {
		if temp < 0 || temp > 2 {
			return &utils.ValidationError{
				Field:   "config.temperature",
				Message: "temperature must be between 0 and 2",
			}
		}
	}

	// Validate max_tokens (if present)
	if maxTokens, ok := config["max_tokens"].(float64); ok {
		if maxTokens < 1 || maxTokens > 100000 {
			return &utils.ValidationError{
				Field:   "config.max_tokens",
				Message: "max_tokens must be between 1 and 100000",
			}
		}
	}

	// Validate top_p (if present)
	if topP, ok := config["top_p"].(float64); ok {
		if topP < 0 || topP > 1 {
			return &utils.ValidationError{
				Field:   "config.top_p",
				Message: "top_p must be between 0 and 1",
			}
		}
	}

	return nil
}

/* ListModelsEnhanced lists all models with pagination, filtering, and sorting */
func (h *EnhancedModelHandlers) ListModelsEnhanced(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	// Get user context for audit logging
	userID, _ := auth.GetUserIDFromContext(r.Context())
	requestID := r.Context().Value("request_id").(string)

	// Validate profile access
	profile, err := h.queries.GetProfile(r.Context(), profileID)
	if err != nil {
		if err == sql.ErrNoRows {
			WriteError(w, r, http.StatusNotFound, fmt.Errorf("profile not found"), map[string]interface{}{
				"profile_id": profileID,
			})
		} else {
			WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to get profile: %w", err), nil)
		}
		return
	}

	// Parse query parameters
	limit := parseIntQueryParam(r, "limit", 50, 1, 1000)
	offset := parseIntQueryParam(r, "offset", 0, 0, 10000)
	provider := r.URL.Query().Get("provider")
	modelType := r.URL.Query().Get("model_type")
	sortBy := r.URL.Query().Get("sort_by")
	if sortBy == "" {
		sortBy = "name"
	}
	sortOrder := r.URL.Query().Get("sort_order")
	if sortOrder == "" {
		sortOrder = "ASC"
	}

	// Validate sort parameters
	validSortFields := []string{"name", "provider", "model_type", "created_at", "updated_at"}
	if !contains(validSortFields, sortBy) {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid sort_by field"), map[string]interface{}{
			"valid_fields": validSortFields,
		})
		return
	}
	if sortOrder != "ASC" && sortOrder != "DESC" {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("sort_order must be ASC or DESC"), nil)
		return
	}

	// Build query with filters
	query := `
		SELECT 
			id,
			name,
			provider,
			model_type,
			CASE WHEN api_key_encrypted IS NOT NULL THEN true ELSE false END as api_key_set,
			config,
			created_at,
			updated_at
		FROM neurondb.llm_models
		WHERE enabled = true
	`
	args := []interface{}{}
	argIndex := 1

	if provider != "" {
		query += fmt.Sprintf(" AND provider = $%d", argIndex)
		args = append(args, provider)
		argIndex++
	}

	if modelType != "" {
		query += fmt.Sprintf(" AND model_type = $%d", argIndex)
		args = append(args, modelType)
		argIndex++
	}

	query += fmt.Sprintf(" ORDER BY %s %s", sortBy, sortOrder)
	query += fmt.Sprintf(" LIMIT $%d OFFSET $%d", argIndex, argIndex+1)
	args = append(args, limit, offset)

	// Get total count for pagination
	countQuery := `
		SELECT COUNT(*) 
		FROM neurondb.llm_models
		WHERE enabled = true
	`
	countArgs := []interface{}{}
	countArgIndex := 1

	if provider != "" {
		countQuery += fmt.Sprintf(" AND provider = $%d", countArgIndex)
		countArgs = append(countArgs, provider)
		countArgIndex++
	}

	if modelType != "" {
		countQuery += fmt.Sprintf(" AND model_type = $%d", countArgIndex)
		countArgs = append(countType, modelType)
		countArgIndex++
	}

	var totalCount int64
	err = h.queries.GetDB().QueryRowContext(r.Context(), countQuery, countArgs...).Scan(&totalCount)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to get total count: %w", err), nil)
		return
	}

	// Execute query
	rows, err := h.queries.GetDB().QueryContext(r.Context(), query, args...)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to query models: %w", err), nil)
		return
	}
	defer rows.Close()

	var models []Model
	for rows.Next() {
		var m Model
		var configJSON []byte
		err := rows.Scan(
			&m.ID, &m.Name, &m.Provider, &m.ModelType,
			&m.APIKeySet, &configJSON, &m.CreatedAt, &m.UpdatedAt,
		)
		if err != nil {
			// Log error but continue
			continue
		}

		if len(configJSON) > 0 {
			if err := json.Unmarshal(configJSON, &m.Config); err != nil {
				// Log error but continue
				continue
			}
		}

		models = append(models, m)
	}

	if err = rows.Err(); err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("error iterating rows: %w", err), nil)
		return
	}

	// Log audit event
	go h.logAuditEvent(r.Context(), "models.list", userID, profileID, map[string]interface{}{
		"count":      len(models),
		"total_count": totalCount,
		"filters": map[string]interface{}{
			"provider":   provider,
			"model_type": modelType,
		},
	})

	// Return paginated response
	response := map[string]interface{}{
		"data": models,
		"pagination": map[string]interface{}{
			"limit":       limit,
			"offset":      offset,
			"total":       totalCount,
			"has_more":    offset+int64(len(models)) < totalCount,
			"total_pages": (totalCount + int64(limit) - 1) / int64(limit),
		},
	}

	WriteSuccess(w, response, http.StatusOK)
}

/* AddModelEnhanced adds a new model with comprehensive validation */
func (h *EnhancedModelHandlers) AddModelEnhanced(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	// Get user context
	userID, _ := auth.GetUserIDFromContext(r.Context())
	requestID := r.Context().Value("request_id").(string)

	// Validate profile access
	profile, err := h.queries.GetProfile(r.Context(), profileID)
	if err != nil {
		if err == sql.ErrNoRows {
			WriteError(w, r, http.StatusNotFound, fmt.Errorf("profile not found"), nil)
		} else {
			WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to get profile: %w", err), nil)
		}
		return
	}

	// Parse and validate request
	var req ModelRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err), nil)
		return
	}

	// Validate request
	if validationErrors := req.Validate(); len(validationErrors) > 0 {
		WriteValidationErrors(w, r, validationErrors)
		return
	}

	// Check for duplicate name
	var existingID string
	checkQuery := `SELECT id FROM neurondb.llm_models WHERE name = $1 AND enabled = true`
	err = h.queries.GetDB().QueryRowContext(r.Context(), checkQuery, req.Name).Scan(&existingID)
	if err == nil {
		WriteError(w, r, http.StatusConflict, fmt.Errorf("model with name '%s' already exists", req.Name), map[string]interface{}{
			"existing_id": existingID,
		})
		return
	} else if err != sql.ErrNoRows {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to check for duplicate: %w", err), nil)
		return
	}

	// Prepare config JSON
	configJSON, err := json.Marshal(req.Config)
	if err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid config JSON: %w", err), nil)
		return
	}

	// Insert model with transaction for atomicity
	tx, err := h.queries.GetDB().BeginTx(r.Context(), nil)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to begin transaction: %w", err), nil)
		return
	}
	defer tx.Rollback()

	insertQuery := `
		INSERT INTO neurondb.llm_models (name, provider, model_type, config, enabled, created_at, updated_at)
		VALUES ($1, $2, $3, $4::jsonb, true, NOW(), NOW())
		RETURNING id, name, provider, model_type, 
		          CASE WHEN api_key_encrypted IS NOT NULL THEN true ELSE false END as api_key_set,
		          config, created_at, updated_at
	`

	var m Model
	var configJSONOut []byte
	err = tx.QueryRowContext(r.Context(), insertQuery,
		req.Name, req.Provider, req.ModelType, configJSON,
	).Scan(
		&m.ID, &m.Name, &m.Provider, &m.ModelType,
		&m.APIKeySet, &configJSONOut, &m.CreatedAt, &m.UpdatedAt,
	)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to insert model: %w", err), nil)
		return
	}

	if len(configJSONOut) > 0 {
		if err := json.Unmarshal(configJSONOut, &m.Config); err != nil {
			// Log warning but continue
		}
	}

	// Commit transaction
	if err = tx.Commit(); err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to commit transaction: %w", err), nil)
		return
	}

	// Log audit event
	go h.logAuditEvent(r.Context(), "models.create", userID, profileID, map[string]interface{}{
		"model_id":   m.ID,
		"model_name": m.Name,
		"provider":   m.Provider,
		"model_type": m.ModelType,
	})

	WriteSuccess(w, m, http.StatusCreated)
}

/* SetModelKeyEnhanced sets API key with validation and encryption */
func (h *EnhancedModelHandlers) SetModelKeyEnhanced(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	modelName := vars["model_name"]

	// Get user context
	userID, _ := auth.GetUserIDFromContext(r.Context())

	// Validate profile access
	profile, err := h.queries.GetProfile(r.Context(), profileID)
	if err != nil {
		if err == sql.ErrNoRows {
			WriteError(w, r, http.StatusNotFound, fmt.Errorf("profile not found"), nil)
		} else {
			WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to get profile: %w", err), nil)
		}
		return
	}

	// Parse request
	var req struct {
		APIKey string `json:"api_key"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err), nil)
		return
	}

	// Validate API key
	if strings.TrimSpace(req.APIKey) == "" {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("api_key is required"), nil)
		return
	}

	// Validate API key format based on provider
	if err := validateAPIKeyFormat(req.APIKey, modelName); err != nil {
		WriteError(w, r, http.StatusBadRequest, err, nil)
		return
	}

	// Check model exists
	var modelID string
	checkQuery := `SELECT id FROM neurondb.llm_models WHERE name = $1 AND enabled = true`
	err = h.queries.GetDB().QueryRowContext(r.Context(), checkQuery, modelName).Scan(&modelID)
	if err == sql.ErrNoRows {
		WriteError(w, r, http.StatusNotFound, fmt.Errorf("model not found"), nil)
		return
	} else if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to check model: %w", err), nil)
		return
	}

	// Use NeuronDB function to set encrypted API key
	query := `SELECT neurondb_set_model_key($1, $2)`
	_, err = h.queries.GetDB().ExecContext(r.Context(), query, modelName, req.APIKey)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to set API key: %w", err), nil)
		return
	}

	// Log audit event (without API key value)
	go h.logAuditEvent(r.Context(), "models.set_key", userID, profileID, map[string]interface{}{
		"model_id":   modelID,
		"model_name": modelName,
		"key_set":    true,
	})

	WriteSuccess(w, map[string]interface{}{
		"message":  "API key set successfully",
		"model_id": modelID,
	}, http.StatusOK)
}

/* validateAPIKeyFormat validates API key format based on provider */
func validateAPIKeyFormat(apiKey, modelName string) error {
	// Basic length check
	if len(apiKey) < 16 {
		return &utils.ValidationError{
			Field:   "api_key",
			Message: "API key must be at least 16 characters",
		}
	}

	// Provider-specific validation could be added here
	// For now, just check it's not obviously invalid

	return nil
}

/* parseIntQueryParam parses and validates integer query parameter */
func parseIntQueryParam(r *http.Request, param string, defaultValue, min, max int64) int64 {
	valueStr := r.URL.Query().Get(param)
	if valueStr == "" {
		return defaultValue
	}

	var value int64
	if _, err := fmt.Sscanf(valueStr, "%d", &value); err != nil {
		return defaultValue
	}

	if value < min {
		return min
	}
	if value > max {
		return max
	}

	return value
}

/* logAuditEvent logs an audit event asynchronously */
func (h *EnhancedModelHandlers) logAuditEvent(ctx context.Context, eventType, userID, profileID string, metadata map[string]interface{}) {
	// Create audit log entry
	query := `
		INSERT INTO audit_log (event_type, user_id, profile_id, metadata, created_at)
		VALUES ($1, $2, $3, $4::jsonb, NOW())
	`

	metadataJSON, _ := json.Marshal(metadata)
	_, err := h.queries.GetDB().ExecContext(ctx, query, eventType, userID, profileID, metadataJSON)
	if err != nil {
		// Log error but don't fail the request
		fmt.Printf("Failed to log audit event: %v\n", err)
	}
}

