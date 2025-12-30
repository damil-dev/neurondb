package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/neurondb/NeuronDesktop/api/internal/middleware"
	"github.com/neurondb/NeuronDesktop/api/internal/utils"
)

// ErrorResponse represents an error response
type ErrorResponse struct {
	Error     string                 `json:"error"`
	Message   string                 `json:"message,omitempty"`
	Code      string                 `json:"code,omitempty"`
	Details   map[string]interface{} `json:"details,omitempty"`
	RequestID string                 `json:"request_id,omitempty"`
}

// WriteError writes an error response
func WriteError(w http.ResponseWriter, r *http.Request, statusCode int, err error, details map[string]interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)

	response := ErrorResponse{
		Error:   http.StatusText(statusCode),
		Message: err.Error(),
	}

	// Add request ID from context
	if r != nil {
		requestID := middleware.GetRequestID(r.Context())
		if requestID != "" {
			response.RequestID = requestID
		}
	}

	if details != nil {
		response.Details = details
	}

	// Add validation errors if applicable
	if validationErr, ok := err.(*utils.ValidationError); ok {
		response.Code = "VALIDATION_ERROR"
		if response.Details == nil {
			response.Details = make(map[string]interface{})
		}
		response.Details["field"] = validationErr.Field
		response.Details["message"] = validationErr.Message
	}

	json.NewEncoder(w).Encode(response)
}

// WriteErrorWithContext writes an error response with context
func WriteErrorWithContext(ctx context.Context, w http.ResponseWriter, statusCode int, err error, details map[string]interface{}) {
	requestID := middleware.GetRequestID(ctx)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)

	response := ErrorResponse{
		Error:     http.StatusText(statusCode),
		Message:   err.Error(),
		RequestID: requestID,
	}

	if details != nil {
		response.Details = details
	}

	// Add validation errors if applicable
	if validationErr, ok := err.(*utils.ValidationError); ok {
		response.Code = "VALIDATION_ERROR"
		if response.Details == nil {
			response.Details = make(map[string]interface{})
		}
		response.Details["field"] = validationErr.Field
		response.Details["message"] = validationErr.Message
	}

	json.NewEncoder(w).Encode(response)
}

// WriteValidationErrors writes multiple validation errors
func WriteValidationErrors(w http.ResponseWriter, r *http.Request, errors []error) {
	requestID := ""
	if r != nil {
		requestID = middleware.GetRequestID(r.Context())
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusBadRequest)

	validationErrors := make([]map[string]interface{}, 0)
	for _, err := range errors {
		if validationErr, ok := err.(*utils.ValidationError); ok {
			validationErrors = append(validationErrors, map[string]interface{}{
				"field":   validationErr.Field,
				"message": validationErr.Message,
			})
		}
	}

	response := ErrorResponse{
		Error:     "Validation Failed",
		Code:      "VALIDATION_ERROR",
		Details:   map[string]interface{}{"errors": validationErrors},
		RequestID: requestID,
	}

	json.NewEncoder(w).Encode(response)
}

// WriteSuccess writes a success response
func WriteSuccess(w http.ResponseWriter, data interface{}, statusCode int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(data)
}

// HandlePanic recovers from panics and writes error response
func HandlePanic(w http.ResponseWriter, r *http.Request) {
	if err := recover(); err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("internal server error: %v", err), nil)
	}
}
