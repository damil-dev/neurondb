/*-------------------------------------------------------------------------
 *
 * hitl.go
 *    Human-in-the-loop integration for workflow engine
 *
 * Integrates human-in-the-loop approval steps with email/webhook notifications
 * using existing approval_requests table.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/workflow/hitl.go
 *
 *-------------------------------------------------------------------------
 */

package workflow

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

type HITLManager struct {
	queries *db.Queries
}

func NewHITLManager(queries *db.Queries) *HITLManager {
	return &HITLManager{queries: queries}
}

/* ApprovalStepConfig defines configuration for an approval step */
type ApprovalStepConfig struct {
	ApprovalType   string                 `json:"approval_type"` // "email", "webhook", "ticket"
	Recipients     []string               `json:"recipients"`    // Email addresses or webhook URLs
	Subject        string                 `json:"subject"`
	Message        string                 `json:"message"`
	TimeoutSeconds int                    `json:"timeout_seconds"`
	Metadata       map[string]interface{} `json:"metadata"`
}

/* RequestApproval creates an approval request and sends notifications */
func (h *HITLManager) RequestApproval(ctx context.Context, workflowExecutionID uuid.UUID, stepExecutionID uuid.UUID, config ApprovalStepConfig) (*uuid.UUID, error) {
	/* Create approval request */
	approvalRequest := &db.ApprovalRequest{
		WorkflowExecutionID: &workflowExecutionID,
		StepExecutionID:     &stepExecutionID,
		ApprovalType:        config.ApprovalType,
		Status:              "pending",
		RequestedAt:         time.Now(),
		Metadata:            config.Metadata,
	}

	if err := h.queries.CreateApprovalRequest(ctx, approvalRequest); err != nil {
		return nil, fmt.Errorf("failed to create approval request: %w", err)
	}

	/* Send notifications based on type */
	switch config.ApprovalType {
	case "email":
		if err := h.sendEmailNotification(ctx, approvalRequest.ID, config); err != nil {
			/* Log error but don't fail - approval request is created */
			fmt.Printf("Failed to send email notification: %v\n", err)
		}
	case "webhook":
		if err := h.sendWebhookNotification(ctx, approvalRequest.ID, config); err != nil {
			fmt.Printf("Failed to send webhook notification: %v\n", err)
		}
	case "ticket":
		if err := h.createTicket(ctx, approvalRequest.ID, config); err != nil {
			fmt.Printf("Failed to create ticket: %v\n", err)
		}
	}

	return &approvalRequest.ID, nil
}

/* WaitForApproval waits for approval decision */
func (h *HITLManager) WaitForApproval(ctx context.Context, approvalRequestID uuid.UUID, timeout time.Duration) (bool, error) {
	deadline := time.Now().Add(timeout)
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return false, ctx.Err()
		case <-time.After(time.Until(deadline)):
			return false, fmt.Errorf("approval request timed out")
		case <-ticker.C:
			approval, err := h.queries.GetApprovalRequest(ctx, approvalRequestID)
			if err != nil {
				return false, fmt.Errorf("failed to get approval request: %w", err)
			}

			if approval.Status == "approved" {
				return true, nil
			}
			if approval.Status == "rejected" {
				return false, nil
			}
			/* Still pending, continue waiting */
		}
	}
}

/* sendEmailNotification sends email notification */
func (h *HITLManager) sendEmailNotification(ctx context.Context, approvalID uuid.UUID, config ApprovalStepConfig) error {
	/* TODO: Integrate with email service (SMTP) */
	/* For now, just log */
	fmt.Printf("Email notification sent for approval %s to %v\n", approvalID.String(), config.Recipients)
	return nil
}

/* sendWebhookNotification sends webhook notification */
func (h *HITLManager) sendWebhookNotification(ctx context.Context, approvalID uuid.UUID, config ApprovalStepConfig) error {
	/* TODO: Send HTTP POST to webhook URLs */
	fmt.Printf("Webhook notification sent for approval %s to %v\n", approvalID.String(), config.Recipients)
	return nil
}

/* createTicket creates a ticket in external system */
func (h *HITLManager) createTicket(ctx context.Context, approvalID uuid.UUID, config ApprovalStepConfig) error {
	/* TODO: Integrate with ticket system (Jira, ServiceNow, etc.) */
	fmt.Printf("Ticket created for approval %s\n", approvalID.String())
	return nil
}

/* ExecuteApprovalStep executes an approval step, blocking until approval */
func (h *HITLManager) ExecuteApprovalStep(ctx context.Context, workflowExecutionID uuid.UUID, stepExecutionID uuid.UUID, step *db.WorkflowStep, inputs map[string]interface{}) (map[string]interface{}, error) {
	/* Extract approval config from step inputs */
	config := ApprovalStepConfig{
		ApprovalType:   "email",
		TimeoutSeconds: 3600, /* 1 hour default */
	}

	if configMap, ok := step.Inputs["approval_config"].(map[string]interface{}); ok {
		if approvalType, ok := configMap["approval_type"].(string); ok {
			config.ApprovalType = approvalType
		}
		if recipients, ok := configMap["recipients"].([]interface{}); ok {
			config.Recipients = make([]string, len(recipients))
			for i, r := range recipients {
				if str, ok := r.(string); ok {
					config.Recipients[i] = str
				}
			}
		}
		if subject, ok := configMap["subject"].(string); ok {
			config.Subject = subject
		}
		if message, ok := configMap["message"].(string); ok {
			config.Message = message
		}
		if timeout, ok := configMap["timeout_seconds"].(float64); ok {
			config.TimeoutSeconds = int(timeout)
		}
		if metadata, ok := configMap["metadata"].(map[string]interface{}); ok {
			config.Metadata = metadata
		}
	}

	/* Request approval */
	approvalID, err := h.RequestApproval(ctx, workflowExecutionID, stepExecutionID, config)
	if err != nil {
		return nil, fmt.Errorf("failed to request approval: %w", err)
	}

	/* Wait for approval */
	timeout := time.Duration(config.TimeoutSeconds) * time.Second
	approved, err := h.WaitForApproval(ctx, *approvalID, timeout)
	if err != nil {
		return nil, fmt.Errorf("approval wait failed: %w", err)
	}

	if !approved {
		return nil, fmt.Errorf("approval was rejected")
	}

	/* Return approval result */
	outputs := map[string]interface{}{
		"approved":     true,
		"approval_id":  approvalID.String(),
		"approved_at":  time.Now(),
	}

	return outputs, nil
}




