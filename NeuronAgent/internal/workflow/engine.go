/*-------------------------------------------------------------------------
 *
 * engine.go
 *    Workflow DAG engine for NeuronAgent
 *
 * Provides DAG workflow execution with steps, inputs, outputs, dependencies,
 * retries, and idempotency keys.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/workflow/engine.go
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

type Engine struct {
	queries *db.Queries
}

func NewEngine(queries *db.Queries) *Engine {
	return &Engine{queries: queries}
}

/* ExecuteWorkflow executes a workflow with given inputs */
func (e *Engine) ExecuteWorkflow(ctx context.Context, workflowID uuid.UUID, triggerType string, triggerData map[string]interface{}, inputs map[string]interface{}) (*db.WorkflowExecution, error) {
	/* Create execution */
	execution := &db.WorkflowExecution{
		WorkflowID:  workflowID,
		Status:      "pending",
		TriggerType: triggerType,
		TriggerData: triggerData,
		Inputs:      inputs,
		Outputs:     make(map[string]interface{}),
	}

	/* Save execution */
	/* TODO: Implement CreateWorkflowExecution query */
	
	/* Load workflow and steps */
	/* TODO: Load workflow and steps from database */
	
	/* Build dependency graph */
	/* TODO: Build DAG from steps */
	
	/* Execute steps in topological order */
	/* TODO: Execute steps respecting dependencies */
	
	return execution, nil
}

/* ExecuteStep executes a single workflow step */
func (e *Engine) ExecuteStep(ctx context.Context, executionID uuid.UUID, step *db.WorkflowStep, inputs map[string]interface{}) (map[string]interface{}, error) {
	/* Check idempotency */
	if step.IdempotencyKey != nil {
		/* TODO: Check if step with this idempotency key already executed */
		/* If yes, return cached result */
	}

	/* Create step execution */
	stepExecution := &db.WorkflowStepExecution{
		WorkflowExecutionID: executionID,
		WorkflowStepID:      step.ID,
		Status:              "running",
		Inputs:              inputs,
		Outputs:             make(map[string]interface{}),
		IdempotencyKey:      step.IdempotencyKey,
	}
	now := time.Now()
	stepExecution.StartedAt = &now

	/* Save step execution */
	if err := e.queries.CreateWorkflowStepExecution(ctx, stepExecution); err != nil {
		return nil, fmt.Errorf("failed to create step execution: %w", err)
	}

	/* Execute step based on type */
	var outputs map[string]interface{}
	var err error

	switch step.StepType {
	case "agent":
		outputs, err = e.executeAgentStep(ctx, step, inputs)
	case "tool":
		outputs, err = e.executeToolStep(ctx, step, inputs)
	case "approval":
		outputs, err = e.executeApprovalStep(ctx, executionID, stepExecution.ID, step, inputs)
	case "http":
		outputs, err = e.executeHTTPStep(ctx, step, inputs)
	case "sql":
		outputs, err = e.executeSQLStep(ctx, step, inputs)
	default:
		err = fmt.Errorf("unknown step type: %s", step.StepType)
	}

	/* Handle retries if error */
	if err != nil {
		stepExecution.Status = "failed"
		errorMsg := err.Error()
		stepExecution.ErrorMessage = &errorMsg
		
		retryConfig := step.RetryConfig
		if retryConfig != nil && stepExecution.RetryCount < getMaxRetries(retryConfig) {
			/* Retry with backoff */
			stepExecution.RetryCount++
			if updateErr := e.queries.UpdateWorkflowStepExecution(ctx, stepExecution); updateErr != nil {
				return nil, fmt.Errorf("failed to update step execution for retry: %w", updateErr)
			}
			
			/* Schedule retry - for now just return error, would need retry scheduler */
			return nil, fmt.Errorf("step failed, retry %d/%d: %w", stepExecution.RetryCount, getMaxRetries(retryConfig), err)
		}
		
		/* Max retries reached or no retry config */
		if updateErr := e.queries.UpdateWorkflowStepExecution(ctx, stepExecution); updateErr != nil {
			return nil, fmt.Errorf("failed to update failed step execution: %w", updateErr)
		}
		return nil, err
	}

	stepExecution.Outputs = outputs
	stepExecution.Status = "completed"
	completedAt := time.Now()
	stepExecution.CompletedAt = &completedAt

	/* Update step execution */
	if err := e.queries.UpdateWorkflowStepExecution(ctx, stepExecution); err != nil {
		return nil, fmt.Errorf("failed to update step execution: %w", err)
	}

	return outputs, nil
}

/* executeAgentStep executes an agent step */
func (e *Engine) executeAgentStep(ctx context.Context, step *db.WorkflowStep, inputs map[string]interface{}) (map[string]interface{}, error) {
	/* TODO: Extract agent_id and user_message from inputs */
	/* TODO: Execute agent via runtime */
	return make(map[string]interface{}), nil
}

/* executeToolStep executes a tool step */
func (e *Engine) executeToolStep(ctx context.Context, step *db.WorkflowStep, inputs map[string]interface{}) (map[string]interface{}, error) {
	/* TODO: Extract tool_name and args from inputs */
	/* TODO: Execute tool */
	return make(map[string]interface{}), nil
}

/* executeApprovalStep executes an approval step */
func (e *Engine) executeApprovalStep(ctx context.Context, workflowExecutionID uuid.UUID, stepExecutionID uuid.UUID, step *db.WorkflowStep, inputs map[string]interface{}) (map[string]interface{}, error) {
	hitlManager := NewHITLManager(e.queries)
	return hitlManager.ExecuteApprovalStep(ctx, workflowExecutionID, stepExecutionID, step, inputs)
}

/* executeHTTPStep executes an HTTP step */
func (e *Engine) executeHTTPStep(ctx context.Context, step *db.WorkflowStep, inputs map[string]interface{}) (map[string]interface{}, error) {
	/* TODO: Execute HTTP request */
	return make(map[string]interface{}), nil
}

/* executeSQLStep executes a SQL step */
func (e *Engine) executeSQLStep(ctx context.Context, step *db.WorkflowStep, inputs map[string]interface{}) (map[string]interface{}, error) {
	/* TODO: Execute SQL query */
	return make(map[string]interface{}), nil
}

/* CompensateStep executes compensation step for rollback */
func (e *Engine) CompensateStep(ctx context.Context, stepExecution *db.WorkflowStepExecution) error {
	if stepExecution.Status != "completed" {
		return nil /* Nothing to compensate */
	}

	step, err := e.queries.GetWorkflowStepByID(ctx, stepExecution.WorkflowStepID)
	if err != nil {
		return fmt.Errorf("failed to get workflow step: %w", err)
	}

	if step.CompensationStepID == nil {
		return nil /* No compensation step */
	}

	compensationStep, err := e.queries.GetWorkflowStepByID(ctx, *step.CompensationStepID)
	if err != nil {
		return fmt.Errorf("failed to get compensation step: %w", err)
	}

	/* Execute compensation step */
	_, err = e.ExecuteStep(ctx, stepExecution.WorkflowExecutionID, compensationStep, stepExecution.Outputs)
	if err != nil {
		return fmt.Errorf("compensation step failed: %w", err)
	}

	stepExecution.Status = "compensated"
	if err := e.queries.UpdateWorkflowStepExecution(ctx, stepExecution); err != nil {
		return fmt.Errorf("failed to update compensated step execution: %w", err)
	}

	return nil
}

/* getMaxRetries extracts max retries from retry config */
func getMaxRetries(retryConfig db.JSONBMap) int {
	if maxRetries, ok := retryConfig["max_retries"].(float64); ok {
		return int(maxRetries)
	}
	return 3 /* Default */
}

