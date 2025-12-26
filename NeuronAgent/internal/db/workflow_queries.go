/*-------------------------------------------------------------------------
 *
 * workflow_queries.go
 *    Database queries for workflow engine
 *
 * Provides database query functions for workflows, steps, executions, and schedules.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/db/workflow_queries.go
 *
 *-------------------------------------------------------------------------
 */

package db

import (
	"context"
	"database/sql"
	"fmt"

	"github.com/google/uuid"
)

/* Workflow queries */
const (
	createWorkflowQuery = `
		INSERT INTO neurondb_agent.workflows (name, dag_definition, status)
		VALUES ($1, $2::jsonb, $3)
		RETURNING id, created_at, updated_at`

	getWorkflowByIDQuery = `SELECT * FROM neurondb_agent.workflows WHERE id = $1`

	listWorkflowsQuery = `
		SELECT * FROM neurondb_agent.workflows 
		WHERE ($1::text IS NULL OR status = $1)
		ORDER BY created_at DESC`
)

/* Workflow step queries */
const (
	createWorkflowStepQuery = `
		INSERT INTO neurondb_agent.workflow_steps 
		(workflow_id, step_name, step_type, inputs, outputs, dependencies, retry_config, idempotency_key, compensation_step_id)
		VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6, $7::jsonb, $8, $9)
		RETURNING id, created_at, updated_at`

	getWorkflowStepByIDQuery = `SELECT * FROM neurondb_agent.workflow_steps WHERE id = $1`

	listWorkflowStepsQuery = `
		SELECT * FROM neurondb_agent.workflow_steps 
		WHERE workflow_id = $1 
		ORDER BY step_name`
)

/* Workflow execution queries */
const (
	createWorkflowExecutionQuery = `
		INSERT INTO neurondb_agent.workflow_executions 
		(workflow_id, status, trigger_type, trigger_data, inputs)
		VALUES ($1, $2, $3, $4::jsonb, $5::jsonb)
		RETURNING id, created_at, updated_at`

	updateWorkflowExecutionQuery = `
		UPDATE neurondb_agent.workflow_executions 
		SET status = $2, outputs = $3::jsonb, error_message = $4, completed_at = NOW(), updated_at = NOW()
		WHERE id = $1
		RETURNING updated_at`

	getWorkflowExecutionByIDQuery = `SELECT * FROM neurondb_agent.workflow_executions WHERE id = $1`
)

/* Workflow step execution queries */
const (
	createWorkflowStepExecutionQuery = `
		INSERT INTO neurondb_agent.workflow_step_executions 
		(workflow_execution_id, workflow_step_id, status, inputs, idempotency_key, started_at)
		VALUES ($1, $2, $3, $4::jsonb, $5, $6)
		RETURNING id, created_at, updated_at`

	updateWorkflowStepExecutionQuery = `
		UPDATE neurondb_agent.workflow_step_executions 
		SET status = $2, outputs = $3::jsonb, error_message = $4, retry_count = $5, completed_at = NOW(), updated_at = NOW()
		WHERE id = $1
		RETURNING updated_at`

	getWorkflowStepExecutionByIDQuery = `SELECT * FROM neurondb_agent.workflow_step_executions WHERE id = $1`

	getWorkflowStepExecutionByIdempotencyKeyQuery = `
		SELECT * FROM neurondb_agent.workflow_step_executions 
		WHERE idempotency_key = $1 AND status = 'completed'`
)

/* Workflow methods */
func (q *Queries) CreateWorkflow(ctx context.Context, workflow *Workflow) error {
	dagDefValue, err := workflow.DAGDefinition.Value()
	if err != nil {
		return fmt.Errorf("failed to convert dag_definition: %w", err)
	}

	params := []interface{}{workflow.Name, dagDefValue, workflow.Status}
	err = q.DB.GetContext(ctx, workflow, createWorkflowQuery, params...)
	if err != nil {
		return fmt.Errorf("workflow creation failed: %w", err)
	}
	return nil
}

func (q *Queries) GetWorkflowByID(ctx context.Context, id uuid.UUID) (*Workflow, error) {
	var workflow Workflow
	err := q.DB.GetContext(ctx, &workflow, getWorkflowByIDQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("workflow not found: %w", err)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get workflow: %w", err)
	}
	return &workflow, nil
}

func (q *Queries) ListWorkflows(ctx context.Context, status *string) ([]Workflow, error) {
	var workflows []Workflow
	err := q.DB.SelectContext(ctx, &workflows, listWorkflowsQuery, status)
	if err != nil {
		return nil, fmt.Errorf("failed to list workflows: %w", err)
	}
	return workflows, nil
}

/* Workflow step methods */
func (q *Queries) CreateWorkflowStep(ctx context.Context, step *WorkflowStep) error {
	inputsValue, err := step.Inputs.Value()
	if err != nil {
		return fmt.Errorf("failed to convert inputs: %w", err)
	}
	outputsValue, err := step.Outputs.Value()
	if err != nil {
		return fmt.Errorf("failed to convert outputs: %w", err)
	}
	retryConfigValue, err := step.RetryConfig.Value()
	if err != nil {
		return fmt.Errorf("failed to convert retry_config: %w", err)
	}

	params := []interface{}{
		step.WorkflowID, step.StepName, step.StepType,
		inputsValue, outputsValue, step.Dependencies,
		retryConfigValue, step.IdempotencyKey, step.CompensationStepID,
	}
	err = q.DB.GetContext(ctx, step, createWorkflowStepQuery, params...)
	if err != nil {
		return fmt.Errorf("workflow step creation failed: %w", err)
	}
	return nil
}

func (q *Queries) GetWorkflowStepByID(ctx context.Context, id uuid.UUID) (*WorkflowStep, error) {
	var step WorkflowStep
	err := q.DB.GetContext(ctx, &step, getWorkflowStepByIDQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("workflow step not found: %w", err)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get workflow step: %w", err)
	}
	return &step, nil
}

func (q *Queries) ListWorkflowSteps(ctx context.Context, workflowID uuid.UUID) ([]WorkflowStep, error) {
	var steps []WorkflowStep
	err := q.DB.SelectContext(ctx, &steps, listWorkflowStepsQuery, workflowID)
	if err != nil {
		return nil, fmt.Errorf("failed to list workflow steps: %w", err)
	}
	return steps, nil
}

/* Workflow execution methods */
func (q *Queries) CreateWorkflowExecution(ctx context.Context, execution *WorkflowExecution) error {
	triggerDataValue, err := execution.TriggerData.Value()
	if err != nil {
		return fmt.Errorf("failed to convert trigger_data: %w", err)
	}
	inputsValue, err := execution.Inputs.Value()
	if err != nil {
		return fmt.Errorf("failed to convert inputs: %w", err)
	}

	params := []interface{}{execution.WorkflowID, execution.Status, execution.TriggerType, triggerDataValue, inputsValue}
	err = q.DB.GetContext(ctx, execution, createWorkflowExecutionQuery, params...)
	if err != nil {
		return fmt.Errorf("workflow execution creation failed: %w", err)
	}
	return nil
}

func (q *Queries) UpdateWorkflowExecution(ctx context.Context, execution *WorkflowExecution) error {
	outputsValue, err := execution.Outputs.Value()
	if err != nil {
		return fmt.Errorf("failed to convert outputs: %w", err)
	}

	params := []interface{}{execution.ID, execution.Status, outputsValue, execution.ErrorMessage}
	err = q.DB.GetContext(ctx, execution, updateWorkflowExecutionQuery, params...)
	if err != nil {
		return fmt.Errorf("workflow execution update failed: %w", err)
	}
	return nil
}

func (q *Queries) GetWorkflowExecutionByID(ctx context.Context, id uuid.UUID) (*WorkflowExecution, error) {
	var execution WorkflowExecution
	err := q.DB.GetContext(ctx, &execution, getWorkflowExecutionByIDQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("workflow execution not found: %w", err)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get workflow execution: %w", err)
	}
	return &execution, nil
}

/* Workflow step execution methods */
func (q *Queries) CreateWorkflowStepExecution(ctx context.Context, stepExecution *WorkflowStepExecution) error {
	inputsValue, err := stepExecution.Inputs.Value()
	if err != nil {
		return fmt.Errorf("failed to convert inputs: %w", err)
	}

	params := []interface{}{stepExecution.WorkflowExecutionID, stepExecution.WorkflowStepID, stepExecution.Status, inputsValue, stepExecution.IdempotencyKey, stepExecution.StartedAt}
	err = q.DB.GetContext(ctx, stepExecution, createWorkflowStepExecutionQuery, params...)
	if err != nil {
		return fmt.Errorf("workflow step execution creation failed: %w", err)
	}
	return nil
}

func (q *Queries) UpdateWorkflowStepExecution(ctx context.Context, stepExecution *WorkflowStepExecution) error {
	outputsValue, err := stepExecution.Outputs.Value()
	if err != nil {
		return fmt.Errorf("failed to convert outputs: %w", err)
	}

	params := []interface{}{stepExecution.ID, stepExecution.Status, outputsValue, stepExecution.ErrorMessage, stepExecution.RetryCount}
	err = q.DB.GetContext(ctx, stepExecution, updateWorkflowStepExecutionQuery, params...)
	if err != nil {
		return fmt.Errorf("workflow step execution update failed: %w", err)
	}
	return nil
}

func (q *Queries) GetWorkflowStepExecutionByID(ctx context.Context, id uuid.UUID) (*WorkflowStepExecution, error) {
	var stepExecution WorkflowStepExecution
	err := q.DB.GetContext(ctx, &stepExecution, getWorkflowStepExecutionByIDQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("workflow step execution not found: %w", err)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get workflow step execution: %w", err)
	}
	return &stepExecution, nil
}

func (q *Queries) GetWorkflowStepExecutionByIdempotencyKey(ctx context.Context, key string) (*WorkflowStepExecution, error) {
	var stepExecution WorkflowStepExecution
	err := q.DB.GetContext(ctx, &stepExecution, getWorkflowStepExecutionByIdempotencyKeyQuery, key)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get workflow step execution by idempotency key: %w", err)
	}
	return &stepExecution, nil
}

