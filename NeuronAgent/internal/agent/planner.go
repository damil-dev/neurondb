/*-------------------------------------------------------------------------
 *
 * planner.go
 *    Advanced planning system with LLM-based task decomposition
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/planner.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

type Planner struct {
	maxIterations int
	llm           *LLMClient
}

func NewPlanner() *Planner {
	return &Planner{
		maxIterations: 10, /* Prevent infinite loops */
		llm:           nil, /* Will be set by runtime */
	}
}

/* NewPlannerWithLLM creates a planner with LLM support */
func NewPlannerWithLLM(llm *LLMClient) *Planner {
	return &Planner{
		maxIterations: 10,
		llm:           llm,
	}
}

/* Plan creates a multi-step plan for complex tasks using LLM */
func (p *Planner) Plan(ctx context.Context, userMessage string, availableTools []string) ([]PlanStep, error) {
	/* Validate input */
	if userMessage == "" {
		return nil, fmt.Errorf("planning failed: user_message_empty=true")
	}
	if len(userMessage) > 50000 {
		return nil, fmt.Errorf("planning failed: user_message_too_large=true, length=%d, max_length=50000", len(userMessage))
	}

	/* If no LLM, fall back to simple plan */
	if p.llm == nil {
		return p.simplePlan(userMessage), nil
	}

	/* Build planning prompt */
	toolsList := strings.Join(availableTools, ", ")
	prompt := fmt.Sprintf(`You are a task planning assistant. Break down the following task into a series of steps.
Each step should specify:
1. The action to take
2. Which tool to use (if any) from: %s
3. The parameters for that tool

Task: %s

Respond with a JSON array of steps, each with:
- "action": description of what to do
- "tool": tool name to use (or empty string if no tool)
- "payload": object with tool parameters

Example format:
[
  {"action": "Search for information", "tool": "sql", "payload": {"query": "SELECT * FROM table"}},
  {"action": "Process results", "tool": "", "payload": {}}
]`, toolsList, userMessage)

	/* Generate plan using LLM */
	llmConfig := map[string]interface{}{
		"temperature": 0.3, /* Lower temperature for more structured output */
		"max_tokens":  2000,
	}

	response, err := p.llm.Generate(ctx, "gpt-4", prompt, llmConfig)
	if err != nil {
		/* Fallback to simple plan on error */
		return p.simplePlan(userMessage), nil
	}

	/* Parse LLM response */
	steps, err := p.parsePlanResponse(response.Content)
	if err != nil {
		/* Fallback to simple plan on parse error */
		return p.simplePlan(userMessage), nil
	}

	/* Validate and optimize plan */
	steps = p.validatePlan(steps, availableTools)

	return steps, nil
}

/* simplePlan creates a simple single-step plan */
func (p *Planner) simplePlan(userMessage string) []PlanStep {
	return []PlanStep{
		{
			Action:  "execute",
			Tool:    "",
			Payload: map[string]interface{}{"query": userMessage},
		},
	}
}

/* parsePlanResponse parses LLM response into plan steps */
func (p *Planner) parsePlanResponse(response string) ([]PlanStep, error) {
	/* Try to extract JSON from response */
	response = strings.TrimSpace(response)
	
	/* Find JSON array in response */
	start := strings.Index(response, "[")
	end := strings.LastIndex(response, "]")
	if start == -1 || end == -1 || end <= start {
		return nil, fmt.Errorf("plan parsing failed: no_json_array_found=true, response_length=%d", len(response))
	}

	jsonStr := response[start : end+1]
	
	var steps []PlanStep
	if err := json.Unmarshal([]byte(jsonStr), &steps); err != nil {
		return nil, fmt.Errorf("plan parsing failed: json_unmarshal_error=true, error=%w", err)
	}

	return steps, nil
}

/* validatePlan validates and optimizes plan steps */
func (p *Planner) validatePlan(steps []PlanStep, availableTools []string) []PlanStep {
	if len(steps) == 0 {
		return steps
	}

	validSteps := make([]PlanStep, 0, len(steps))
	toolSet := make(map[string]bool)
	for _, tool := range availableTools {
		toolSet[tool] = true
	}

	for _, step := range steps {
		/* Validate action */
		if step.Action == "" {
			step.Action = "execute" /* Default action */
		}

		/* Validate tool if specified */
		if step.Tool != "" && !toolSet[step.Tool] {
			/* Skip invalid tool steps but log warning */
			continue
		}

		/* Ensure payload is not nil */
		if step.Payload == nil {
			step.Payload = make(map[string]interface{})
		}

		validSteps = append(validSteps, step)
	}

	/* Limit to max iterations */
	if len(validSteps) > p.maxIterations {
		validSteps = validSteps[:p.maxIterations]
	}

	/* Ensure at least one step */
	if len(validSteps) == 0 {
		validSteps = []PlanStep{
			{
				Action:  "execute",
				Tool:    "",
				Payload: make(map[string]interface{}),
			},
		}
	}

	return validSteps
}

type PlanStep struct {
	Action  string                 `json:"action"`
	Tool    string                 `json:"tool"`
	Payload map[string]interface{} `json:"payload"`
}

/* ExecutePlan executes a multi-step plan */
func (p *Planner) ExecutePlan(ctx context.Context, steps []PlanStep, executor func(step PlanStep) (interface{}, error)) ([]interface{}, error) {
	var results []interface{}
	iterations := 0

	for i, step := range steps {
		if iterations >= p.maxIterations {
			return results, fmt.Errorf("max iterations reached: completed_steps=%d, total_steps=%d", i, len(steps))
		}

		result, err := executor(step)
		if err != nil {
			return results, fmt.Errorf("step %d failed: action='%s', tool='%s', error=%w", i+1, step.Action, step.Tool, err)
		}

		results = append(results, result)
		iterations++
	}

	return results, nil
}

/* Helper functions */
func floatPtr(f float64) *float64 {
	return &f
}

func intPtr(i int) *int {
	return &i
}

