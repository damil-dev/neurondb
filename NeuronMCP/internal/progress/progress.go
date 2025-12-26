/*-------------------------------------------------------------------------
 *
 * progress.go
 *    Progress tracking for NeuronMCP
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/progress/progress.go
 *
 *-------------------------------------------------------------------------
 */

package progress

import (
	"fmt"
	"sync"
	"time"
)

/* ProgressStatus represents progress status */
type ProgressStatus struct {
	ID        string                 `json:"id"`
	Status    string                 `json:"status"` /* pending, running, completed, failed */
	Progress  float64                `json:"progress"` /* 0.0 to 1.0 */
	Message   string                 `json:"message,omitempty"`
	StartedAt time.Time              `json:"startedAt"`
	UpdatedAt time.Time              `json:"updatedAt"`
	CompletedAt *time.Time           `json:"completedAt,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	Error     string                 `json:"error,omitempty"`
}

/* Tracker tracks progress for operations */
type Tracker struct {
	progresses map[string]*ProgressStatus
	mu         sync.RWMutex
}

/* NewTracker creates a new progress tracker */
func NewTracker() *Tracker {
	return &Tracker{
		progresses: make(map[string]*ProgressStatus),
	}
}

/* Start starts tracking progress */
func (t *Tracker) Start(id string, message string) *ProgressStatus {
	t.mu.Lock()
	defer t.mu.Unlock()

	now := time.Now()
	progress := &ProgressStatus{
		ID:        id,
		Status:    "running",
		Progress:  0.0,
		Message:   message,
		StartedAt: now,
		UpdatedAt: now,
		Metadata:  make(map[string]interface{}),
	}

	t.progresses[id] = progress
	return progress
}

/* Update updates progress */
func (t *Tracker) Update(id string, progress float64, message string) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	p, exists := t.progresses[id]
	if !exists {
		return fmt.Errorf("progress not found: %s", id)
	}

	if progress < 0.0 {
		progress = 0.0
	}
	if progress > 1.0 {
		progress = 1.0
	}

	p.Progress = progress
	if message != "" {
		p.Message = message
	}
	p.UpdatedAt = time.Now()

	return nil
}

/* Complete marks progress as completed */
func (t *Tracker) Complete(id string, message string) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	p, exists := t.progresses[id]
	if !exists {
		return fmt.Errorf("progress not found: %s", id)
	}

	now := time.Now()
	p.Status = "completed"
	p.Progress = 1.0
	if message != "" {
		p.Message = message
	}
	p.UpdatedAt = now
	p.CompletedAt = &now

	return nil
}

/* Fail marks progress as failed */
func (t *Tracker) Fail(id string, err error) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	p, exists := t.progresses[id]
	if !exists {
		return fmt.Errorf("progress not found: %s", id)
	}

	now := time.Now()
	p.Status = "failed"
	p.UpdatedAt = now
	p.CompletedAt = &now
	if err != nil {
		p.Error = err.Error()
	}

	return nil
}

/* Get gets progress status */
func (t *Tracker) Get(id string) (*ProgressStatus, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	p, exists := t.progresses[id]
	if !exists {
		return nil, fmt.Errorf("progress not found: %s", id)
	}

	return p, nil
}

/* List lists all progress statuses */
func (t *Tracker) List() []*ProgressStatus {
	t.mu.RLock()
	defer t.mu.RUnlock()

	statuses := make([]*ProgressStatus, 0, len(t.progresses))
	for _, p := range t.progresses {
		statuses = append(statuses, p)
	}

	return statuses
}

/* Cleanup removes old completed progress entries */
func (t *Tracker) Cleanup(maxAge time.Duration) {
	t.mu.Lock()
	defer t.mu.Unlock()

	now := time.Now()
	for id, p := range t.progresses {
		if p.CompletedAt != nil {
			if now.Sub(*p.CompletedAt) > maxAge {
				delete(t.progresses, id)
			}
		}
	}
}

