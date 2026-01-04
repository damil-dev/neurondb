package initialization

import (
	"time"

	"github.com/neurondb/NeuronDesktop/api/internal/logging"
)

/* BootstrapMetrics tracks bootstrap performance metrics */
type BootstrapMetrics struct {
	StartTime           time.Time
	EndTime             time.Time
	Duration            time.Duration
	AdminUserDuration   time.Duration
	ProfileDuration     time.Duration
	SchemaDuration      time.Duration
	ValidationDuration  time.Duration
	HealthCheckDuration time.Duration
	TotalSteps          int
	SuccessfulSteps     int
	FailedSteps         int
}

/* NewBootstrapMetrics creates a new metrics tracker */
func NewBootstrapMetrics() *BootstrapMetrics {
	return &BootstrapMetrics{
		StartTime: time.Now(),
	}
}

/* Finish marks the bootstrap as complete and calculates final metrics */
func (bm *BootstrapMetrics) Finish() {
	bm.EndTime = time.Now()
	bm.Duration = bm.EndTime.Sub(bm.StartTime)
}

/* LogMetrics logs the bootstrap metrics */
func (bm *BootstrapMetrics) LogMetrics(logger *logging.Logger) {
	logger.Info("Bootstrap metrics", map[string]interface{}{
		"total_duration":        bm.Duration.String(),
		"admin_user_duration":   bm.AdminUserDuration.String(),
		"profile_duration":      bm.ProfileDuration.String(),
		"schema_duration":       bm.SchemaDuration.String(),
		"validation_duration":   bm.ValidationDuration.String(),
		"health_check_duration": bm.HealthCheckDuration.String(),
		"total_steps":           bm.TotalSteps,
		"successful_steps":      bm.SuccessfulSteps,
		"failed_steps":          bm.FailedSteps,
		"success_rate":          float64(bm.SuccessfulSteps) / float64(bm.TotalSteps) * 100,
	})
}

/* TrackStep tracks a step execution */
func (bm *BootstrapMetrics) TrackStep(name string, duration time.Duration, success bool) {
	bm.TotalSteps++
	if success {
		bm.SuccessfulSteps++
	} else {
		bm.FailedSteps++
	}

	switch name {
	case "admin_user":
		bm.AdminUserDuration = duration
	case "profile":
		bm.ProfileDuration = duration
	case "schema":
		bm.SchemaDuration = duration
	case "validation":
		bm.ValidationDuration = duration
	case "health_check":
		bm.HealthCheckDuration = duration
	}
}
