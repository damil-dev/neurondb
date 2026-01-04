package handlers

import (
	"net/http"

	"github.com/neurondb/NeuronDesktop/api/internal/metrics"
)

/* MetricsHandlers handles metrics endpoints */
type MetricsHandlers struct {
	metrics *metrics.Metrics
}

/* NewMetricsHandlers creates new metrics handlers */
func NewMetricsHandlers() *MetricsHandlers {
	return &MetricsHandlers{
		metrics: metrics.GetGlobalMetrics(),
	}
}

/* GetMetrics returns current metrics */
func (h *MetricsHandlers) GetMetrics(w http.ResponseWriter, r *http.Request) {
	stats := h.metrics.GetStats()
	WriteSuccess(w, stats, http.StatusOK)
}

/* ResetMetrics resets all metrics */
func (h *MetricsHandlers) ResetMetrics(w http.ResponseWriter, r *http.Request) {
	h.metrics.Reset()
	WriteSuccess(w, map[string]string{"message": "Metrics reset"}, http.StatusOK)
}
