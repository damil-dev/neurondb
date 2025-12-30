package handlers

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/gorilla/websocket"
	"github.com/neurondb/NeuronDesktop/api/internal/logging"
	"github.com/neurondb/NeuronDesktop/api/internal/metrics"
)

// SystemMetricsHandlers handles system metrics endpoints
type SystemMetricsHandlers struct {
	logger *logging.Logger
}

// NewSystemMetricsHandlers creates new system metrics handlers
func NewSystemMetricsHandlers(logger *logging.Logger) *SystemMetricsHandlers {
	return &SystemMetricsHandlers{
		logger: logger,
	}
}

// GetSystemMetrics returns current system metrics
func (h *SystemMetricsHandlers) GetSystemMetrics(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	systemMetrics, err := metrics.CollectSystemMetrics(ctx)
	if err != nil {
		h.logger.Error("Failed to collect system metrics", err, nil)
		http.Error(w, "Failed to collect system metrics: "+err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(systemMetrics)
}

// SystemMetricsWebSocket streams system metrics via WebSocket
func (h *SystemMetricsHandlers) SystemMetricsWebSocket(w http.ResponseWriter, r *http.Request) {
	upgrader := websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			return true // Allow all origins in development
		},
	}

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		h.logger.Error("Failed to upgrade WebSocket connection", err, nil)
		return
	}
	defer conn.Close()

	ctx := r.Context()
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	// Send initial metrics
	systemMetrics, err := metrics.CollectSystemMetrics(ctx)
	if err == nil {
		conn.WriteJSON(systemMetrics)
	}

	for {
		select {
		case <-ticker.C:
			systemMetrics, err := metrics.CollectSystemMetrics(ctx)
			if err != nil {
				h.logger.Error("Failed to collect system metrics", err, nil)
				conn.WriteJSON(map[string]interface{}{
					"error": "Failed to collect metrics: " + err.Error(),
				})
				continue
			}

			if err := conn.WriteJSON(systemMetrics); err != nil {
				h.logger.Error("Failed to write metrics to WebSocket", err, nil)
				return
			}
		case <-ctx.Done():
			return
		}
	}
}
