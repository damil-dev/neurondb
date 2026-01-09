package metrics

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	// HTTP request metrics
	httpRequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondesktop_api_http_requests_total",
			Help: "Total number of HTTP requests",
		},
		[]string{"method", "endpoint", "status"},
	)

	httpRequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "neurondesktop_api_http_request_duration_seconds",
			Help:    "HTTP request duration in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method", "endpoint"},
	)

	// Active connections gauge
	activeConnections = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "neurondesktop_api_active_connections",
			Help: "Number of active connections",
		},
		[]string{"type"},
	)
)

// RecordHTTPRequest records an HTTP request
func RecordHTTPRequest(method, endpoint string, statusCode int, durationSeconds float64) {
	status := "unknown"
	if statusCode >= 200 && statusCode < 300 {
		status = "2xx"
	} else if statusCode >= 300 && statusCode < 400 {
		status = "3xx"
	} else if statusCode >= 400 && statusCode < 500 {
		status = "4xx"
	} else if statusCode >= 500 {
		status = "5xx"
	}

	httpRequestsTotal.WithLabelValues(method, endpoint, status).Inc()
	httpRequestDuration.WithLabelValues(method, endpoint).Observe(durationSeconds)
}

// SetActiveConnections sets the number of active connections by type
func SetActiveConnections(connType string, count float64) {
	activeConnections.WithLabelValues(connType).Set(count)
}

// Handler returns the Prometheus metrics HTTP handler
func Handler() http.Handler {
	return promhttp.Handler()
}


