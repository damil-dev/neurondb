package metrics

import (
	"sync"
	"time"
)

/* Metrics collects application metrics */
type Metrics struct {
	mu sync.RWMutex

	TotalRequests      int64
	SuccessfulRequests int64
	FailedRequests     int64

	TotalResponseTime time.Duration
	MinResponseTime   time.Duration
	MaxResponseTime   time.Duration

	EndpointCounts map[string]int64
	EndpointErrors map[string]int64

	ActiveMCPConnections      int
	ActiveNeuronDBConnections int
	ActiveAgentConnections    int

	ErrorCounts map[string]int64
}

var globalMetrics = NewMetrics()

/* NewMetrics creates a new metrics instance */
func NewMetrics() *Metrics {
	return &Metrics{
		EndpointCounts:  make(map[string]int64),
		EndpointErrors:  make(map[string]int64),
		ErrorCounts:     make(map[string]int64),
		MinResponseTime: time.Hour, /* Initialize to large value */
	}
}

/* GetGlobalMetrics returns the global metrics instance */
func GetGlobalMetrics() *Metrics {
	return globalMetrics
}

/* RecordRequest records a request */
func (m *Metrics) RecordRequest(endpoint string, success bool, duration time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.TotalRequests++
	if success {
		m.SuccessfulRequests++
	} else {
		m.FailedRequests++
		m.EndpointErrors[endpoint]++
	}

	m.EndpointCounts[endpoint]++
	m.TotalResponseTime += duration

	if duration < m.MinResponseTime {
		m.MinResponseTime = duration
	}
	if duration > m.MaxResponseTime {
		m.MaxResponseTime = duration
	}
}

/* RecordError records an error */
func (m *Metrics) RecordError(errorType string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.ErrorCounts[errorType]++
}

/* SetActiveConnections sets active connection counts */
func (m *Metrics) SetActiveConnections(mcp, neurondb, agent int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.ActiveMCPConnections = mcp
	m.ActiveNeuronDBConnections = neurondb
	m.ActiveAgentConnections = agent
}

/* GetStats returns current statistics */
func (m *Metrics) GetStats() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	avgResponseTime := time.Duration(0)
	if m.TotalRequests > 0 {
		avgResponseTime = m.TotalResponseTime / time.Duration(m.TotalRequests)
	}

	return map[string]interface{}{
		"requests": map[string]interface{}{
			"total":      m.TotalRequests,
			"successful": m.SuccessfulRequests,
			"failed":     m.FailedRequests,
		},
		"response_time": map[string]interface{}{
			"avg_ms": avgResponseTime.Milliseconds(),
			"min_ms": m.MinResponseTime.Milliseconds(),
			"max_ms": m.MaxResponseTime.Milliseconds(),
		},
		"connections": map[string]interface{}{
			"mcp":      m.ActiveMCPConnections,
			"neurondb": m.ActiveNeuronDBConnections,
			"agent":    m.ActiveAgentConnections,
		},
		"endpoints": m.EndpointCounts,
		"errors":    m.ErrorCounts,
	}
}

/* Reset resets all metrics */
func (m *Metrics) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.TotalRequests = 0
	m.SuccessfulRequests = 0
	m.FailedRequests = 0
	m.TotalResponseTime = 0
	m.MinResponseTime = time.Hour
	m.MaxResponseTime = 0
	m.EndpointCounts = make(map[string]int64)
	m.EndpointErrors = make(map[string]int64)
	m.ErrorCounts = make(map[string]int64)
}
