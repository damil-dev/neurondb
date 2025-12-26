/*-------------------------------------------------------------------------
 *
 * tracing.go
 *    OpenTelemetry tracing integration for NeuronMCP
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/observability/tracing.go
 *
 *-------------------------------------------------------------------------
 */

package observability

import (
	"context"
	"time"
)

/* Tracer provides distributed tracing capabilities */
type Tracer struct {
	enabled bool
	spans   map[string]*Span
}

/* Span represents a tracing span */
type Span struct {
	TraceID    string
	SpanID     string
	ParentID   string
	Name       string
	StartTime  time.Time
	EndTime    time.Time
	Attributes map[string]interface{}
}

/* NewTracer creates a new tracer */
func NewTracer(enabled bool) *Tracer {
	return &Tracer{
		enabled: enabled,
		spans:   make(map[string]*Span),
	}
}

/* StartSpan starts a new span */
func (t *Tracer) StartSpan(ctx context.Context, name string) (context.Context, string) {
	if !t.enabled {
		return ctx, ""
	}

	spanID := generateID()
	traceID := getOrCreateTraceID(ctx)

	span := &Span{
		TraceID:    traceID,
		SpanID:     spanID,
		Name:       name,
		StartTime:  time.Now(),
		Attributes: make(map[string]interface{}),
	}

	/* Get parent span ID if exists */
	if parentSpanID := ctx.Value("span_id"); parentSpanID != nil {
		span.ParentID = parentSpanID.(string)
	}

	t.spans[spanID] = span

	/* Add to context */
	ctx = context.WithValue(ctx, "trace_id", traceID)
	ctx = context.WithValue(ctx, "span_id", spanID)

	return ctx, spanID
}

/* EndSpan ends a span */
func (t *Tracer) EndSpan(spanID string) {
	if !t.enabled || spanID == "" {
		return
	}

	if span, exists := t.spans[spanID]; exists {
		span.EndTime = time.Now()
		/* In production, send to tracing backend (e.g., Jaeger, Zipkin) */
		delete(t.spans, spanID)
	}
}

/* AddAttribute adds an attribute to a span */
func (t *Tracer) AddAttribute(spanID string, key string, value interface{}) {
	if !t.enabled || spanID == "" {
		return
	}

	if span, exists := t.spans[spanID]; exists {
		span.Attributes[key] = value
	}
}

/* GetSpan returns a span by ID */
func (t *Tracer) GetSpan(spanID string) (*Span, bool) {
	if !t.enabled || spanID == "" {
		return nil, false
	}

	span, exists := t.spans[spanID]
	return span, exists
}

/* GetTraceIDFromContext gets trace ID from context */
func GetTraceIDFromContext(ctx context.Context) (string, bool) {
	traceID, ok := ctx.Value("trace_id").(string)
	return traceID, ok
}

/* getOrCreateTraceID gets trace ID from context or creates a new one */
func getOrCreateTraceID(ctx context.Context) string {
	if traceID, ok := ctx.Value("trace_id").(string); ok {
		return traceID
	}
	return generateID()
}

/* generateID generates a unique ID */
func generateID() string {
	/* Simple ID generation - in production, use proper UUID library */
	return time.Now().Format("20060102150405") + "-" + randomString(8)
}

/* randomString generates a random string */
func randomString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	b := make([]byte, length)
	now := time.Now().UnixNano()
	for i := range b {
		b[i] = charset[int(now+int64(i))%len(charset)]
	}
	return string(b)
}

