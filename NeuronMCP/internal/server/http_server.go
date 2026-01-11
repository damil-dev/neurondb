/*-------------------------------------------------------------------------
 *
 * http_server.go
 *    Standalone HTTP server for NeuronMCP metrics (runs in parallel with stdio)
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/server/http_server.go
 *
 *-------------------------------------------------------------------------
 */

package server

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"time"
)

/* HTTPServer provides HTTP endpoints for health and metrics */
type HTTPServer struct {
	server            *http.Server
	prometheusHandler http.Handler
}

/* NewHTTPServer creates a new HTTP server for metrics */
func NewHTTPServer(addr string, prometheusHandler http.Handler) *HTTPServer {
	mux := http.NewServeMux()
	
	/* Health endpoint */
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{"status":"ok"}`)
	})
	
	/* Prometheus metrics endpoint */
	if prometheusHandler != nil {
		mux.Handle("/metrics", prometheusHandler)
	}
	
	return &HTTPServer{
		server: &http.Server{
			Addr:         addr,
			Handler:      mux,
			ReadTimeout:  5 * time.Second,
			WriteTimeout: 10 * time.Second,
			IdleTimeout:  60 * time.Second,
		},
		prometheusHandler: prometheusHandler,
	}
}

/* Start starts the HTTP server in a goroutine */
func (h *HTTPServer) Start() {
	go func() {
		if err := h.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			/* Log error to stderr - never write to stdout as it breaks MCP protocol */
			fmt.Fprintf(os.Stderr, "HTTP metrics server error: %v\n", err)
		}
	}()
}

/* Shutdown gracefully shuts down the HTTP server */
func (h *HTTPServer) Shutdown(ctx context.Context) error {
	return h.server.Shutdown(ctx)
}



