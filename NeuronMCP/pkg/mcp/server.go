package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
)

// HandlerFunc is a function that handles an MCP request
type HandlerFunc func(ctx context.Context, params json.RawMessage) (interface{}, error)

// Server is an MCP protocol server
type Server struct {
	transport *StdioTransport
	handlers  map[string]HandlerFunc
	info      ServerInfo
	caps      ServerCapabilities
}

// NewServer creates a new MCP server
func NewServer(name, version string) *Server {
	return &Server{
		transport: NewStdioTransport(),
		handlers:  make(map[string]HandlerFunc),
		info: ServerInfo{
			Name:    name,
			Version: version,
		},
		caps: ServerCapabilities{
			Tools:     make(map[string]interface{}),
			Resources: make(map[string]interface{}),
		},
	}
}

// SetHandler registers a handler for a method
func (s *Server) SetHandler(method string, handler HandlerFunc) {
	s.handlers[method] = handler
}

// SetCapabilities sets server capabilities
func (s *Server) SetCapabilities(caps ServerCapabilities) {
	s.caps = caps
}

// HandleInitialize handles the initialize request
func (s *Server) HandleInitialize(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var req InitializeRequest
	if err := json.Unmarshal(params, &req); err != nil {
		return nil, fmt.Errorf("failed to parse initialize request: %w", err)
	}

	return InitializeResponse{
		ProtocolVersion: ProtocolVersion,
		Capabilities:    s.caps,
		ServerInfo:      s.info,
	}, nil
}

// Run starts the server and processes requests
func (s *Server) Run(ctx context.Context) error {
	// Register initialize handler
	s.SetHandler("initialize", s.HandleInitialize)
	
	var initializedSent bool

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			req, err := s.transport.ReadMessage()
			if err != nil {
				if err == io.EOF {
					return nil
				}
				// Check if it's an EOF error string
				if err.Error() == "EOF" {
					return nil
				}
				s.transport.WriteError(err)
				continue
			}

			// Handle initialize specially - send initialized notification
			if req.Method == "initialize" && !initializedSent {
				resp := s.handleRequest(ctx, req)
				if resp.Error == nil {
					// Send initialized notification
					if err := s.transport.WriteNotification("notifications/initialized", nil); err != nil {
						s.transport.WriteError(err)
					}
					initializedSent = true
				}
				
				// Write response if it's a request (has ID)
				if !IsNotification(req) {
					if err := s.transport.WriteMessage(resp); err != nil {
						s.transport.WriteError(err)
						continue
					}
				}
			} else {
				// Handle other requests
				resp := s.handleRequest(ctx, req)
				
				// Only send response if it's a request (has ID), not a notification
				if !IsNotification(req) {
					if err := s.transport.WriteMessage(resp); err != nil {
						s.transport.WriteError(err)
						continue
					}
				}
			}
		}
	}
}

func (s *Server) handleRequest(ctx context.Context, req *JSONRPCRequest) *JSONRPCResponse {
	// Validate request
	if err := ValidateRequest(req); err != nil {
		return CreateErrorResponse(req.ID, ErrCodeInvalidRequest, err.Error(), nil)
	}

	// Find handler
	handler, exists := s.handlers[req.Method]
	if !exists {
		return CreateErrorResponse(req.ID, ErrCodeMethodNotFound,
			fmt.Sprintf("method not found: %s", req.Method), nil)
	}

	// Execute handler
	result, err := handler(ctx, req.Params)
	if err != nil {
		return CreateErrorResponse(req.ID, ErrCodeInternalError, err.Error(), nil)
	}

	return CreateResponse(req.ID, result)
}

