/*-------------------------------------------------------------------------
 *
 * transport.go
 *    Database operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/pkg/mcp/transport.go
 *
 *-------------------------------------------------------------------------
 */

package mcp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
)

/* StdioTransport handles MCP communication over stdio */
type StdioTransport struct {
	stdin          *bufio.Reader
	stdout         *bufio.Writer
	stderr         io.Writer
	maxRequestSize int64
	clientUsesHeaders bool /* Track if client uses Content-Length headers */
}

/* NewStdioTransport creates a new stdio transport */
func NewStdioTransport() *StdioTransport {
	return NewStdioTransportWithMaxSize(0) /* Default: unlimited */
}

/* NewStdioTransportWithMaxSize creates a new stdio transport with max request size */
func NewStdioTransportWithMaxSize(maxRequestSize int64) *StdioTransport {
	/* Use a small buffered writer for stdout
	 * MCP protocol requires precise control over when data is sent
	 * Small buffer (1KB) ensures quick flushing while maintaining efficiency
	 * Note: Must flush after each message to ensure immediate transmission
	 */
	return &StdioTransport{
		stdin:             bufio.NewReader(os.Stdin),
		stdout:            bufio.NewWriterSize(os.Stdout, 1024), /* 1KB buffer for efficiency with immediate flushing */
		stderr:            os.Stderr,
		maxRequestSize:    maxRequestSize,
		clientUsesHeaders: true, /* Default to Content-Length headers (MCP standard) */
	}
}

/* ReadMessage reads a JSON-RPC message from stdin */
func (t *StdioTransport) ReadMessage() (*JSONRPCRequest, error) {
	t.WriteError(fmt.Errorf("DEBUG: ReadMessage() called, starting to read headers"))
  /* Read headers */
	var contentLength int
	headerLines := 0
 	maxHeaders := 10 /* Prevent infinite loop */
	
	for headerLines < maxHeaders {
		t.WriteError(fmt.Errorf("DEBUG: Reading header line %d", headerLines))
		line, err := t.stdin.ReadString('\n')
		t.WriteError(fmt.Errorf("DEBUG: Read header line: %q, err=%v", line, err))
		if err != nil {
			if err == io.EOF {
     /* If we got EOF while reading headers and haven't found Content-Length, */
     /* this means the connection closed */
				if contentLength == 0 {
					return nil, io.EOF
				}
     /* If we have Content-Length but got EOF, it's still EOF */
				return nil, io.EOF
			}
			return nil, fmt.Errorf("failed to read header: %w", err)
		}
		headerLines++

   /* Remove trailing newline/carriage return */
		line = strings.TrimRight(line, "\r\n")
		
   /* Backward compatibility: Check if the first line is JSON (starts with '{') */
   /* Standard MCP protocol always uses Content-Length headers */
   /* Claude Desktop sends with Content-Length, but we support direct JSON for compatibility */
		if headerLines == 1 && strings.HasPrefix(strings.TrimSpace(line), "{") {
			t.WriteError(fmt.Errorf("DEBUG: First line is JSON (fallback mode, no Content-Length headers), parsing directly"))
			/* Client doesn't use headers - we'll respond without headers too */
			t.clientUsesHeaders = false
			
			/* Enforce maximum request size for JSON without Content-Length */
			if t.maxRequestSize > 0 && int64(len(line)) > t.maxRequestSize {
				return nil, fmt.Errorf("request size %d exceeds maximum allowed size %d bytes", len(line), t.maxRequestSize)
			}
			
			/* Parse the JSON directly */
			return ParseRequest([]byte(line))
		}
		
   /* Empty line indicates end of headers */
		if line == "" {
			break
		}

   /* Parse Content-Length */
		lineLower := strings.ToLower(line)
		if strings.HasPrefix(lineLower, "content-length:") {
    /* Try both capitalized and lowercase */
			if _, err := fmt.Sscanf(line, "Content-Length: %d", &contentLength); err != nil {
				if _, err := fmt.Sscanf(line, "content-length: %d", &contentLength); err != nil {
					return nil, fmt.Errorf("invalid Content-Length header: %s", line)
				}
			}
		}
   /* Skip other headers (Content-Type, etc.) */
	}

	if contentLength <= 0 {
   /* This can happen if we read an empty line before getting Content-Length */
   /* or if there's malformed input. Return error but don't treat as fatal. */
		t.WriteError(fmt.Errorf("DEBUG: No valid Content-Length found after %d headers", headerLines))
		return nil, fmt.Errorf("missing or invalid Content-Length header")
	}

	/* Client uses Content-Length headers */
	t.clientUsesHeaders = true
	t.WriteError(fmt.Errorf("DEBUG: Headers parsed, contentLength=%d, reading body", contentLength))
	
	/* Enforce maximum request size */
	if t.maxRequestSize > 0 && int64(contentLength) > t.maxRequestSize {
		return nil, fmt.Errorf("request size %d exceeds maximum allowed size %d bytes", contentLength, t.maxRequestSize)
	}
	
	/* Read message body */
	body := make([]byte, contentLength)
	if _, err := io.ReadFull(t.stdin, body); err != nil {
		if err == io.EOF {
			return nil, io.EOF
		}
		return nil, fmt.Errorf("failed to read message body: %w", err)
	}

	return ParseRequest(body)
}

/* WriteMessage writes a JSON-RPC message to stdout */
func (t *StdioTransport) WriteMessage(resp *JSONRPCResponse) error {
	if resp == nil {
		return fmt.Errorf("cannot write nil response")
	}
	
	data, err := SerializeResponse(resp)
	if err != nil {
		return fmt.Errorf("failed to serialize response: %w", err)
	}

	t.WriteError(fmt.Errorf("DEBUG: Writing response: %s (clientUsesHeaders=%v)", string(data), t.clientUsesHeaders))

  /* WORKAROUND: Claude Desktop appears to have issues parsing Content-Length headers in responses
   * Even though it sends requests with headers, it expects responses without headers
   * This is a known issue with some MCP clients - always respond without headers for compatibility
   * TODO: This should be fixed in Claude Desktop to properly support MCP spec
   */
	forceNoHeaders := os.Getenv("NEURONMCP_FORCE_NO_HEADERS") == "true"
	
  /* IMPORTANT: Claude Desktop sends requests with Content-Length headers
   * Per MCP spec, we should respond with headers when client uses headers
   * Previous assumption that Claude Desktop can't parse headers was incorrect
   * Always match the client's header usage for proper MCP protocol compliance
   */
	if forceNoHeaders {
		/* Only disable headers if explicitly forced */
		t.clientUsesHeaders = false
	}
	
	if t.clientUsesHeaders {
    /* MCP protocol format: Content-Length: <len>\r\n\r\n<body> */
    /* Per MCP spec: headers must end with \r\n\r\n (CRLF CRLF) */
		header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(data))
		headerBytes := []byte(header)
		
    /* Write header first, then body, then flush */
		if _, err := t.stdout.Write(headerBytes); err != nil {
			return fmt.Errorf("failed to write header: %w", err)
		}
	}
	
  /* Write JSON body */
	if _, err := t.stdout.Write(data); err != nil {
		return fmt.Errorf("failed to write body: %w", err)
	}
	
  /* When sending without headers (direct JSON), add a newline for compatibility
   * Some clients expect responses to end with a newline when using direct JSON format
   */
	if !t.clientUsesHeaders {
		if _, err := t.stdout.Write([]byte("\n")); err != nil {
			return fmt.Errorf("failed to write newline: %w", err)
		}
	}

  /* CRITICAL: Flush immediately after writing to ensure message is sent
   * Without flushing, the buffer might hold the data and cause protocol issues
   */
	if err := t.stdout.Flush(); err != nil {
		return fmt.Errorf("failed to flush stdout: %w", err)
	}

	t.WriteError(fmt.Errorf("DEBUG: Response written and flushed (format: %s)", map[bool]string{true: "Content-Length headers", false: "direct JSON"}[t.clientUsesHeaders]))

	return nil
}

/* WriteNotification writes a JSON-RPC notification (no response expected) */
func (t *StdioTransport) WriteNotification(method string, params interface{}) error {
	notification := map[string]interface{}{
		"jsonrpc": "2.0",
		"method":  method,
	}
	
	if params != nil {
		notification["params"] = params
	}
	
	data, err := json.Marshal(notification)
	if err != nil {
		return fmt.Errorf("failed to serialize notification: %w", err)
	}

	t.WriteError(fmt.Errorf("DEBUG: Writing notification: %s (clientUsesHeaders=%v)", string(data), t.clientUsesHeaders))

  /* WORKAROUND: Check environment variable for forced no headers */
	forceNoHeaders := os.Getenv("NEURONMCP_FORCE_NO_HEADERS") == "true"
	
  /* Match client's request format (notifications follow same format) */
	if t.clientUsesHeaders && !forceNoHeaders {
    /* MCP protocol format: Content-Length: <len>\r\n\r\n<body> */
		header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(data))
		headerBytes := []byte(header)
		if _, err := t.stdout.Write(headerBytes); err != nil {
			return fmt.Errorf("failed to write header: %w", err)
		}
	}
	
  /* Write JSON body */
	if _, err := t.stdout.Write(data); err != nil {
		return fmt.Errorf("failed to write body: %w", err)
	}
	
  /* When sending without headers (direct JSON), add a newline for compatibility */
	if !t.clientUsesHeaders || forceNoHeaders {
		if _, err := t.stdout.Write([]byte("\n")); err != nil {
			return fmt.Errorf("failed to write newline: %w", err)
		}
	}

  /* CRITICAL: Flush immediately after writing to ensure message is sent */
	if err := t.stdout.Flush(); err != nil {
		return fmt.Errorf("failed to flush stdout: %w", err)
	}

	t.WriteError(fmt.Errorf("DEBUG: Notification written and flushed"))

	return nil
}

/* WriteError writes an error to stderr (only in debug mode) */
func (t *StdioTransport) WriteError(err error) {
  /* Only write debug errors if DEBUG environment variable is set */
  /* This prevents stderr pollution in production */
	if os.Getenv("NEURONDB_DEBUG") == "true" {
		fmt.Fprintf(t.stderr, "DEBUG: %v\n", err)
	}
}

