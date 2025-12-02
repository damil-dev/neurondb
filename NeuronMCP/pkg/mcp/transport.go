package mcp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
)

// StdioTransport handles MCP communication over stdio
type StdioTransport struct {
	stdin  *bufio.Reader
	stdout io.Writer
	stderr io.Writer
}

// NewStdioTransport creates a new stdio transport
func NewStdioTransport() *StdioTransport {
	return &StdioTransport{
		stdin:  bufio.NewReader(os.Stdin),
		stdout: os.Stdout,
		stderr: os.Stderr,
	}
}

// ReadMessage reads a JSON-RPC message from stdin
func (t *StdioTransport) ReadMessage() (*JSONRPCRequest, error) {
	// Read headers
	var contentLength int
	
	for {
		line, err := t.stdin.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				return nil, io.EOF
			}
			return nil, fmt.Errorf("failed to read header: %w", err)
		}

		// Remove trailing newline/carriage return
		line = strings.TrimRight(line, "\r\n")
		
		// Empty line indicates end of headers
		if line == "" {
			break
		}

		// Parse Content-Length
		if strings.HasPrefix(strings.ToLower(line), "content-length:") {
			if _, err := fmt.Sscanf(line, "Content-Length: %d", &contentLength); err != nil {
				if _, err := fmt.Sscanf(line, "content-length: %d", &contentLength); err != nil {
					return nil, fmt.Errorf("invalid Content-Length header: %s", line)
				}
			}
		}
		
		// Parse Content-Type (optional, but we don't need to store it)
		// Just skip Content-Type header if present
	}

	if contentLength <= 0 {
		return nil, fmt.Errorf("missing or invalid Content-Length header")
	}

	// Read message body
	body := make([]byte, contentLength)
	if _, err := io.ReadFull(t.stdin, body); err != nil {
		if err == io.EOF {
			return nil, io.EOF
		}
		return nil, fmt.Errorf("failed to read message body: %w", err)
	}

	return ParseRequest(body)
}

// WriteMessage writes a JSON-RPC message to stdout
func (t *StdioTransport) WriteMessage(resp *JSONRPCResponse) error {
	data, err := SerializeResponse(resp)
	if err != nil {
		return fmt.Errorf("failed to serialize response: %w", err)
	}

	// Write headers
	header := fmt.Sprintf("Content-Length: %d\r\nContent-Type: application/json\r\n\r\n", len(data))
	if _, err := t.stdout.Write([]byte(header)); err != nil {
		return fmt.Errorf("failed to write header: %w", err)
	}

	// Write message body
	if _, err := t.stdout.Write(data); err != nil {
		return fmt.Errorf("failed to write body: %w", err)
	}

	return nil
}

// WriteNotification writes a JSON-RPC notification (no response expected)
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

	// Write headers
	header := fmt.Sprintf("Content-Length: %d\r\nContent-Type: application/json\r\n\r\n", len(data))
	if _, err := t.stdout.Write([]byte(header)); err != nil {
		return fmt.Errorf("failed to write header: %w", err)
	}

	// Write message body
	if _, err := t.stdout.Write(data); err != nil {
		return fmt.Errorf("failed to write body: %w", err)
	}

	return nil
}

// WriteError writes an error to stderr
func (t *StdioTransport) WriteError(err error) {
	fmt.Fprintf(t.stderr, "Error: %v\n", err)
}

