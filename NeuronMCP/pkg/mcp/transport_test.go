package mcp

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"testing"
)

func TestStdioTransport_ReadMessage(t *testing.T) {
	// Create a test message
	message := map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "test",
		"params":  map[string]interface{}{},
	}
	messageJSON, _ := json.Marshal(message)
	messageStr := string(messageJSON)

	// Create input with Content-Length header
	input := fmt.Sprintf("Content-Length: %d\r\nContent-Type: application/json\r\n\r\n%s", len(messageJSON), messageStr)

	transport := &StdioTransport{
		stdin:  strings.NewReader(input),
		stdout: &bytes.Buffer{},
		stderr: &bytes.Buffer{},
	}

	req, err := transport.ReadMessage()
	if err != nil {
		t.Fatalf("ReadMessage() error = %v", err)
	}

	if req.Method != "test" {
		t.Errorf("ReadMessage() method = %v, want test", req.Method)
	}
}

func TestStdioTransport_WriteMessage(t *testing.T) {
	var buf bytes.Buffer
	transport := &StdioTransport{
		stdin:  strings.NewReader(""),
		stdout: &buf,
		stderr: &bytes.Buffer{},
	}

	resp := CreateResponse(json.RawMessage("1"), map[string]string{"test": "value"})
	err := transport.WriteMessage(resp)
	if err != nil {
		t.Fatalf("WriteMessage() error = %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "Content-Length:") {
		t.Error("WriteMessage() should include Content-Length header")
	}
	if !strings.Contains(output, "Content-Type: application/json") {
		t.Error("WriteMessage() should include Content-Type header")
	}
}

func TestStdioTransport_WriteNotification(t *testing.T) {
	var buf bytes.Buffer
	transport := &StdioTransport{
		stdin:  strings.NewReader(""),
		stdout: &buf,
		stderr: &bytes.Buffer{},
	}

	err := transport.WriteNotification("test/notification", map[string]string{"test": "value"})
	if err != nil {
		t.Fatalf("WriteNotification() error = %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "Content-Length:") {
		t.Error("WriteNotification() should include Content-Length header")
	}
	if !strings.Contains(output, "method") {
		t.Error("WriteNotification() should include method in JSON")
	}
}

func TestStdioTransport_ReadMessage_EOF(t *testing.T) {
	transport := &StdioTransport{
		stdin:  strings.NewReader(""),
		stdout: &bytes.Buffer{},
		stderr: &bytes.Buffer{},
	}

	_, err := transport.ReadMessage()
	if err != io.EOF {
		t.Errorf("ReadMessage() error = %v, want EOF", err)
	}
}

