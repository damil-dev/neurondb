package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"syscall"
	"time"
)

// Client wraps an MCP server process and provides JSON-RPC communication
type Client struct {
	cmd      *exec.Cmd
	stdin    io.WriteCloser
	stdout   *bufio.Reader
	stderr   io.ReadCloser
	mu       sync.Mutex
	requests map[string]chan *JSONRPCResponse
	nextID   int64
	ctx      context.Context
	cancel   context.CancelFunc
}

// MCPConfig defines how to spawn an MCP server
type MCPConfig struct {
	Command string            `json:"command"`
	Args    []string          `json:"args,omitempty"`
	Env     map[string]string `json:"env,omitempty"`
}

// NewClient creates a new MCP client and spawns the server process
func NewClient(config MCPConfig) (*Client, error) {
	ctx, cancel := context.WithCancel(context.Background())
	
	cmd := exec.CommandContext(ctx, config.Command, config.Args...)
	
	// Set environment variables
	for k, v := range config.Env {
		cmd.Env = append(cmd.Env, fmt.Sprintf("%s=%s", k, v))
	}
	
	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create stdin pipe: %w", err)
	}
	
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		stdin.Close()
		return nil, fmt.Errorf("failed to create stdout pipe: %w", err)
	}
	
	stderr, err := cmd.StderrPipe()
	if err != nil {
		cancel()
		stdin.Close()
		return nil, fmt.Errorf("failed to create stderr pipe: %w", err)
	}
	
	client := &Client{
		cmd:      cmd,
		stdin:    stdin,
		stdout:   bufio.NewReader(stdout),
		stderr:   stderr,
		requests: make(map[string]chan *JSONRPCResponse),
		ctx:      ctx,
		cancel:   cancel,
	}
	
	// Start the process
	if err := cmd.Start(); err != nil {
		cancel()
		stdin.Close()
		return nil, fmt.Errorf("failed to start MCP server: %w", err)
	}
	
	// Start response reader
	go client.readResponses()
	
	// Initialize the MCP connection
	if err := client.initialize(); err != nil {
		client.Close()
		return nil, fmt.Errorf("failed to initialize MCP connection: %w", err)
	}
	
	return client, nil
}

// initialize performs the MCP initialize handshake
func (c *Client) initialize() error {
	req := InitializeRequest{
		ProtocolVersion: ProtocolVersion,
		Capabilities: map[string]interface{}{
			"tools":     map[string]interface{}{},
			"resources": map[string]interface{}{},
		},
		ClientInfo: map[string]interface{}{
			"name":    "neurondesk",
			"version": "1.0.0",
		},
	}
	
	resp, err := c.Call("initialize", req)
	if err != nil {
		return fmt.Errorf("initialize failed: %w", err)
	}
	
	// Send initialized notification
	if err := c.Notify("notifications/initialized", nil); err != nil {
		return fmt.Errorf("initialized notification failed: %w", err)
	}
	
	// Verify response
	var initResp InitializeResponse
	respBytes, _ := json.Marshal(resp)
	if err := json.Unmarshal(respBytes, &initResp); err != nil {
		return fmt.Errorf("invalid initialize response: %w", err)
	}
	
	return nil
}

// Call sends a JSON-RPC request and waits for a response
func (c *Client) Call(method string, params interface{}) (interface{}, error) {
	c.mu.Lock()
	id := fmt.Sprintf("%d", c.nextID)
	c.nextID++
	responseChan := make(chan *JSONRPCResponse, 1)
	c.requests[id] = responseChan
	c.mu.Unlock()
	
	defer func() {
		c.mu.Lock()
		delete(c.requests, id)
		c.mu.Unlock()
	}()
	
	req := JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      json.RawMessage(fmt.Sprintf(`"%s"`, id)),
		Method:  method,
	}
	
	if params != nil {
		paramsBytes, err := json.Marshal(params)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal params: %w", err)
		}
		req.Params = paramsBytes
	}
	
	reqBytes, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	
	// Write request (MCP uses newline-delimited JSON)
	if _, err := c.stdin.Write(reqBytes); err != nil {
		return nil, fmt.Errorf("failed to write request: %w", err)
	}
	if _, err := c.stdin.Write([]byte("\n")); err != nil {
		return nil, fmt.Errorf("failed to write newline: %w", err)
	}
	
	// Wait for response with timeout
	select {
	case resp := <-responseChan:
		if resp.Error != nil {
			return nil, fmt.Errorf("MCP error: %s (code: %d)", resp.Error.Message, resp.Error.Code)
		}
		return resp.Result, nil
	case <-time.After(30 * time.Second):
		return nil, fmt.Errorf("request timeout")
	case <-c.ctx.Done():
		return nil, fmt.Errorf("client closed")
	}
}

// Notify sends a JSON-RPC notification (no response expected)
func (c *Client) Notify(method string, params interface{}) error {
	req := map[string]interface{}{
		"jsonrpc": "2.0",
		"method":  method,
	}
	
	if params != nil {
		req["params"] = params
	}
	
	reqBytes, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("failed to marshal notification: %w", err)
	}
	
	if _, err := c.stdin.Write(reqBytes); err != nil {
		return fmt.Errorf("failed to write notification: %w", err)
	}
	if _, err := c.stdin.Write([]byte("\n")); err != nil {
		return fmt.Errorf("failed to write newline: %w", err)
	}
	
	return nil
}

// readResponses reads JSON-RPC responses from stdout
func (c *Client) readResponses() {
	for {
		line, err := c.stdout.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				return
			}
			// Handle error - notify all pending requests
			c.mu.Lock()
			for _, ch := range c.requests {
				select {
				case ch <- nil:
				default:
				}
			}
			c.mu.Unlock()
			return
		}
		
		var resp JSONRPCResponse
		if err := json.Unmarshal(line, &resp); err != nil {
			continue // Skip malformed responses
		}
		
		// Extract ID and route to waiting request
		if len(resp.ID) > 0 {
			id := string(resp.ID)
			// Remove quotes
			if len(id) >= 2 && id[0] == '"' && id[len(id)-1] == '"' {
				id = id[1 : len(id)-1]
			}
			
			c.mu.Lock()
			ch, ok := c.requests[id]
			c.mu.Unlock()
			
			if ok {
				select {
				case ch <- &resp:
				default:
				}
			}
		}
	}
}

// ListTools lists available tools from the MCP server
func (c *Client) ListTools() (*ListToolsResponse, error) {
	resp, err := c.Call("tools/list", nil)
	if err != nil {
		return nil, err
	}
	
	respBytes, _ := json.Marshal(resp)
	var toolsResp ListToolsResponse
	if err := json.Unmarshal(respBytes, &toolsResp); err != nil {
		return nil, fmt.Errorf("failed to parse tools/list response: %w", err)
	}
	
	return &toolsResp, nil
}

// CallTool calls a tool on the MCP server
func (c *Client) CallTool(name string, arguments map[string]interface{}) (*ToolResult, error) {
	req := CallToolRequest{
		Name:      name,
		Arguments: arguments,
	}
	
	resp, err := c.Call("tools/call", req)
	if err != nil {
		return nil, err
	}
	
	respBytes, _ := json.Marshal(resp)
	var toolResp ToolResult
	if err := json.Unmarshal(respBytes, &toolResp); err != nil {
		return nil, fmt.Errorf("failed to parse tools/call response: %w", err)
	}
	
	return &toolResp, nil
}

// ListResources lists available resources
func (c *Client) ListResources() (*ListResourcesResponse, error) {
	resp, err := c.Call("resources/list", nil)
	if err != nil {
		return nil, err
	}
	
	respBytes, _ := json.Marshal(resp)
	var resourcesResp ListResourcesResponse
	if err := json.Unmarshal(respBytes, &resourcesResp); err != nil {
		return nil, fmt.Errorf("failed to parse resources/list response: %w", err)
	}
	
	return &resourcesResp, nil
}

// Close shuts down the MCP client and process
func (c *Client) Close() error {
	c.cancel()
	
	if c.stdin != nil {
		c.stdin.Close()
	}
	
	if c.cmd != nil && c.cmd.Process != nil {
		// Give process a chance to exit gracefully
		c.cmd.Process.Signal(os.Interrupt)
		time.Sleep(100 * time.Millisecond)
		c.cmd.Process.Kill()
		c.cmd.Wait()
	}
	
	return nil
}

// IsAlive checks if the MCP process is still running
func (c *Client) IsAlive() bool {
	if c.cmd == nil || c.cmd.Process == nil {
		return false
	}
	
	// Check if process is still running by sending signal 0
	// Signal 0 doesn't actually send a signal, just checks if process exists
	err := c.cmd.Process.Signal(syscall.Signal(0))
	return err == nil
}

