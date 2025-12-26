/*-------------------------------------------------------------------------
 *
 * main.go
 *    CLI tool for NeuronAgent management
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronAgent/cmd/neuronagent-cli/main.go
 *
 *-------------------------------------------------------------------------
 */

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/google/uuid"
	"github.com/spf13/cobra"
)

var (
	apiURL    string
	apiKey    string
	outputFormat string
)

var rootCmd = &cobra.Command{
	Use:   "neuronagent",
	Short: "NeuronAgent CLI - Manage agents, sessions, and messages",
	Long:  "Command-line interface for NeuronAgent API",
}

var agentsCmd = &cobra.Command{
	Use:   "agents",
	Short: "Manage agents",
}

var agentsListCmd = &cobra.Command{
	Use:   "list",
	Short: "List all agents",
	RunE:  listAgents,
}

var agentsGetCmd = &cobra.Command{
	Use:   "get [id]",
	Short: "Get agent details",
	Args:  cobra.ExactArgs(1),
	RunE:  getAgent,
}

var agentsCreateCmd = &cobra.Command{
	Use:   "create",
	Short: "Create a new agent",
	RunE:  createAgent,
}

var agentsDeleteCmd = &cobra.Command{
	Use:   "delete [id]",
	Short: "Delete an agent",
	Args:  cobra.ExactArgs(1),
	RunE:  deleteAgent,
}

var sessionsCmd = &cobra.Command{
	Use:   "sessions",
	Short: "Manage sessions",
}

var sessionsCreateCmd = &cobra.Command{
	Use:   "create [agent-id]",
	Short: "Create a new session",
	Args:  cobra.ExactArgs(1),
	RunE:  createSession,
}

var sessionsGetCmd = &cobra.Command{
	Use:   "get [id]",
	Short: "Get session details",
	Args:  cobra.ExactArgs(1),
	RunE:  getSession,
}

var messagesCmd = &cobra.Command{
	Use:   "messages",
	Short: "Manage messages",
}

var messagesSendCmd = &cobra.Command{
	Use:   "send [session-id] [message]",
	Short: "Send a message to an agent",
	Args:  cobra.ExactArgs(2),
	RunE:  sendMessage,
}

func init() {
	rootCmd.PersistentFlags().StringVar(&apiURL, "url", "http://localhost:8080", "API URL")
	rootCmd.PersistentFlags().StringVar(&apiKey, "key", "", "API key (required)")
	rootCmd.PersistentFlags().StringVar(&outputFormat, "format", "json", "Output format (json, table)")

	rootCmd.MarkPersistentFlagRequired("key")

	rootCmd.AddCommand(agentsCmd)
	agentsCmd.AddCommand(agentsListCmd)
	agentsCmd.AddCommand(agentsGetCmd)
	agentsCmd.AddCommand(agentsCreateCmd)
	agentsCmd.AddCommand(agentsDeleteCmd)

	rootCmd.AddCommand(sessionsCmd)
	sessionsCmd.AddCommand(sessionsCreateCmd)
	sessionsCmd.AddCommand(sessionsGetCmd)

	rootCmd.AddCommand(messagesCmd)
	messagesCmd.AddCommand(messagesSendCmd)
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

/* API client functions */
func makeRequest(method, endpoint string, body io.Reader) (*http.Response, error) {
	url := apiURL + endpoint
	req, err := http.NewRequest(method, url, body)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Authorization", "Bearer "+apiKey)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	return client.Do(req)
}

func listAgents(cmd *cobra.Command, args []string) error {
	resp, err := makeRequest("GET", "/api/v1/agents", nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	return printJSON(resp.Body)
}

func getAgent(cmd *cobra.Command, args []string) error {
	id := args[0]
	if _, err := uuid.Parse(id); err != nil {
		return fmt.Errorf("invalid agent ID: %s", id)
	}

	resp, err := makeRequest("GET", fmt.Sprintf("/api/v1/agents/%s", id), nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to get agent: status %d", resp.StatusCode)
	}

	return printJSON(resp.Body)
}

func createAgent(cmd *cobra.Command, args []string) error {
	/* For now, create a basic agent */
	// In production, would read from stdin or flags
	return fmt.Errorf("create agent not yet implemented - use API directly")
}

func deleteAgent(cmd *cobra.Command, args []string) error {
	id := args[0]
	if _, err := uuid.Parse(id); err != nil {
		return fmt.Errorf("invalid agent ID: %s", id)
	}

	resp, err := makeRequest("DELETE", fmt.Sprintf("/api/v1/agents/%s", id), nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNoContent {
		return fmt.Errorf("failed to delete agent: status %d", resp.StatusCode)
	}

	fmt.Println("Agent deleted successfully")
	return nil
}

func createSession(cmd *cobra.Command, args []string) error {
	agentID := args[0]
	if _, err := uuid.Parse(agentID); err != nil {
		return fmt.Errorf("invalid agent ID: %s", agentID)
	}

	payload := map[string]interface{}{
		"agent_id": agentID,
	}
	body, _ := json.Marshal(payload)

	resp, err := makeRequest("POST", "/api/v1/sessions", bytes.NewReader(body))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	return printJSON(resp.Body)
}

func getSession(cmd *cobra.Command, args []string) error {
	id := args[0]
	if _, err := uuid.Parse(id); err != nil {
		return fmt.Errorf("invalid session ID: %s", id)
	}

	resp, err := makeRequest("GET", fmt.Sprintf("/api/v1/sessions/%s", id), nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	return printJSON(resp.Body)
}

func sendMessage(cmd *cobra.Command, args []string) error {
	sessionID := args[0]
	message := args[1]

	if _, err := uuid.Parse(sessionID); err != nil {
		return fmt.Errorf("invalid session ID: %s", sessionID)
	}

	payload := map[string]interface{}{
		"content": message,
		"role":    "user",
		"stream":  false,
	}
	body, _ := json.Marshal(payload)

	resp, err := makeRequest("POST", fmt.Sprintf("/api/v1/sessions/%s/messages", sessionID), bytes.NewReader(body))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	return printJSON(resp.Body)
}

func printJSON(r io.Reader) error {
	var data interface{}
	if err := json.NewDecoder(r).Decode(&data); err != nil {
		return err
	}

	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(data)
}

