package handlers

import (
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all origins in development
	},
}

// MCPWebSocket handles WebSocket connections for MCP chat
func (h *MCPHandlers) MCPWebSocket(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()
	
	// Get MCP client with detailed error handling
	client, err := h.GetMCPManager().GetClient(r.Context(), profileID)
	if err != nil {
		log.Printf("Failed to get MCP client for profile %s: %v", profileID, err)
		conn.WriteJSON(map[string]interface{}{
			"type":  "error",
			"error": fmt.Sprintf("Failed to connect to MCP server: %v. Please check your profile configuration in Settings.", err),
		})
		return
	}
	
	// Verify client is alive
	if !client.IsAlive() {
		conn.WriteJSON(map[string]interface{}{
			"type":  "error",
			"error": "MCP server process is not running. Please check your MCP configuration and ensure the server command is correct.",
		})
		return
	}
	
	// Send connection confirmation
	conn.WriteJSON(map[string]interface{}{
		"type":    "connected",
		"message": "Connected to MCP server successfully",
	})
	
	// Read messages from WebSocket and forward to MCP
	go func() {
		for {
			var msg map[string]interface{}
			if err := conn.ReadJSON(&msg); err != nil {
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					log.Printf("WebSocket error: %v", err)
				}
				break
			}
			
			// Handle different message types
			msgType, _ := msg["type"].(string)
			switch msgType {
			case "tool_call":
				name, _ := msg["name"].(string)
				args, _ := msg["arguments"].(map[string]interface{})
				
				result, err := client.CallTool(name, args)
				if err != nil {
					conn.WriteJSON(map[string]interface{}{
						"type":  "error",
						"error": err.Error(),
					})
				} else {
					conn.WriteJSON(map[string]interface{}{
						"type":   "tool_result",
						"result": result,
					})
				}
			case "list_tools":
				tools, err := client.ListTools()
				if err != nil {
					conn.WriteJSON(map[string]interface{}{
						"type":  "error",
						"error": err.Error(),
					})
				} else {
					conn.WriteJSON(map[string]interface{}{
						"type":  "tools",
						"tools": tools,
					})
				}
			default:
				conn.WriteJSON(map[string]interface{}{
					"type":  "error",
					"error": "Unknown message type: " + msgType,
				})
			}
		}
	}()
	
	// Keep connection alive
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			if err := conn.WriteJSON(map[string]interface{}{
				"type": "ping",
			}); err != nil {
				return
			}
		}
	}
}

// AgentWebSocket handles WebSocket connections for NeuronAgent
func (h *AgentHandlers) AgentWebSocket(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	sessionID := r.URL.Query().Get("session_id")
	
	if sessionID == "" {
		http.Error(w, "session_id required", http.StatusBadRequest)
		return
	}
	
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()
	
	// Get profile to construct URL
	profile, err := h.GetQueries().GetProfile(r.Context(), profileID)
	if err != nil {
		conn.WriteJSON(map[string]interface{}{
			"type":  "error",
			"error": "Failed to get profile: " + err.Error(),
		})
		return
	}
	if err != nil {
		conn.WriteJSON(map[string]interface{}{
			"type":  "error",
			"error": "Failed to get profile",
		})
		return
	}
	
	agentWSURL := profile.AgentEndpoint + "/ws?session_id=" + sessionID
	agentConn, _, err := websocket.DefaultDialer.Dial(agentWSURL, http.Header{
		"Authorization": []string{"Bearer " + profile.AgentAPIKey},
	})
	if err != nil {
		conn.WriteJSON(map[string]interface{}{
			"type":  "error",
			"error": "Failed to connect to agent: " + err.Error(),
		})
		return
	}
	defer agentConn.Close()
	
	// Proxy messages both ways
	go func() {
		for {
			var msg map[string]interface{}
			if err := conn.ReadJSON(&msg); err != nil {
				break
			}
			agentConn.WriteJSON(msg)
		}
	}()
	
	for {
		var msg map[string]interface{}
		if err := agentConn.ReadJSON(&msg); err != nil {
			break
		}
		conn.WriteJSON(msg)
	}
}

