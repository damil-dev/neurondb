package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
)

/* newUpgrader creates a WebSocket upgrader with CORS checking */
func newUpgrader(allowedOrigins []string) websocket.Upgrader {
	return websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			origin := r.Header.Get("Origin")
			if origin == "" {
				return true /* No origin header means same-origin request */
			}

			/* Check against allowed origins */
			for _, allowedOrigin := range allowedOrigins {
				if allowedOrigin == "*" {
					return true /* Allow all origins if configured */
				}
				if allowedOrigin == origin {
					return true
				}
				/* Support wildcard subdomains like *.example.com */
				if strings.HasPrefix(allowedOrigin, "*.") {
					domain := strings.TrimPrefix(allowedOrigin, "*.")
					if strings.HasSuffix(origin, domain) && len(origin) > len(domain) && origin[len(origin)-len(domain)-1] == '.' {
						return true
					}
				}
			}
			return false
		},
	}
}

/* Global upgrader for backwards compatibility (will be replaced with config-based one) */
var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true /* Allow all origins - will be replaced when handlers are updated */
	},
}

/* Helper functions */
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func keys(m map[string]interface{}) []string {
	result := make([]string, 0, len(m))
	for k := range m {
		result = append(result, k)
	}
	return result
}

/* safeWriteJSON safely writes JSON to WebSocket connection with error handling */
func safeWriteJSON(conn *websocket.Conn, data map[string]interface{}) bool {
	if err := conn.WriteJSON(data); err != nil {
		log.Printf("WebSocket write error: %v", err)
		return false
	}
	return true
}

/* MCPWebSocket handles WebSocket connections for MCP chat */
func (h *MCPHandlers) MCPWebSocket(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	log.Printf("WebSocket upgrade attempt for profile: %s", profileID)

	/* Use CORS-aware upgrader */
	wsUpgrader := newUpgrader(h.corsConfig.AllowedOrigins)
	conn, err := wsUpgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	log.Printf("WebSocket connection established for profile: %s", profileID)

	client, err := h.GetMCPManager().GetClient(r.Context(), profileID)
	if err != nil {
		log.Printf("Failed to get MCP client for profile %s: %v", profileID, err)
		conn.WriteJSON(map[string]interface{}{
			"type":  "error",
			"error": fmt.Sprintf("Failed to connect to MCP server: %v. Please check your profile configuration in Settings.", err),
		})
		return
	}

	if !client.IsAlive() {
		conn.WriteJSON(map[string]interface{}{
			"type":  "error",
			"error": "MCP server process is not running. Please check your MCP configuration and ensure the server command is correct.",
		})
		return
	}

	conn.WriteJSON(map[string]interface{}{
		"type":    "connected",
		"message": "Connected to MCP server successfully",
	})

	/* Create context for goroutine cancellation */
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	go func() {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("WebSocket reader panic recovered: %v", r)
				conn.WriteJSON(map[string]interface{}{
					"type":  "error",
					"error": "Internal server error in WebSocket handler",
				})
			}
			cancel() /* Cancel context when goroutine exits */
		}()

		conn.SetReadDeadline(time.Now().Add(60 * time.Second))

		for {
			/* Check context cancellation */
			select {
			case <-ctx.Done():
				return
			default:
			}

			var msg map[string]interface{}
			if err := conn.ReadJSON(&msg); err != nil {
				if netErr, ok := err.(interface{ Timeout() bool }); ok && netErr.Timeout() {
					log.Printf("WebSocket read timeout for profile %s", profileID)
					conn.WriteJSON(map[string]interface{}{
						"type":  "error",
						"error": "Connection timeout. Please refresh the page.",
					})
					break
				}

				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					log.Printf("WebSocket unexpected close for profile %s: %v", profileID, err)
					conn.WriteJSON(map[string]interface{}{
						"type":  "error",
						"error": fmt.Sprintf("WebSocket connection closed unexpectedly: %v", err),
					})
				} else if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
					log.Printf("WebSocket closed normally for profile %s", profileID)
				} else {
					log.Printf("WebSocket read error for profile %s: %v", profileID, err)
					conn.WriteJSON(map[string]interface{}{
						"type":  "error",
						"error": fmt.Sprintf("Failed to read message: %v", err),
					})
				}
				break
			}

			conn.SetReadDeadline(time.Now().Add(60 * time.Second))

			log.Printf("WebSocket received message: type=%v", msg["type"])

			msgType, _ := msg["type"].(string)
			switch msgType {
			case "tool_call":
				name, _ := msg["name"].(string)
				args, _ := msg["arguments"].(map[string]interface{})

				if name == "" {
					conn.WriteJSON(map[string]interface{}{
						"type":  "error",
						"error": "Tool name is required for tool_call",
					})
					continue
				}

				if !client.IsAlive() {
					conn.WriteJSON(map[string]interface{}{
						"type":  "error",
						"error": "MCP server connection lost. Please reconnect.",
					})
					continue
				}

				result, err := client.CallTool(name, args)
				if err != nil {
					log.Printf("Tool call error for %s: %v", name, err)
					conn.WriteJSON(map[string]interface{}{
						"type":  "error",
						"error": fmt.Sprintf("Tool call failed: %v", err),
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
			case "list_resources":
				resources, err := client.ListResources()
				if err != nil {
					conn.WriteJSON(map[string]interface{}{
						"type":  "error",
						"error": err.Error(),
					})
				} else {
					conn.WriteJSON(map[string]interface{}{
						"type":      "resources",
						"resources": resources,
					})
				}
			case "read_resource":
				uri, _ := msg["uri"].(string)
				if uri == "" {
					conn.WriteJSON(map[string]interface{}{
						"type":  "error",
						"error": "uri is required for read_resource",
					})
				} else {
					resource, err := client.ReadResource(uri)
					if err != nil {
						conn.WriteJSON(map[string]interface{}{
							"type":  "error",
							"error": err.Error(),
						})
					} else {
						conn.WriteJSON(map[string]interface{}{
							"type":     "resource",
							"resource": resource,
						})
					}
				}
			case "list_prompts":
				prompts, err := client.ListPrompts()
				if err != nil {
					conn.WriteJSON(map[string]interface{}{
						"type":  "error",
						"error": err.Error(),
					})
				} else {
					conn.WriteJSON(map[string]interface{}{
						"type":    "prompts",
						"prompts": prompts,
					})
				}
			case "get_prompt":
				name, _ := msg["name"].(string)
				args, _ := msg["arguments"].(map[string]interface{})
				if name == "" {
					conn.WriteJSON(map[string]interface{}{
						"type":  "error",
						"error": "name is required for get_prompt",
					})
				} else {
					prompt, err := client.GetPrompt(name, args)
					if err != nil {
						conn.WriteJSON(map[string]interface{}{
							"type":  "error",
							"error": err.Error(),
						})
					} else {
						conn.WriteJSON(map[string]interface{}{
							"type":   "prompt",
							"prompt": prompt,
						})
					}
				}
			case "chat":
				content, _ := msg["content"].(string)
				modelID, _ := msg["model_id"].(string)

				log.Printf("Chat request: content=%q, modelID=%s", content, modelID)

				if content == "" {
					log.Printf("Error: Empty content")
					conn.WriteJSON(map[string]interface{}{
						"type":  "error",
						"error": "Message content is required",
					})
					continue
				}

				if modelID == "" {
					log.Printf("Error: Empty modelID")
					conn.WriteJSON(map[string]interface{}{
						"type":  "error",
						"error": "Model ID is required",
					})
					continue
				}

				log.Printf("Getting model config for ID: %s", modelID)
				modelConfig, err := h.mcpManager.queries.GetModelConfig(r.Context(), modelID, false)
				if err != nil {
					log.Printf("Failed to get model config: %v", err)
					conn.WriteJSON(map[string]interface{}{
						"type":  "error",
						"error": fmt.Sprintf("Model not found: %v", err),
					})
					continue
				}

				log.Printf("Model config retrieved: name=%s", modelConfig.ModelName)

				messages := []map[string]interface{}{
					{
						"role":    "user",
						"content": content,
					},
				}

				log.Printf("Calling MCP CreateMessage with model=%s", modelConfig.ModelName)
				result, err := client.CreateMessage(messages, modelConfig.ModelName, 0.7)
				if err != nil {
					log.Printf("MCP CreateMessage error: %v", err)
					conn.WriteJSON(map[string]interface{}{
						"type":  "error",
						"error": fmt.Sprintf("Failed to create message via MCP: %v", err),
					})
					continue
				}

				log.Printf("MCP CreateMessage result type: %T, value: %+v", result, result)

				/* Parse MCP response and forward to client
				 * The response can be in multiple formats:
				 * 1. Direct CreateMessageResponse with "content" as string
				 * 2. Wrapped in middleware with "content" as array of ContentBlock */
				resultMap, ok := result.(map[string]interface{})
				if !ok {
					log.Printf("Unexpected result type: %T, value: %v", result, result)
					conn.WriteJSON(map[string]interface{}{
						"type":    "assistant",
						"content": fmt.Sprintf("%v", result),
					})
					continue
				}

				var responseText string

				if contentBlocks, ok := resultMap["content"].([]interface{}); ok {
					for _, block := range contentBlocks {
						if blockMap, ok := block.(map[string]interface{}); ok {
								if text, ok := blockMap["text"].(string); ok {
									log.Printf("Parsing text block: %s", text[:min(100, len(text))])
									var nestedContent map[string]interface{}
								if err := json.Unmarshal([]byte(text), &nestedContent); err == nil {
									log.Printf("Successfully parsed nested JSON, keys: %v", keys(nestedContent))
									if contentVal, ok := nestedContent["content"].(string); ok {
										log.Printf("Extracted content: %s", contentVal[:min(50, len(contentVal))])
										responseText += contentVal
									} else {
										log.Printf("No 'content' field in nested JSON, using raw text")
										responseText += text
									}
								} else {
									log.Printf("Text is not JSON (err: %v), using as-is", err)
									responseText += text
								}
							} else if blockType, _ := blockMap["type"].(string); blockType == "text" {
								if textContent, ok := blockMap["content"].(string); ok {
									responseText += textContent
								}
							}
						}
					}
				} else if contentStr, ok := resultMap["content"].(string); ok {
					responseText = contentStr
				} else {
					log.Printf("Could not extract content from result: %+v", resultMap)
					if resultJSON, err := json.Marshal(resultMap); err == nil {
						responseText = string(resultJSON)
					} else {
						responseText = fmt.Sprintf("%v", result)
					}
				}

				if responseText == "" {
					log.Printf("Empty response text from result: %+v", resultMap)
					conn.WriteJSON(map[string]interface{}{
						"type":  "error",
						"error": "Received empty response from MCP server",
					})
					continue
				}

				log.Printf("Sending assistant response: %s", responseText[:min(100, len(responseText))])
				conn.WriteJSON(map[string]interface{}{
					"type":    "assistant",
					"content": responseText,
				})
			case "pong":
			default:
				conn.WriteJSON(map[string]interface{}{
					"type":  "error",
					"error": "Unknown message type: " + msgType,
				})
			}
		}
	}()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	conn.SetWriteDeadline(time.Now().Add(10 * time.Second))

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if !client.IsAlive() {
				log.Printf("MCP client not alive, closing WebSocket for profile %s", profileID)
				conn.WriteJSON(map[string]interface{}{
					"type":  "error",
					"error": "MCP server connection lost",
				})
				return
			}

			conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := conn.WriteJSON(map[string]interface{}{
				"type": "ping",
			}); err != nil {
				log.Printf("Failed to send ping for profile %s: %v", profileID, err)
				return
			}
		}
	}
}

/* AgentWebSocket handles WebSocket connections for NeuronAgent */
func (h *AgentHandlers) AgentWebSocket(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	sessionID := r.URL.Query().Get("session_id")

	if sessionID == "" {
		http.Error(w, "session_id required", http.StatusBadRequest)
		return
	}

	/* Use CORS-aware upgrader */
	wsUpgrader := newUpgrader(h.corsConfig.AllowedOrigins)
	conn, err := wsUpgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	profile, err := h.GetQueries().GetProfile(r.Context(), profileID)
	if err != nil {
		conn.WriteJSON(map[string]interface{}{
			"type":  "error",
			"error": "Failed to get profile: " + err.Error(),
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

	/* Create context for goroutine cancellation */
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	done := make(chan struct{})

	go func() {
		defer close(done)
		defer cancel()
		for {
			select {
			case <-ctx.Done():
				return
			default:
			}

			var msg map[string]interface{}
			if err := conn.ReadJSON(&msg); err != nil {
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					log.Printf("Agent WebSocket client read error: %v", err)
				}
				break
			}
			if err := agentConn.WriteJSON(msg); err != nil {
				log.Printf("Agent WebSocket write error: %v", err)
				break
			}
		}
	}()

	for {
		select {
		case <-ctx.Done():
			return
		case <-done:
			return
		default:
		}

		var msg map[string]interface{}
		if err := agentConn.ReadJSON(&msg); err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("Agent WebSocket agent read error: %v", err)
			}
			break
		}
		if err := conn.WriteJSON(msg); err != nil {
			log.Printf("Agent WebSocket client write error: %v", err)
			break
		}
	}

	<-done
}
