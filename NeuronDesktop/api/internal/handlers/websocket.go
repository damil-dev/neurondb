package handlers

import (
	"encoding/json"
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

// Helper functions
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

// MCPWebSocket handles WebSocket connections for MCP chat
func (h *MCPHandlers) MCPWebSocket(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	
	log.Printf("WebSocket upgrade attempt for profile: %s", profileID)
	
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()
	
	log.Printf("WebSocket connection established for profile: %s", profileID)
	
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
			
			log.Printf("WebSocket received message: type=%v", msg["type"])
			
			// Handle different message types (MCP protocol like Claude Desktop)
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
				// Chat messages using MCP sampling/createMessage (like Claude Desktop)
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
				
				// Get model configuration to extract model name
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
				
				// Build messages array for MCP protocol
				messages := []map[string]interface{}{
					{
						"role":    "user",
						"content": content,
					},
				}
				
				// Call MCP sampling/createMessage with model name (not ID)
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
				
				// Parse MCP response and forward to client
				// The response can be in multiple formats:
				// 1. Direct CreateMessageResponse with "content" as string
				// 2. Wrapped in middleware with "content" as array of ContentBlock
				resultMap, ok := result.(map[string]interface{})
				if !ok {
					// If result is not a map, try to convert it
					log.Printf("Unexpected result type: %T, value: %v", result, result)
					conn.WriteJSON(map[string]interface{}{
						"type":    "assistant",
						"content": fmt.Sprintf("%v", result),
					})
					continue
				}
				
				var responseText string
				
				// Try content as array of content blocks (middleware format)
				if contentBlocks, ok := resultMap["content"].([]interface{}); ok {
					for _, block := range contentBlocks {
						if blockMap, ok := block.(map[string]interface{}); ok {
							if text, ok := blockMap["text"].(string); ok {
								log.Printf("Parsing text block: %s", text[:min(100, len(text))])
								// Try to parse text as JSON in case it's a nested JSON string
								var nestedContent map[string]interface{}
								if err := json.Unmarshal([]byte(text), &nestedContent); err == nil {
									// Successfully parsed as JSON, extract the content field
									log.Printf("Successfully parsed nested JSON, keys: %v", keys(nestedContent))
									if contentVal, ok := nestedContent["content"].(string); ok {
										log.Printf("Extracted content: %s", contentVal[:min(50, len(contentVal))])
										responseText += contentVal
									} else {
										log.Printf("No 'content' field in nested JSON, using raw text")
										responseText += text
									}
								} else {
									// Not JSON, use as-is
									log.Printf("Text is not JSON (err: %v), using as-is", err)
									responseText += text
								}
							} else if blockType, _ := blockMap["type"].(string); blockType == "text" {
								// Handle text content directly
								if textContent, ok := blockMap["content"].(string); ok {
									responseText += textContent
								}
							}
						}
					}
				} else if contentStr, ok := resultMap["content"].(string); ok {
					// Content as string (direct format)
					responseText = contentStr
				} else {
					// Try to find content in nested structure or use entire result
					log.Printf("Could not extract content from result: %+v", resultMap)
					// Try to marshal the result as JSON and extract readable text
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
				// Respond to ping with pong (already handled by ping ticker, but acknowledge if received)
				// No response needed for pong
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

