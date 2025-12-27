'use client'

import { useState, useEffect, useRef } from 'react'
import { mcpAPI, profilesAPI, modelConfigAPI, type Profile, type ToolDefinition, type ToolResult, type ModelConfig } from '@/lib/api'
import ProfileSelector from '@/components/ProfileSelector'
import StatusBadge from '@/components/StatusBadge'
import JSONViewer from '@/components/JSONViewer'
import { 
  PaperAirplaneIcon, 
  WrenchScrewdriverIcon,
  CheckCircleIcon,
  XCircleIcon,
  ChatBubbleLeftRightIcon,
  SparklesIcon,
  PlusIcon
} from '@/components/Icons'

interface Message {
  id: string
  role: 'user' | 'assistant' | 'system' | 'tool'
  content: string
  toolName?: string
  timestamp: Date
  data?: any
}

export default function MCPPage() {
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [selectedProfile, setSelectedProfile] = useState<string>('')
  const [tools, setTools] = useState<ToolDefinition[]>([])
  const [modelConfigs, setModelConfigs] = useState<ModelConfig[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [modelError, setModelError] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const [connected, setConnected] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const [showTools, setShowTools] = useState(false)
  const [profileError, setProfileError] = useState<string | null>(null)
  const [loadingProfiles, setLoadingProfiles] = useState(true)

  useEffect(() => {
    loadProfiles()
  }, [])

  useEffect(() => {
    if (selectedProfile) {
      loadTools()
      loadModelConfigs()
      connectWebSocket()
    }
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [selectedProfile])

  useEffect(() => {
    if (selectedModel) {
      validateModel()
    } else {
      setModelError(null)
    }
  }, [selectedModel])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const loadProfiles = async () => {
    setLoadingProfiles(true)
    setProfileError(null)
    try {
      // Check if API key exists
      const apiKey = localStorage.getItem('neurondesk_api_key') || localStorage.getItem('api_key')
      if (!apiKey) {
        const errorMsg = 'API key not set. Please go to Settings and add your API key.'
        setProfileError(errorMsg)
        addMessage('system', errorMsg)
        setLoadingProfiles(false)
        return
      }
      
      const response = await profilesAPI.list()
      console.log('Profiles loaded:', response.data)
      setProfiles(response.data)
      if (response.data.length > 0 && !selectedProfile) {
        // Prefer default profile if available
        const defaultProfile = response.data.find((p: Profile) => p.is_default)
        setSelectedProfile(defaultProfile ? defaultProfile.id : response.data[0].id)
      } else if (response.data.length === 0) {
        setProfileError('No profiles found. Please create a profile in Settings.')
      }
    } catch (error: any) {
      console.error('Failed to load profiles:', error)
      const errorMsg = error.response?.status === 401 
        ? 'Authentication failed. Please check your API key in Settings. Current key: ' + (localStorage.getItem('neurondesk_api_key')?.substring(0, 8) || 'not set') + '...'
        : error.response?.data?.error || error.message || 'Failed to load profiles'
      setProfileError(errorMsg)
      addMessage('system', `Error loading profiles: ${errorMsg}`)
    } finally {
      setLoadingProfiles(false)
    }
  }

  const loadTools = async () => {
    if (!selectedProfile) return
    try {
      const response = await mcpAPI.listTools(selectedProfile)
      setTools(response.data.tools || [])
    } catch (error: any) {
      console.error('Failed to load tools:', error)
      addMessage('system', `Failed to load tools: ${error.response?.data?.error || error.response?.data?.message || error.message || 'Failed to load tools'}. Make sure MCP server is configured correctly in Settings.`)
    }
  }

  const loadModelConfigs = async () => {
    if (!selectedProfile) return
    try {
      const response = await modelConfigAPI.list(selectedProfile, false)
      setModelConfigs(response.data)
      // Set default model if available
      const defaultModel = response.data.find(m => m.is_default)
      if (defaultModel) {
        setSelectedModel(defaultModel.id)
      } else if (response.data.length > 0) {
        setSelectedModel(response.data[0].id)
      }
    } catch (error) {
      console.error('Failed to load model configs:', error)
    }
  }

  const validateModel = async () => {
    if (!selectedModel || !selectedProfile) {
      setModelError(null)
      return
    }

    const model = modelConfigs.find(m => m.id === selectedModel)
    if (!model) {
      setModelError('Model not found')
      return
    }

    // Check if API key is required and set
    const provider = model.model_provider
    if (provider !== 'ollama' && !model.is_free) {
      try {
        const response = await modelConfigAPI.list(selectedProfile, true)
        const fullModel = response.data.find(m => m.id === selectedModel)
        if (!fullModel || !fullModel.api_key || fullModel.api_key.trim() === '') {
          setModelError(`API key not set for ${model.model_name}. Please configure it in Model Settings.`)
          return
        }
      } catch (error) {
        setModelError('Failed to validate model configuration')
        return
      }
    }

    setModelError(null)
  }

  const connectWebSocket = () => {
    if (!selectedProfile) return
    
    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    
    // Get API key from localStorage
    const apiKey = typeof window !== 'undefined' 
      ? (localStorage.getItem('neurondesk_api_key') || localStorage.getItem('api_key'))
      : null
    
    const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8081/api/v1'
    const wsBaseUrl = apiBaseUrl.replace(/^http/, 'ws')
    const wsUrl = apiKey
      ? `${wsBaseUrl}/profiles/${selectedProfile}/mcp/ws?api_key=${encodeURIComponent(apiKey)}`
      : `${wsBaseUrl}/profiles/${selectedProfile}/mcp/ws`
    
    const ws = new WebSocket(wsUrl)
    
    ws.onopen = () => {
      setConnected(true)
      addMessage('system', 'Connected to MCP server')
      ws.send(JSON.stringify({ type: 'list_tools' }))
    }
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.type === 'connected') {
          addMessage('system', data.message)
        } else if (data.type === 'tool_result') {
          addMessage('tool', JSON.stringify(data.result, null, 2), undefined, data.result)
        } else if (data.type === 'tools') {
          setTools(data.tools || [])
        } else if (data.type === 'error') {
          addMessage('system', `Error: ${data.error || 'Unknown error'}`)
          setConnected(false)
        } else if (data.type === 'assistant') {
          addMessage('assistant', data.content || data.message || '')
        } else if (data.type === 'ping') {
          ws.send(JSON.stringify({ type: 'pong' }))
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error)
        addMessage('system', 'Failed to parse message')
      }
    }
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      addMessage('system', 'WebSocket connection error. Check if MCP server is running and profile is configured correctly.')
      setConnected(false)
    }
    
    ws.onclose = (event) => {
      setConnected(false)
      if (event.code !== 1000) {
        addMessage('system', `Connection closed unexpectedly (code: ${event.code}). Check MCP server configuration.`)
      } else {
        addMessage('system', 'Disconnected from MCP server')
      }
      if (selectedProfile && event.code !== 1000) {
        setTimeout(() => {
          if (selectedProfile) {
            connectWebSocket()
          }
        }, 3000)
      }
    }
    
    wsRef.current = ws
  }

  const addMessage = (role: Message['role'], content: string, toolName?: string, data?: any) => {
    setMessages((prev) => [...prev, { 
      id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
      role, 
      content,
      toolName,
      timestamp: new Date(),
      data
    }])
  }

  const handleSendMessage = async () => {
    if (!input.trim() || loading || !selectedModel || modelError) return
    
    const userMessage = input.trim()
    setInput('')
    addMessage('user', userMessage)
    setLoading(true)
    
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'chat',
          content: userMessage,
          model_id: selectedModel
        }))
      } else {
        // Fallback to HTTP if WebSocket not available
        addMessage('assistant', 'WebSocket not connected. Please check your connection.')
      }
    } catch (error: any) {
      addMessage('system', `Error: ${error.message || 'Failed to send message'}`)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const handleToolCall = async (tool: ToolDefinition, args: Record<string, any>) => {
    if (!selectedProfile || !selectedModel || modelError) return
    
    setLoading(true)
    addMessage('user', `Calling tool: ${tool.name}`)
    
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'tool_call',
          name: tool.name,
          arguments: args,
          model_id: selectedModel
        }))
      } else {
        const result = await mcpAPI.callTool(selectedProfile, tool.name, args)
        addMessage('tool', JSON.stringify(result.data, null, 2), tool.name, result.data)
      }
    } catch (error: any) {
      addMessage('system', `Error: ${error.response?.data?.error || error.message || 'Tool call failed'}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="h-full flex flex-col bg-[#1a1a1a]">
      {/* Top Bar - Model Selection and Profile */}
      <div className="bg-[#252525] border-b border-[#333333] px-6 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4 flex-1">
            <div className="flex items-center gap-2">
              <SparklesIcon className="w-5 h-5 text-[#999999]" />
              <label className="text-sm font-medium text-[#c8c8c8]">Model:</label>
            </div>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="input bg-[#1a1a1a] border-[#333333] text-[#e0e0e0] max-w-xs"
            >
              <option value="">Select a model</option>
              {modelConfigs.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.model_provider} - {model.model_name} {model.is_default && '(Default)'}
                </option>
              ))}
            </select>
            {modelError && (
              <div className="flex items-center gap-2 text-red-400 text-sm">
                <XCircleIcon className="w-4 h-4" />
                <span>{modelError}</span>
              </div>
            )}
            {selectedModel && !modelError && (
              <div className="flex items-center gap-2 text-green-400 text-sm">
                <CheckCircleIcon className="w-4 h-4" />
                <span>Ready</span>
              </div>
            )}
          </div>
          <div className="flex items-center gap-4">
            <div className="w-64">
              {profileError ? (
                <div className="px-4 py-2 bg-red-900/20 border border-red-700 rounded-lg">
                  <div className="text-xs text-red-400 font-medium">Profile Error</div>
                  <div className="text-xs text-red-300 mt-1">{profileError}</div>
                  <a 
                    href="/settings" 
                    className="text-xs text-blue-400 hover:text-blue-300 underline mt-1 inline-block"
                  >
                    Go to Settings →
                  </a>
                </div>
              ) : loadingProfiles ? (
                <div className="px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-400">
                  Loading profiles...
                </div>
              ) : (
                <ProfileSelector
                  profiles={profiles}
                  selectedProfile={selectedProfile}
                  onSelect={setSelectedProfile}
                />
              )}
            </div>
            <StatusBadge status={connected ? 'connected' : 'disconnected'} />
            <button
              onClick={() => setShowTools(!showTools)}
              className="p-2 text-[#999999] hover:text-[#e0e0e0] hover:bg-[#2d2d2d] rounded-lg transition-colors"
              title="Toggle Tools"
            >
              <WrenchScrewdriverIcon className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Tools Sidebar - Collapsible */}
        {showTools && (
          <div className="w-80 bg-[#252525] border-r border-[#333333] flex flex-col">
            <div className="p-4 border-b border-[#333333]">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-[#e0e0e0]">Available Tools</h2>
                <button
                  onClick={() => setShowTools(false)}
                  className="text-[#999999] hover:text-[#e0e0e0]"
                >
                  <XCircleIcon className="w-5 h-5" />
                </button>
              </div>
              <p className="text-sm text-[#999999] mt-1">{tools.length} tools available</p>
            </div>
            <div className="flex-1 overflow-y-auto p-2">
              {tools.map((tool) => (
                <ToolCard
                  key={tool.name}
                  tool={tool}
                  onCall={handleToolCall}
                  disabled={!selectedModel || !!modelError || loading}
                />
              ))}
            </div>
          </div>
        )}

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.length === 0 && (
              <div className="text-center py-12">
                <ChatBubbleLeftRightIcon className="w-16 h-16 text-[#999999] mx-auto mb-4" />
                <h2 className="text-xl font-semibold text-[#e0e0e0] mb-2">Start a conversation</h2>
                <p className="text-[#999999] mb-6">Select a model and start chatting with MCP tools</p>
                {!showTools && (
                  <button
                    onClick={() => setShowTools(true)}
                    className="btn btn-primary flex items-center gap-2 mx-auto"
                  >
                    <WrenchScrewdriverIcon className="w-4 h-4" />
                    View Available Tools
                  </button>
                )}
              </div>
            )}
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            {loading && messages.length > 0 && (
              <div className="flex items-center gap-2 text-[#999999]">
                <div className="w-2 h-2 bg-[#999999] rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-[#999999] rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                <div className="w-2 h-2 bg-[#999999] rounded-full animate-bounce" style={{ animationDelay: '0.4s' }} />
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="border-t border-[#333333] bg-[#252525] p-4">
            <div className="max-w-4xl mx-auto">
              <div className="flex items-end gap-3">
                <div className="flex-1 relative">
                  <textarea
                    ref={inputRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder={selectedModel ? "Type your message..." : "Select a model first..."}
                    disabled={!selectedModel || !!modelError || loading}
                    rows={1}
                    className="input w-full resize-none min-h-[44px] max-h-32 bg-[#1a1a1a] border-[#333333] text-[#e0e0e0] placeholder-[#666666] disabled:opacity-50 disabled:cursor-not-allowed"
                    style={{
                      height: 'auto',
                      overflow: 'auto'
                    }}
                    onInput={(e) => {
                      const target = e.target as HTMLTextAreaElement
                      target.style.height = 'auto'
                      target.style.height = `${Math.min(target.scrollHeight, 128)}px`
                    }}
                  />
                </div>
                <button
                  onClick={handleSendMessage}
                  disabled={!input.trim() || !selectedModel || !!modelError || loading}
                  className="btn btn-primary p-3 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <PaperAirplaneIcon className="w-5 h-5" />
                </button>
              </div>
              {!selectedModel && (
                <p className="text-xs text-red-400 mt-2">Please select a model to start chatting</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// Tool Card Component
function ToolCard({ tool, onCall, disabled }: { tool: ToolDefinition, onCall: (tool: ToolDefinition, args: Record<string, any>) => void, disabled: boolean }) {
  const [expanded, setExpanded] = useState(false)
  const [args, setArgs] = useState<Record<string, any>>({})

  const handleCall = () => {
    onCall(tool, args)
    setArgs({})
    setExpanded(false)
  }

  return (
    <div className="mb-2 border border-[#333333] rounded-lg bg-[#1a1a1a]">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left p-3 hover:bg-[#2d2d2d] transition-colors"
      >
        <div className="font-medium text-sm text-[#e0e0e0]">{tool.name}</div>
        <div className="text-xs text-[#999999] mt-1 line-clamp-2">{tool.description}</div>
      </button>
      {expanded && (
        <div className="p-3 border-t border-[#333333] space-y-3">
          {Object.entries(tool.inputSchema?.properties || {}).map(([key, schema]: [string, any]) => (
            <div key={key}>
              <label className="block text-xs font-medium text-[#c8c8c8] mb-1">
                {key} {schema.required && <span className="text-red-500">*</span>}
              </label>
              <input
                type="text"
                value={args[key] || ''}
                onChange={(e) => {
                  let value: any = e.target.value
                  if (value.startsWith('{') || value.startsWith('[')) {
                    try {
                      value = JSON.parse(value)
                    } catch {}
                  }
                  setArgs({ ...args, [key]: value })
                }}
                className="input text-sm bg-[#252525] border-[#333333] text-[#e0e0e0]"
                placeholder={schema.description || schema.type || key}
              />
            </div>
          ))}
          <button
            onClick={handleCall}
            disabled={disabled}
            className="btn btn-primary w-full text-sm disabled:opacity-50"
          >
            Call Tool
          </button>
        </div>
      )}
    </div>
  )
}

// Message Bubble Component
function MessageBubble({ message }: { message: Message }) {
  const getRoleColor = () => {
    switch (message.role) {
      case 'user':
        return 'bg-[#8b5cf6]/20 border-[#8b5cf6]/50'
      case 'assistant':
        return 'bg-[#2d2d2d] border-[#333333]'
      case 'tool':
        return 'bg-green-500/10 border-green-500/50'
      case 'system':
        return 'bg-blue-500/10 border-blue-500/50'
      default:
        return 'bg-[#2d2d2d] border-[#333333]'
    }
  }

  const getRoleIcon = () => {
    switch (message.role) {
      case 'user':
        return null
      case 'assistant':
        return <SparklesIcon className="w-5 h-5 text-[#8b5cf6]" />
      case 'tool':
        return <WrenchScrewdriverIcon className="w-5 h-5 text-green-500" />
      case 'system':
        return <CheckCircleIcon className="w-5 h-5 text-blue-500" />
      default:
        return null
    }
  }

  return (
    <div className={`card border-2 ${getRoleColor()} animate-fade-in`}>
      <div className="flex items-start gap-3">
        {getRoleIcon() && <div className="mt-1">{getRoleIcon()}</div>}
        <div className="flex-1">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-semibold text-[#e0e0e0] capitalize">
              {message.role === 'user' ? 'You' : message.role === 'assistant' ? 'Assistant' : message.role === 'tool' ? `Tool: ${message.toolName || 'Unknown'}` : 'System'}
            </span>
            <span className="text-xs text-[#999999]">{message.timestamp.toLocaleTimeString()}</span>
          </div>
          {message.data ? (
            <JSONViewer data={message.data} defaultExpanded={message.role !== 'system'} />
          ) : (
            <div className="text-[#e0e0e0] whitespace-pre-wrap">{message.content}</div>
          )}
        </div>
      </div>
    </div>
  )
}
