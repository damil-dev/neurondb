import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8081/api/v1'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add API key to requests
api.interceptors.request.use((config) => {
  const apiKey = localStorage.getItem('neurondesk_api_key') || localStorage.getItem('api_key')
  if (apiKey) {
    config.headers.Authorization = `Bearer ${apiKey}`
  }
  return config
})

export interface Profile {
  id: string
  name: string
  user_id: string
  mcp_config: Record<string, any>
  neurondb_dsn: string
  agent_endpoint?: string
  agent_api_key?: string
  default_collection?: string
  is_default?: boolean
  created_at: string
  updated_at: string
}

export interface ToolDefinition {
  name: string
  description: string
  inputSchema: Record<string, any>
}

export interface ToolResult {
  content: Array<{ type: string; text: string }>
  isError?: boolean
  metadata?: any
}

export interface CollectionInfo {
  name: string
  schema: string
  vector_col?: string
  indexes?: Array<{
    name: string
    type: string
    definition: string
    size?: string
  }>
  row_count?: number
}

export interface SearchRequest {
  collection: string
  schema?: string
  query_vector: number[]
  query_text?: string
  limit?: number
  filter?: Record<string, any>
  distance_type?: 'l2' | 'cosine' | 'inner_product'
}

export interface SearchResult {
  id: any
  score: number
  distance: number
  data: Record<string, any>
}

export interface ModelConfig {
  id: string
  profile_id: string
  model_provider: string // 'openai', 'anthropic', 'google', 'ollama', 'custom'
  model_name: string
  api_key?: string
  base_url?: string
  is_default: boolean
  is_free: boolean
  metadata?: Record<string, any>
  created_at: string
  updated_at: string
}

// Profiles API
export const profilesAPI = {
  list: () => api.get<Profile[]>('/profiles'),
  get: (id: string) => api.get<Profile>(`/profiles/${id}`),
  create: (profile: Partial<Profile>) => api.post<Profile>('/profiles', profile),
  update: (id: string, profile: Partial<Profile>) => api.put<Profile>(`/profiles/${id}`, profile),
  delete: (id: string) => api.delete(`/profiles/${id}`),
}

// MCP API
export const mcpAPI = {
  listConnections: () => api.get('/mcp/connections'),
  listTools: (profileId: string) => api.get<{ tools: ToolDefinition[] }>(`/profiles/${profileId}/mcp/tools`),
  callTool: (profileId: string, name: string, args: Record<string, any>) =>
    api.post<ToolResult>(`/profiles/${profileId}/mcp/tools/call`, { name, arguments: args }),
  testConfig: (config: { command: string; args?: string[]; env?: Record<string, string> }) =>
    api.post('/mcp/test', config),
}

// NeuronDB API
export const neurondbAPI = {
  listCollections: (profileId: string) => api.get<CollectionInfo[]>(`/profiles/${profileId}/neurondb/collections`),
  search: (profileId: string, request: SearchRequest) =>
    api.post<SearchResult[]>(`/profiles/${profileId}/neurondb/search`, request),
  executeSQL: (profileId: string, query: string) =>
    api.post(`/profiles/${profileId}/neurondb/sql`, { query }),
}

// Agent API
export interface Agent {
  id: string
  name: string
  description?: string
  system_prompt?: string
  model_name?: string
  enabled_tools?: string[]
  config?: Record<string, any>
  created_at?: string
}

export interface Model {
  name: string
  display_name: string
  provider: string
  description: string
}

export interface CreateAgentRequest {
  name: string
  description?: string
  system_prompt?: string
  model_name: string
  enabled_tools?: string[]
  config?: Record<string, any>
}

export interface Session {
  id: string
  agent_id: string
  external_user_id?: string
  metadata?: Record<string, any>
  created_at?: string
}

export interface Message {
  id: string
  session_id: string
  role: string
  content: string
  metadata?: Record<string, any>
  created_at?: string
}

export const agentAPI = {
  listAgents: (profileId: string) => api.get<Agent[]>(`/profiles/${profileId}/agent/agents`),
  getAgent: (profileId: string, agentId: string) => api.get<Agent>(`/profiles/${profileId}/agent/agents/${agentId}`),
  createAgent: (profileId: string, request: CreateAgentRequest) =>
    api.post<Agent>(`/profiles/${profileId}/agent/agents`, request),
  listModels: (profileId: string) => api.get<{ models: Model[] }>(`/profiles/${profileId}/agent/models`),
  createSession: (profileId: string, agentId: string, externalUserId?: string) =>
    api.post<Session>(`/profiles/${profileId}/agent/sessions`, { agent_id: agentId, external_user_id: externalUserId }),
  sendMessage: (profileId: string, sessionId: string, content: string, stream?: boolean) =>
    api.post<Message>(`/profiles/${profileId}/agent/sessions/${sessionId}/messages`, {
      role: 'user',
      content,
      stream: stream || false,
    }),
  getMessages: (profileId: string, sessionId: string) =>
    api.get<Message[]>(`/profiles/${profileId}/agent/sessions/${sessionId}/messages`),
  testConfig: (config: { endpoint: string; api_key: string }) =>
    api.post('/agent/test', config),
}

// Model Config API
export const modelConfigAPI = {
  list: (profileId: string, includeApiKey?: boolean) => 
    api.get<ModelConfig[]>(`/profiles/${profileId}/models`, { params: { include_api_key: includeApiKey } }),
  get: (profileId: string, id: string) => 
    api.get<ModelConfig>(`/profiles/${profileId}/models/${id}`),
  create: (profileId: string, config: Partial<ModelConfig>) => 
    api.post<ModelConfig>(`/profiles/${profileId}/models`, config),
  update: (profileId: string, id: string, config: Partial<ModelConfig>) => 
    api.put<ModelConfig>(`/profiles/${profileId}/models/${id}`, config),
  delete: (profileId: string, id: string) => 
    api.delete(`/profiles/${profileId}/models/${id}`),
  getDefault: (profileId: string) => 
    api.get<ModelConfig>(`/profiles/${profileId}/models/default`),
  setDefault: (profileId: string, id: string) => 
    api.post(`/profiles/${profileId}/models/${id}/set-default`),
}

export default api

