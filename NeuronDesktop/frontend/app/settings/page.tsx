'use client'

import { useState, useEffect } from 'react'
import { profilesAPI, mcpAPI, agentAPI, modelConfigAPI, type Profile, type ModelConfig } from '@/lib/api'
import { setAPIKey, getAPIKey } from '@/lib/auth'
import { 
  KeyIcon,
  ServerIcon,
  DatabaseIcon,
  PlusIcon,
  TrashIcon,
  PencilIcon,
  SparklesIcon,
  CpuIcon,
  CheckCircleIcon,
  XCircleIcon
} from '@/components/Icons'

export default function SettingsPage() {
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [apiKey, setApiKey] = useState('')
  const [showApiKey, setShowApiKey] = useState(false)
  const [expandedProfile, setExpandedProfile] = useState<string | null>(null)
  const [editingProfile, setEditingProfile] = useState<Profile | null>(null)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [newProfileName, setNewProfileName] = useState('')
  const [newProfileDSN, setNewProfileDSN] = useState('')
  const [creatingProfile, setCreatingProfile] = useState(false)

  // MCP Config state
  const [mcpCommand, setMcpCommand] = useState('')
  const [mcpArgs, setMcpArgs] = useState<string[]>([''])
  const [mcpEnv, setMcpEnv] = useState<Array<{ key: string; value: string }>>([{ key: '', value: '' }])

  // Agent Config state
  const [agentEndpoint, setAgentEndpoint] = useState('')
  const [agentApiKey, setAgentApiKey] = useState('')
  const [showAgentApiKey, setShowAgentApiKey] = useState(false)

  // Test state
  const [mcpTestPassed, setMcpTestPassed] = useState(false)
  const [mcpTestLoading, setMcpTestLoading] = useState(false)
  const [mcpTestError, setMcpTestError] = useState<string | null>(null)
  const [agentTestPassed, setAgentTestPassed] = useState(false)
  const [agentTestLoading, setAgentTestLoading] = useState(false)
  const [agentTestError, setAgentTestError] = useState<string | null>(null)

  // Model Config state
  const [modelConfigs, setModelConfigs] = useState<ModelConfig[]>([])
  const [selectedProfileForModels, setSelectedProfileForModels] = useState<string>('')
  const [showModelModal, setShowModelModal] = useState(false)
  const [editingModel, setEditingModel] = useState<ModelConfig | null>(null)
  const [newModelProvider, setNewModelProvider] = useState<string>('openai')
  const [newModelName, setNewModelName] = useState<string>('')
  const [newModelApiKey, setNewModelApiKey] = useState<string>('')
  const [newModelBaseUrl, setNewModelBaseUrl] = useState<string>('')
  const [newModelIsDefault, setNewModelIsDefault] = useState(false)
  const [showModelApiKey, setShowModelApiKey] = useState(false)

  // Model providers configuration
  const MODEL_PROVIDERS = {
    openai: {
      name: 'OpenAI',
      models: [
        { name: 'gpt-4', displayName: 'GPT-4' },
        { name: 'gpt-4-turbo', displayName: 'GPT-4 Turbo' },
        { name: 'gpt-4o', displayName: 'GPT-4o' },
        { name: 'gpt-3.5-turbo', displayName: 'GPT-3.5 Turbo' },
      ],
      requiresKey: true,
      requiresBaseUrl: false,
    },
    anthropic: {
      name: 'Anthropic',
      models: [
        { name: 'claude-3-opus-20240229', displayName: 'Claude 3 Opus' },
        { name: 'claude-3-sonnet-20240229', displayName: 'Claude 3 Sonnet' },
        { name: 'claude-3-haiku-20240307', displayName: 'Claude 3 Haiku' },
        { name: 'claude-3-5-sonnet-20241022', displayName: 'Claude 3.5 Sonnet' },
      ],
      requiresKey: true,
      requiresBaseUrl: false,
    },
    google: {
      name: 'Google',
      models: [
        { name: 'gemini-pro', displayName: 'Gemini Pro' },
        { name: 'gemini-pro-vision', displayName: 'Gemini Pro Vision' },
        { name: 'gemini-1.5-pro', displayName: 'Gemini 1.5 Pro' },
      ],
      requiresKey: true,
      requiresBaseUrl: false,
    },
    ollama: {
      name: 'Ollama (Free)',
      models: [
        { name: 'llama2', displayName: 'Llama 2' },
        { name: 'llama2:13b', displayName: 'Llama 2 13B' },
        { name: 'mistral', displayName: 'Mistral' },
        { name: 'codellama', displayName: 'Code Llama' },
      ],
      requiresKey: false,
      requiresBaseUrl: true,
      defaultBaseUrl: 'http://localhost:11434',
    },
    custom: {
      name: 'Custom',
      models: [],
      requiresKey: true,
      requiresBaseUrl: true,
    },
  }

  useEffect(() => {
    loadProfiles()
    const stored = localStorage.getItem('neurondesk_api_key') || localStorage.getItem('api_key')
    if (stored) {
      setApiKey(stored)
    }
  }, [])

  useEffect(() => {
    if (selectedProfileForModels) {
      loadModelConfigs()
    }
  }, [selectedProfileForModels])

  const loadProfiles = async () => {
    try {
      const response = await profilesAPI.list()
      setProfiles(response.data)
      if (response.data.length > 0 && !selectedProfileForModels) {
        const defaultProfile = response.data.find(p => p.is_default)
        setSelectedProfileForModels(defaultProfile ? defaultProfile.id : response.data[0].id)
      }
    } catch (error) {
      console.error('Failed to load profiles:', error)
    }
  }

  const loadModelConfigs = async () => {
    if (!selectedProfileForModels) return
    try {
      const response = await modelConfigAPI.list(selectedProfileForModels, false)
      setModelConfigs(response.data)
    } catch (error) {
      console.error('Failed to load model configs:', error)
    }
  }

  const handleCreateModel = async () => {
    if (!selectedProfileForModels) {
      alert('Please select a profile first')
      return
    }

    const provider = MODEL_PROVIDERS[newModelProvider as keyof typeof MODEL_PROVIDERS]
    if (!provider) {
      alert('Invalid provider')
      return
    }

    if (provider.requiresKey && !newModelApiKey.trim()) {
      alert('API key is required for this provider')
      return
    }

    if (provider.requiresBaseUrl && !newModelBaseUrl.trim()) {
      alert('Base URL is required for this provider')
      return
    }

    if (!newModelName.trim()) {
      alert('Model name is required')
      return
    }

    try {
      await modelConfigAPI.create(selectedProfileForModels, {
        model_provider: newModelProvider,
        model_name: newModelName,
        api_key: newModelApiKey || undefined,
        base_url: newModelBaseUrl || (provider as any).defaultBaseUrl || undefined,
        is_default: newModelIsDefault,
        is_free: newModelProvider === 'ollama',
      })
      await loadModelConfigs()
      setShowModelModal(false)
      setNewModelProvider('openai')
      setNewModelName('')
      setNewModelApiKey('')
      setNewModelBaseUrl('')
      setNewModelIsDefault(false)
      alert('Model configuration created successfully!')
    } catch (error: any) {
      console.error('Failed to create model config:', error)
      alert('Failed to create model configuration: ' + (error.response?.data?.error || error.message))
    }
  }

  const handleDeleteModel = async (id: string) => {
    if (!confirm('Are you sure you want to delete this model configuration?')) {
      return
    }
    if (!selectedProfileForModels) return
    try {
      await modelConfigAPI.delete(selectedProfileForModels, id)
      await loadModelConfigs()
      alert('Model configuration deleted successfully!')
    } catch (error: any) {
      console.error('Failed to delete model config:', error)
      alert('Failed to delete model configuration: ' + (error.response?.data?.error || error.message))
    }
  }

  const handleEditModel = (config: ModelConfig) => {
    setEditingModel(config)
    setNewModelProvider(config.model_provider)
    setNewModelName(config.model_name)
    setNewModelApiKey(config.api_key || '')
    setNewModelBaseUrl(config.base_url || '')
    setNewModelIsDefault(config.is_default)
    setShowModelModal(true)
  }

  const handleUpdateModel = async () => {
    if (!editingModel || !selectedProfileForModels) return

    const provider = MODEL_PROVIDERS[newModelProvider as keyof typeof MODEL_PROVIDERS]
    if (provider?.requiresKey && !newModelApiKey.trim()) {
      alert('API key is required for this provider')
      return
    }

    try {
      await modelConfigAPI.update(selectedProfileForModels, editingModel.id, {
        model_provider: newModelProvider,
        model_name: newModelName,
        api_key: newModelApiKey || undefined,
        base_url: newModelBaseUrl || undefined,
        is_default: newModelIsDefault,
      })
      await loadModelConfigs()
      setShowModelModal(false)
      setEditingModel(null)
      alert('Model configuration updated successfully!')
    } catch (error: any) {
      console.error('Failed to update model config:', error)
      alert('Failed to update model configuration: ' + (error.response?.data?.error || error.message))
    }
  }

  const handleSaveApiKey = () => {
    localStorage.setItem('neurondesk_api_key', apiKey)
    // Also set legacy key for backward compatibility
    localStorage.setItem('api_key', apiKey)
    alert('API key saved!')
  }

  const handleEditProfile = (profile: Profile) => {
    setEditingProfile(profile)
    setExpandedProfile(profile.id)
    
    // Reset test states
    setMcpTestPassed(false)
    setMcpTestError(null)
    setAgentTestPassed(false)
    setAgentTestError(null)
    
    // Load MCP config
    if (profile.mcp_config) {
      setMcpCommand((profile.mcp_config.command as string) || 'neurondb-mcp')
      const args = profile.mcp_config.args as string[] || []
      setMcpArgs(args.length > 0 ? args : [''])
      
      const env = profile.mcp_config.env as Record<string, string> || {}
      const envEntries = Object.entries(env)
      setMcpEnv(envEntries.length > 0 ? envEntries.map(([k, v]) => ({ key: k, value: v })) : [{ key: '', value: '' }])
    } else {
      setMcpCommand('neurondb-mcp')
      setMcpArgs([''])
      setMcpEnv([{ key: '', value: '' }])
    }
    
    // Load Agent config
    setAgentEndpoint(profile.agent_endpoint || '')
    setAgentApiKey(profile.agent_api_key || '')
  }

  const handleTestMCP = async () => {
    setMcpTestLoading(true)
    setMcpTestError(null)
    setMcpTestPassed(false)
    
    try {
      const env: Record<string, string> = {}
      mcpEnv.forEach(({ key, value }) => {
        if (key.trim() && value.trim()) {
          env[key.trim()] = value.trim()
        }
      })
      
      const config = {
        command: mcpCommand || 'neurondb-mcp',
        args: mcpArgs.filter(arg => arg.trim() !== ''),
        env: env
      }
      
      await mcpAPI.testConfig(config)
      setMcpTestPassed(true)
      setMcpTestError(null)
    } catch (error: any) {
      setMcpTestPassed(false)
      const errorMsg = error.response?.data?.message || error.response?.data?.error || error.message || 'Test failed'
      setMcpTestError(errorMsg)
    } finally {
      setMcpTestLoading(false)
    }
  }

  const handleTestAgent = async () => {
    setAgentTestLoading(true)
    setAgentTestError(null)
    setAgentTestPassed(false)
    
    try {
      if (!agentEndpoint) {
        throw new Error('Agent endpoint is required')
      }
      
      await agentAPI.testConfig({
        endpoint: agentEndpoint,
        api_key: agentApiKey
      })
      setAgentTestPassed(true)
      setAgentTestError(null)
    } catch (error: any) {
      setAgentTestPassed(false)
      const errorMsg = error.response?.data?.message || error.response?.data?.error || error.message || 'Test failed'
      setAgentTestError(errorMsg)
    } finally {
      setAgentTestLoading(false)
    }
  }

  const handleSaveMCPConfig = async () => {
    if (!editingProfile) return
    
    try {
      const mcpConfig: Record<string, any> = {
        command: mcpCommand,
        args: mcpArgs.filter(arg => arg.trim() !== ''),
        env: {}
      }
      
      mcpEnv.forEach(({ key, value }) => {
        if (key.trim() && value.trim()) {
          mcpConfig.env[key.trim()] = value.trim()
        }
      })
      
      const updatedProfile = {
        ...editingProfile,
        mcp_config: mcpConfig
      }
      
      await profilesAPI.update(editingProfile.id, updatedProfile)
      alert('MCP configuration saved!')
      await loadProfiles()
    } catch (error) {
      console.error('Failed to save MCP config:', error)
      alert('Failed to save MCP configuration')
    }
  }

  const handleSaveAgentConfig = async () => {
    if (!editingProfile) return
    
    try {
      const updatedProfile = {
        ...editingProfile,
        agent_endpoint: agentEndpoint,
        agent_api_key: agentApiKey
      }
      
      await profilesAPI.update(editingProfile.id, updatedProfile)
      alert('Agent configuration saved!')
      await loadProfiles()
    } catch (error) {
      console.error('Failed to save agent config:', error)
      alert('Failed to save agent configuration')
    }
  }

  const addMcpArg = () => {
    setMcpArgs([...mcpArgs, ''])
    setMcpTestPassed(false)
    setMcpTestError(null)
  }

  const removeMcpArg = (index: number) => {
    setMcpArgs(mcpArgs.filter((_, i) => i !== index))
    setMcpTestPassed(false)
    setMcpTestError(null)
  }

  const updateMcpArg = (index: number, value: string) => {
    const newArgs = [...mcpArgs]
    newArgs[index] = value
    setMcpArgs(newArgs)
  }

  const addMcpEnv = () => {
    setMcpEnv([...mcpEnv, { key: '', value: '' }])
    setMcpTestPassed(false)
    setMcpTestError(null)
  }

  const removeMcpEnv = (index: number) => {
    setMcpEnv(mcpEnv.filter((_, i) => i !== index))
    setMcpTestPassed(false)
    setMcpTestError(null)
  }

  const updateMcpEnv = (index: number, field: 'key' | 'value', value: string) => {
    const newEnv = [...mcpEnv]
    newEnv[index] = { ...newEnv[index], [field]: value }
    setMcpEnv(newEnv)
  }

  const handleCreateProfile = async () => {
    if (!newProfileName.trim()) {
      alert('Profile name is required')
      return
    }
    
    if (!newProfileDSN.trim()) {
      alert('NeuronDB DSN is required')
      return
    }

    setCreatingProfile(true)
    try {
      const newProfile = {
        name: newProfileName.trim(),
        neurondb_dsn: newProfileDSN.trim(),
        mcp_config: {
          command: 'neurondb-mcp',
          args: [],
          env: {}
        },
        agent_endpoint: '',
        agent_api_key: '',
        default_collection: ''
      }

      await profilesAPI.create(newProfile)
      setShowCreateModal(false)
      setNewProfileName('')
      setNewProfileDSN('')
      await loadProfiles()
      alert('Profile created successfully!')
    } catch (error: any) {
      console.error('Failed to create profile:', error)
      const errorMsg = error.response?.data?.message || error.response?.data?.error || error.message || 'Failed to create profile'
      alert(`Failed to create profile: ${errorMsg}`)
    } finally {
      setCreatingProfile(false)
    }
  }

  return (
    <div className="h-full overflow-auto bg-slate-800">
      <div className="max-w-4xl mx-auto p-6">
        <h1 className="text-3xl font-bold text-slate-100 mb-8">Settings</h1>

        {/* API Key Section */}
        <div className="card mb-6">
          <div className="flex items-center gap-3 mb-4">
            <KeyIcon className="w-6 h-6 text-slate-400" />
            <h2 className="text-xl font-semibold text-slate-100">API Key</h2>
          </div>
          <p className="text-sm text-slate-400 mb-4">
            Your API key is stored locally in your browser. It's used to authenticate all requests to the backend.
          </p>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-200 mb-2">
                API Key
              </label>
              <div className="flex gap-2">
                <input
                  type={showApiKey ? 'text' : 'password'}
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  className="input flex-1"
                  placeholder="Enter your API key"
                />
                <button
                  onClick={() => setShowApiKey(!showApiKey)}
                  className="px-4 py-2 border border-slate-700 rounded-lg hover:bg-slate-800"
                >
                  {showApiKey ? 'Hide' : 'Show'}
                </button>
              </div>
            </div>
            <button
              onClick={handleSaveApiKey}
              className="btn btn-primary"
            >
              Save API Key
            </button>
          </div>
        </div>

        {/* Profiles Section */}
        <div className="card mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <ServerIcon className="w-6 h-6 text-slate-400" />
              <h2 className="text-xl font-semibold text-slate-100">Connection Profiles</h2>
            </div>
            <button 
              onClick={() => setShowCreateModal(true)}
              className="btn btn-primary flex items-center gap-2"
            >
              <PlusIcon className="w-4 h-4" />
              New Profile
            </button>
          </div>
          
          <div className="space-y-3">
            {profiles.map((profile) => (
              <div key={profile.id} className="border border-slate-700 rounded-lg p-4 hover:border-slate-700 transition-colors">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <h3 className="font-semibold text-slate-100">{profile.name}</h3>
                      <span className="text-xs text-slate-400">({profile.id.slice(0, 8)}...)</span>
                    </div>
                    <div className="space-y-1 text-sm text-slate-400">
                      <div className="flex items-center gap-2">
                        <DatabaseIcon className="w-4 h-4" />
                        <span className="font-mono text-xs">{profile.neurondb_dsn.split('@')[1] || profile.neurondb_dsn}</span>
                      </div>
                      {profile.agent_endpoint && (
                        <div className="flex items-center gap-2">
                          <ServerIcon className="w-4 h-4" />
                          <span>{profile.agent_endpoint}</span>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button 
                      onClick={() => handleEditProfile(profile)}
                      className="p-2 text-slate-500 hover:text-blue-600"
                    >
                      <PencilIcon className="w-4 h-4" />
                    </button>
                    <button 
                      onClick={async () => {
                        if (confirm(`Are you sure you want to delete profile "${profile.name}"?`)) {
                          try {
                            await profilesAPI.delete(profile.id)
                            await loadProfiles()
                            if (expandedProfile === profile.id) {
                              setExpandedProfile(null)
                              setEditingProfile(null)
                            }
                            alert('Profile deleted successfully')
                          } catch (error: any) {
                            console.error('Failed to delete profile:', error)
                            alert('Failed to delete profile: ' + (error.response?.data?.error || error.message))
                          }
                        }
                      }}
                      className="p-2 text-slate-500 hover:text-red-600"
                    >
                      <TrashIcon className="w-4 h-4" />
                    </button>
                  </div>
                </div>
                
                {/* Expanded Configuration Sections */}
                {expandedProfile === profile.id && editingProfile?.id === profile.id && (
                  <div className="mt-4 pt-4 border-t border-slate-700 space-y-6">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-slate-400">Configuration for {profile.name}</span>
                      <button
                        onClick={() => {
                          setExpandedProfile(null)
                          setEditingProfile(null)
                        }}
                        className="text-sm text-slate-400 hover:text-slate-200"
                      >
                        Close
                      </button>
                    </div>
                    
                    {/* MCP Configuration */}
                    <div>
                      <div className="flex items-center gap-2 mb-3">
                        <SparklesIcon className="w-5 h-5 text-slate-400" />
                        <h3 className="text-lg font-semibold text-slate-100">MCP Configuration</h3>
                      </div>
                      <div className="space-y-4">
                        <div>
                          <label className="block text-sm font-medium text-slate-200 mb-2">
                            Command
                          </label>
                          <input
                            type="text"
                            value={mcpCommand}
                            onChange={(e) => {
                              setMcpCommand(e.target.value)
                              setMcpTestPassed(false)
                              setMcpTestError(null)
                            }}
                            className="input w-full"
                            placeholder="neurondb-mcp"
                          />
                        </div>
                        
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <label className="block text-sm font-medium text-slate-200">
                              Arguments
                            </label>
                            <button
                              onClick={addMcpArg}
                              className="text-xs text-blue-400 hover:text-blue-300"
                            >
                              + Add Argument
                            </button>
                          </div>
                          <div className="space-y-2">
                            {mcpArgs.map((arg, index) => (
                              <div key={index} className="flex gap-2">
                                <input
                                  type="text"
                                  value={arg}
                                  onChange={(e) => {
                                    updateMcpArg(index, e.target.value)
                                    setMcpTestPassed(false)
                                    setMcpTestError(null)
                                  }}
                                  className="input flex-1"
                                  placeholder="--config /path/to/config.json"
                                />
                                {mcpArgs.length > 1 && (
                                  <button
                                    onClick={() => removeMcpArg(index)}
                                    className="px-3 py-2 text-red-400 hover:text-red-300"
                                  >
                                    <TrashIcon className="w-4 h-4" />
                                  </button>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                        
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <label className="block text-sm font-medium text-slate-200">
                              Environment Variables
                            </label>
                            <button
                              onClick={addMcpEnv}
                              className="text-xs text-blue-400 hover:text-blue-300"
                            >
                              + Add Variable
                            </button>
                          </div>
                          <div className="space-y-2">
                            {mcpEnv.map((env, index) => (
                              <div key={index} className="flex gap-2">
                                <input
                                  type="text"
                                  value={env.key}
                                  onChange={(e) => {
                                    updateMcpEnv(index, 'key', e.target.value)
                                    setMcpTestPassed(false)
                                    setMcpTestError(null)
                                  }}
                                  className="input flex-1"
                                  placeholder="Variable name"
                                />
                                <input
                                  type="text"
                                  value={env.value}
                                  onChange={(e) => {
                                    updateMcpEnv(index, 'value', e.target.value)
                                    setMcpTestPassed(false)
                                    setMcpTestError(null)
                                  }}
                                  className="input flex-1"
                                  placeholder="Variable value"
                                />
                                {mcpEnv.length > 1 && (
                                  <button
                                    onClick={() => removeMcpEnv(index)}
                                    className="px-3 py-2 text-red-400 hover:text-red-300"
                                  >
                                    <TrashIcon className="w-4 h-4" />
                                  </button>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                        
                        <div className="flex items-center gap-3">
                          <button
                            onClick={handleTestMCP}
                            disabled={mcpTestLoading}
                            className="btn border border-slate-600 hover:bg-slate-700 disabled:opacity-50"
                          >
                            {mcpTestLoading ? 'Testing...' : 'Test MCP Configuration'}
                          </button>
                          {mcpTestPassed && (
                            <div className="flex items-center gap-2 text-green-400">
                              <CheckCircleIcon className="w-5 h-5" />
                              <span className="text-sm">Test passed</span>
                            </div>
                          )}
                          {mcpTestError && (
                            <div className="flex items-center gap-2 text-red-400">
                              <XCircleIcon className="w-5 h-5" />
                              <span className="text-sm">{mcpTestError}</span>
                            </div>
                          )}
                        </div>
                        <button
                          onClick={handleSaveMCPConfig}
                          disabled={!mcpTestPassed}
                          className="btn btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          Save MCP Configuration
                        </button>
                      </div>
                    </div>
                    
                    {/* Agent Configuration */}
                    <div>
                      <div className="flex items-center gap-2 mb-3">
                        <CpuIcon className="w-5 h-5 text-slate-400" />
                        <h3 className="text-lg font-semibold text-slate-100">Agent Configuration</h3>
                      </div>
                      <div className="space-y-4">
                        <div>
                          <label className="block text-sm font-medium text-slate-200 mb-2">
                            Agent Endpoint
                          </label>
                          <input
                            type="text"
                            value={agentEndpoint}
                            onChange={(e) => {
                              setAgentEndpoint(e.target.value)
                              setAgentTestPassed(false)
                              setAgentTestError(null)
                            }}
                            className="input w-full"
                            placeholder="http://localhost:8080"
                          />
                        </div>
                        
                        <div>
                          <label className="block text-sm font-medium text-slate-200 mb-2">
                            Agent API Key
                          </label>
                          <div className="flex gap-2">
                            <input
                              type={showAgentApiKey ? 'text' : 'password'}
                              value={agentApiKey}
                              onChange={(e) => {
                                setAgentApiKey(e.target.value)
                                setAgentTestPassed(false)
                                setAgentTestError(null)
                              }}
                              className="input flex-1"
                              placeholder="Enter agent API key"
                            />
                            <button
                              onClick={() => setShowAgentApiKey(!showAgentApiKey)}
                              className="px-4 py-2 border border-slate-700 rounded-lg hover:bg-slate-800"
                            >
                              {showAgentApiKey ? 'Hide' : 'Show'}
                            </button>
                          </div>
                        </div>
                        
                        <div className="flex items-center gap-3">
                          <button
                            onClick={handleTestAgent}
                            disabled={agentTestLoading}
                            className="btn border border-slate-600 hover:bg-slate-700 disabled:opacity-50"
                          >
                            {agentTestLoading ? 'Testing...' : 'Test Agent Configuration'}
                          </button>
                          {agentTestPassed && (
                            <div className="flex items-center gap-2 text-green-400">
                              <CheckCircleIcon className="w-5 h-5" />
                              <span className="text-sm">Test passed</span>
                            </div>
                          )}
                          {agentTestError && (
                            <div className="flex items-center gap-2 text-red-400">
                              <XCircleIcon className="w-5 h-5" />
                              <span className="text-sm">{agentTestError}</span>
                            </div>
                          )}
                        </div>
                        <button
                          onClick={handleSaveAgentConfig}
                          disabled={!agentTestPassed}
                          className="btn btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          Save Agent Configuration
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
            
            {profiles.length === 0 && (
              <div className="text-center py-8 text-slate-400">
                <ServerIcon className="w-12 h-12 mx-auto mb-2 text-slate-500" />
                <p>No profiles configured. Create one to get started.</p>
              </div>
            )}
          </div>

        {/* Model Configurations Section */}
        <div className="card mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <SparklesIcon className="w-6 h-6 text-slate-400" />
              <h2 className="text-xl font-semibold text-slate-100">Model Configurations</h2>
            </div>
            <button 
              onClick={() => {
                setEditingModel(null)
                setNewModelProvider('openai')
                setNewModelName('')
                setNewModelApiKey('')
                setNewModelBaseUrl('')
                setNewModelIsDefault(false)
                setShowModelModal(true)
              }}
              className="btn btn-primary flex items-center gap-2"
              disabled={!selectedProfileForModels}
            >
              <PlusIcon className="w-4 h-4" />
              New Model
            </button>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-200 mb-2">
                Select Profile
              </label>
              <select
                value={selectedProfileForModels}
                onChange={(e) => setSelectedProfileForModels(e.target.value)}
                className="input w-full"
              >
                <option value="">Select a profile</option>
                {profiles.map((profile) => (
                  <option key={profile.id} value={profile.id}>
                    {profile.name} {profile.is_default && '(Default)'}
                  </option>
                ))}
              </select>
            </div>

            {selectedProfileForModels && (
              <div className="space-y-3">
                {modelConfigs.length === 0 ? (
                  <div className="text-center py-8 text-slate-400">
                    <SparklesIcon className="w-12 h-12 mx-auto mb-2 text-slate-500" />
                    <p>No model configurations. Create one to get started.</p>
                  </div>
                ) : (
                  modelConfigs.map((config) => (
                    <div key={config.id} className="border border-slate-700 rounded-lg p-4">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="font-semibold text-slate-100">
                              {config.model_provider} - {config.model_name}
                            </span>
                            {config.is_default && (
                              <span className="px-2 py-1 text-xs bg-blue-600 text-white rounded">Default</span>
                            )}
                            {config.is_free && (
                              <span className="px-2 py-1 text-xs bg-green-600 text-white rounded">Free</span>
                            )}
                          </div>
                          <div className="text-sm text-slate-400 space-y-1">
                            {config.api_key && (
                              <div>API Key: {config.api_key.substring(0, 8)}...</div>
                            )}
                            {config.base_url && (
                              <div>Base URL: {config.base_url}</div>
                            )}
                          </div>
                        </div>
                        <div className="flex gap-2">
                          <button
                            onClick={() => handleEditModel(config)}
                            className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-700 rounded"
                            title="Edit"
                          >
                            <PencilIcon className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => handleDeleteModel(config.id)}
                            className="p-2 text-red-400 hover:text-red-300 hover:bg-slate-700 rounded"
                            title="Delete"
                          >
                            <TrashIcon className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        </div>
        </div>
      </div>

      {/* Create Profile Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-slate-900 rounded-xl shadow-2xl max-w-md w-full">
            <div className="p-6 border-b border-slate-800">
              <h2 className="text-2xl font-bold text-slate-100">Create New Profile</h2>
            </div>

            <div className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Profile Name *
                </label>
                <input
                  type="text"
                  value={newProfileName}
                  onChange={(e) => setNewProfileName(e.target.value)}
                  className="input w-full"
                  placeholder="my-profile"
                  autoFocus
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  NeuronDB DSN *
                </label>
                <input
                  type="text"
                  value={newProfileDSN}
                  onChange={(e) => setNewProfileDSN(e.target.value)}
                  className="input w-full"
                  placeholder="postgresql://user:password@localhost:5432/dbname"
                />
                <p className="text-xs text-slate-400 mt-1">
                  PostgreSQL connection string for NeuronDB
                </p>
              </div>
            </div>

            <div className="p-6 border-t border-slate-800 flex justify-end gap-3">
              <button
                onClick={() => {
                  setShowCreateModal(false)
                  setNewProfileName('')
                  setNewProfileDSN('')
                }}
                className="btn btn-secondary"
                disabled={creatingProfile}
              >
                Cancel
              </button>
              <button
                onClick={handleCreateProfile}
                disabled={creatingProfile || !newProfileName.trim() || !newProfileDSN.trim()}
                className="btn btn-primary"
              >
                {creatingProfile ? 'Creating...' : 'Create Profile'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Model Configuration Modal */}
      {showModelModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-slate-900 rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-slate-800">
              <h2 className="text-2xl font-bold text-slate-100">
                {editingModel ? 'Edit Model Configuration' : 'Add Model Configuration'}
              </h2>
            </div>

            <div className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Provider *
                </label>
                <select
                  value={newModelProvider}
                  onChange={(e) => {
                    setNewModelProvider(e.target.value)
                    setNewModelName('')
                    const provider = MODEL_PROVIDERS[e.target.value as keyof typeof MODEL_PROVIDERS]
                    if ((provider as any)?.defaultBaseUrl) {
                      setNewModelBaseUrl((provider as any).defaultBaseUrl)
                    } else {
                      setNewModelBaseUrl('')
                    }
                  }}
                  className="input w-full"
                >
                  {Object.entries(MODEL_PROVIDERS).map(([key, provider]) => (
                    <option key={key} value={key}>
                      {provider.name}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Model *
                </label>
                {newModelProvider === 'custom' ? (
                  <input
                    type="text"
                    value={newModelName}
                    onChange={(e) => setNewModelName(e.target.value)}
                    className="input w-full"
                    placeholder="Enter custom model name"
                  />
                ) : (
                  <select
                    value={newModelName}
                    onChange={(e) => setNewModelName(e.target.value)}
                    className="input w-full"
                  >
                    <option value="">Select a model</option>
                    {MODEL_PROVIDERS[newModelProvider as keyof typeof MODEL_PROVIDERS]?.models.map((model) => (
                      <option key={model.name} value={model.name}>
                        {model.displayName}
                      </option>
                    ))}
                  </select>
                )}
              </div>

              {MODEL_PROVIDERS[newModelProvider as keyof typeof MODEL_PROVIDERS]?.requiresKey && (
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    API Key *
                  </label>
                  <div className="flex gap-2">
                    <input
                      type={showModelApiKey ? 'text' : 'password'}
                      value={newModelApiKey}
                      onChange={(e) => setNewModelApiKey(e.target.value)}
                      className="input flex-1"
                      placeholder="Enter API key"
                    />
                    <button
                      onClick={() => setShowModelApiKey(!showModelApiKey)}
                      className="px-4 py-2 border border-slate-700 rounded-lg hover:bg-slate-800"
                    >
                      {showModelApiKey ? 'Hide' : 'Show'}
                    </button>
                  </div>
                </div>
              )}

              {MODEL_PROVIDERS[newModelProvider as keyof typeof MODEL_PROVIDERS]?.requiresBaseUrl && (
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Base URL *
                  </label>
                  <input
                    type="text"
                    value={newModelBaseUrl}
                    onChange={(e) => setNewModelBaseUrl(e.target.value)}
                    className="input w-full"
                    placeholder="http://localhost:11434"
                  />
                </div>
              )}

              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="isDefaultModel"
                  checked={newModelIsDefault}
                  onChange={(e) => setNewModelIsDefault(e.target.checked)}
                  className="w-4 h-4 rounded border-slate-700 bg-slate-800"
                />
                <label htmlFor="isDefaultModel" className="text-sm text-slate-300">
                  Set as default model for this profile
                </label>
              </div>
            </div>

            <div className="p-6 border-t border-slate-800 flex justify-end gap-3">
              <button
                onClick={() => {
                  setShowModelModal(false)
                  setEditingModel(null)
                  setNewModelProvider('openai')
                  setNewModelName('')
                  setNewModelApiKey('')
                  setNewModelBaseUrl('')
                  setNewModelIsDefault(false)
                }}
                className="btn btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={editingModel ? handleUpdateModel : handleCreateModel}
                className="btn btn-primary"
              >
                {editingModel ? 'Update' : 'Create'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

