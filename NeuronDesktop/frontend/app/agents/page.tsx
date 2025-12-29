'use client'

import { useState, useEffect } from 'react'
import { agentAPI, profilesAPI, type Profile, type Agent, type Model, type CreateAgentRequest } from '@/lib/api'
import StatusBadge from '@/components/StatusBadge'
import JSONViewer from '@/components/JSONViewer'
import {
  PlusIcon,
  TrashIcon,
  PencilIcon,
  ChatBubbleLeftRightIcon,
  CpuChipIcon
} from '@/components/Icons'

export default function AgentsPage() {
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [selectedProfile, setSelectedProfile] = useState<string>('')
  const [agents, setAgents] = useState<Agent[]>([])
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(false)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [editingAgent, setEditingAgent] = useState<Agent | null>(null)
  const [formData, setFormData] = useState<CreateAgentRequest>({
    name: '',
    description: '',
    system_prompt: '',
    model_name: 'gpt-4',
    enabled_tools: [],
    config: {},
  })
  const [customModelName, setCustomModelName] = useState('')
  const [selectedModelType, setSelectedModelType] = useState<'preset' | 'custom'>('preset')

  useEffect(() => {
    loadProfiles()
  }, [])

  useEffect(() => {
    if (selectedProfile) {
      loadAgents()
      loadModels()
    }
  }, [selectedProfile])

  const loadProfiles = async () => {
    try {
      const response = await profilesAPI.list()
      setProfiles(response.data)
      if (response.data.length > 0 && !selectedProfile) {
        const activeProfileId = localStorage.getItem('active_profile_id')
        if (activeProfileId) {
          const active = response.data.find((p: Profile) => p.id === activeProfileId)
          if (active) {
            setSelectedProfile(activeProfileId)
            return
          }
        }
        const defaultProfile = response.data.find((p: Profile) => p.is_default)
        setSelectedProfile(defaultProfile ? defaultProfile.id : response.data[0].id)
      }
    } catch (error) {
      console.error('Failed to load profiles:', error)
    }
  }

  const loadAgents = async () => {
    if (!selectedProfile) return
    setLoading(true)
    try {
      const response = await agentAPI.listAgents(selectedProfile)
      setAgents(response.data)
    } catch (error) {
      console.error('Failed to load agents:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadModels = async () => {
    if (!selectedProfile) return
    try {
      const response = await agentAPI.listModels(selectedProfile)
      setModels(response.data.models)
    } catch (error) {
      console.error('Failed to load models:', error)
    }
  }

  const handleCreateAgent = async () => {
    if (!selectedProfile) {
      console.warn('Cannot create agent: no profile selected')
      return
    }
    
    // Check if agent endpoint is configured
    const currentProfile = profiles.find((p) => p.id === selectedProfile)
    if (!currentProfile?.agent_endpoint) {
      const message = 'Agent endpoint is not configured for this profile.\n\n' +
        'Please configure the Agent Endpoint in Settings:\n' +
        '1. Go to Settings page\n' +
        '2. Select this profile\n' +
        '3. Configure the Agent Endpoint (e.g., http://localhost:8080)\n' +
        '4. Save the configuration\n\n' +
        'Then try creating the agent again.'
      alert(message)
      return
    }
    
    if (!formData.name.trim()) {
      alert('Agent name is required')
      return
    }
    
    if (selectedModelType === 'custom' && !customModelName.trim()) {
      alert('Custom model name is required')
      return
    }
    
    const finalModelName = selectedModelType === 'custom' ? customModelName.trim() : formData.model_name
    if (!finalModelName) {
      alert('Model name is required')
      return
    }
    
    setLoading(true)
    try {
      const request: CreateAgentRequest = {
        ...formData,
        model_name: finalModelName,
      }
      console.log('Creating agent:', { profileId: selectedProfile, request })
      const response = await agentAPI.createAgent(selectedProfile, request)
      console.log('Agent created successfully:', response.data)
      setShowCreateModal(false)
      resetForm()
      loadAgents()
    } catch (error: any) {
      console.error('Failed to create agent:', error)
      console.error('Error details:', {
        status: error.response?.status,
        statusText: error.response?.statusText,
        data: error.response?.data,
        message: error.message,
      })
      let errorMessage = error.response?.data?.message || error.response?.data?.error || error.message || 'Failed to create agent'
      
      // Provide more helpful error message for endpoint configuration issues
      if (errorMessage.toLowerCase().includes('agent endpoint not configured') || 
          errorMessage.toLowerCase().includes('endpoint not configured')) {
        errorMessage = 'Agent endpoint is not configured for this profile.\n\n' +
          'Please configure the Agent Endpoint in Settings:\n' +
          '1. Go to Settings page\n' +
          '2. Select this profile\n' +
          '3. Configure the Agent Endpoint (e.g., http://localhost:8080)\n' +
          '4. Save the configuration\n\n' +
          'Then try creating the agent again.'
      }
      
      alert('Failed to create agent: ' + errorMessage)
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteAgent = async (agentId: string) => {
    if (!confirm('Are you sure you want to delete this agent?')) return
    if (!selectedProfile) return
    
    setLoading(true)
    try {
      // Note: Delete endpoint needs to be added to backend
      // For now, we'll show an error
      alert('Delete agent functionality requires backend endpoint. Please use the NeuronAgent API directly.')
      // await agentAPI.deleteAgent(selectedProfile, agentId)
      // loadAgents()
    } catch (error: any) {
      console.error('Failed to delete agent:', error)
      alert('Failed to delete agent: ' + (error.response?.data?.error || error.message))
    } finally {
      setLoading(false)
    }
  }

  const resetForm = () => {
    setFormData({
      name: '',
      description: '',
      system_prompt: '',
      model_name: 'gpt-4',
      enabled_tools: [],
      config: {},
    })
    setCustomModelName('')
    setSelectedModelType('preset')
    setEditingAgent(null)
  }

  const openEditModal = (agent: Agent) => {
    setEditingAgent(agent)
    setFormData({
      name: agent.name,
      description: agent.description || '',
      system_prompt: agent.system_prompt || '',
      model_name: agent.model_name || 'gpt-4',
      enabled_tools: agent.enabled_tools || [],
      config: agent.config || {},
    })
    setSelectedModelType(agent.model_name && !models.find(m => m.name === agent.model_name) ? 'custom' : 'preset')
    setCustomModelName(agent.model_name && !models.find(m => m.name === agent.model_name) ? agent.model_name : '')
    setShowCreateModal(true)
  }

  const selectedModel = models.find(m => m.name === formData.model_name)

  return (
    <div className="h-full overflow-auto bg-transparent p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-slate-100 mb-2">NeuronAgent Management</h1>
          <p className="text-gray-700 dark:text-slate-400">Create and manage AI agents with model selection</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-sm text-gray-700 dark:text-slate-300">
            Profile: <span className="text-gray-900 dark:text-slate-100 font-medium">{profiles.find((p) => p.id === selectedProfile)?.name || '...'}</span>
            {selectedProfile && !profiles.find((p) => p.id === selectedProfile)?.agent_endpoint && (
              <span className="ml-2 text-xs text-yellow-600 dark:text-yellow-400 bg-yellow-100 dark:bg-yellow-400/10 px-2 py-1 rounded">
                ⚠ Agent endpoint not configured
              </span>
            )}
          </div>
          {selectedProfile && (
            <button
              onClick={() => {
                resetForm()
                setShowCreateModal(true)
              }}
              className="btn btn-primary flex items-center gap-2"
            >
              <PlusIcon className="w-5 h-5" />
              Create Agent
            </button>
          )}
        </div>
      </div>

      {!selectedProfile ? (
        <div className="card text-center py-12">
          <p className="text-gray-700 dark:text-slate-400">Please select a profile to manage agents</p>
        </div>
      ) : selectedProfile && !profiles.find((p) => p.id === selectedProfile)?.agent_endpoint ? (
        <div className="card bg-yellow-50 dark:bg-yellow-400/10 border-yellow-200 dark:border-yellow-400/20 mb-6">
          <div className="flex items-start gap-3">
            <div className="text-yellow-600 dark:text-yellow-400 text-xl">⚠</div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-yellow-700 dark:text-yellow-400 mb-2">Agent Endpoint Not Configured</h3>
              <p className="text-gray-800 dark:text-slate-300 mb-3">
                The selected profile does not have an Agent Endpoint configured. You need to configure it before creating agents.
              </p>
              <p className="text-sm text-gray-700 dark:text-slate-400 mb-4">
                To configure the Agent Endpoint:
              </p>
              <ol className="list-decimal list-inside text-sm text-gray-800 dark:text-slate-300 space-y-1 mb-4">
                <li>Go to the <strong className="text-gray-900 dark:text-slate-200">Settings</strong> page</li>
                <li>Select this profile: <strong className="text-gray-900 dark:text-slate-200">{profiles.find((p) => p.id === selectedProfile)?.name}</strong></li>
                <li>Configure the <strong className="text-gray-900 dark:text-slate-200">Agent Endpoint</strong> (e.g., <code className="text-yellow-700 dark:text-yellow-400 bg-yellow-100 dark:bg-yellow-400/20 px-1 rounded">http://localhost:8080</code>)</li>
                <li>Optionally configure the <strong className="text-gray-900 dark:text-slate-200">Agent API Key</strong> if required</li>
                <li>Click <strong className="text-gray-900 dark:text-slate-200">Save Agent Configuration</strong></li>
              </ol>
              <p className="text-xs text-gray-600 dark:text-slate-500">
                The Agent Endpoint should point to your NeuronAgent server URL. If you're running NeuronAgent locally, it's typically <code className="text-gray-700 dark:text-slate-400 bg-gray-100 dark:bg-slate-800 px-1 rounded">http://localhost:8080</code>.
              </p>
            </div>
          </div>
        </div>
      ) : loading ? (
        <div className="card text-center py-12">
          <p className="text-gray-700 dark:text-slate-400">Loading agents...</p>
        </div>
      ) : agents.length === 0 ? (
        <div className="card text-center py-12">
          <ChatBubbleLeftRightIcon className="w-16 h-16 text-gray-500 dark:text-slate-600 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-900 dark:text-slate-200 mb-2">No Agents Yet</h3>
          <p className="text-gray-700 dark:text-slate-400 mb-6">Create your first agent to get started</p>
          <button
            onClick={() => {
              resetForm()
              setShowCreateModal(true)
            }}
            className="btn btn-primary"
          >
            Create Agent
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {agents.map((agent) => (
            <div key={agent.id} className="card hover:shadow-xl transition-shadow">
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-slate-100 mb-1">{agent.name}</h3>
                  {agent.description && (
                    <p className="text-sm text-gray-600 dark:text-slate-400 mb-2">{agent.description}</p>
                  )}
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => openEditModal(agent)}
                    className="p-2 text-gray-600 dark:text-slate-400 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-slate-100 dark:hover:bg-slate-800 rounded"
                  >
                    <PencilIcon className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => handleDeleteAgent(agent.id)}
                    className="p-2 text-gray-600 dark:text-slate-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-slate-100 dark:hover:bg-slate-800 rounded"
                  >
                    <TrashIcon className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2">
                  <CpuChipIcon className="w-4 h-4 text-blue-400" />
                  <span className="text-gray-700 dark:text-slate-300">Model:</span>
                  <span className="text-gray-900 dark:text-slate-100 font-medium">{agent.model_name || 'Not set'}</span>
                </div>
                {agent.enabled_tools && agent.enabled_tools.length > 0 && (
                  <div>
                    <span className="text-gray-700 dark:text-slate-300">Tools: </span>
                    <span className="text-gray-900 dark:text-slate-100">{agent.enabled_tools.join(', ')}</span>
                  </div>
                )}
                {agent.system_prompt && (
                  <div className="text-xs text-gray-600 dark:text-slate-400 truncate" title={agent.system_prompt}>
                    Prompt: {agent.system_prompt.substring(0, 50)}...
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Create/Edit Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-slate-900 rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-slate-200 dark:border-slate-800">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-slate-100">
                {editingAgent ? 'Edit Agent' : 'Create New Agent'}
              </h2>
            </div>

            <div className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">Name *</label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="input"
                  placeholder="my-agent"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Description</label>
                <input
                  type="text"
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  className="input"
                  placeholder="A helpful AI agent"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">System Prompt</label>
                <textarea
                  value={formData.system_prompt}
                  onChange={(e) => setFormData({ ...formData, system_prompt: e.target.value })}
                  className="input min-h-[100px]"
                  placeholder="You are a helpful assistant..."
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Model *</label>
                <div className="space-y-3">
                  <div className="flex gap-4">
                    <label className="flex items-center gap-2 text-gray-700 dark:text-slate-300">
                      <input
                        type="radio"
                        checked={selectedModelType === 'preset'}
                        onChange={() => setSelectedModelType('preset')}
                        className="w-4 h-4"
                      />
                      Preset Model
                    </label>
                    <label className="flex items-center gap-2 text-gray-700 dark:text-slate-300">
                      <input
                        type="radio"
                        checked={selectedModelType === 'custom'}
                        onChange={() => setSelectedModelType('custom')}
                        className="w-4 h-4"
                      />
                      Custom Model
                    </label>
                  </div>

                  {selectedModelType === 'preset' ? (
                    <select
                      value={formData.model_name}
                      onChange={(e) => setFormData({ ...formData, model_name: e.target.value })}
                      className="input"
                    >
                      {models.map((model) => (
                        <option key={model.name} value={model.name}>
                          {model.display_name} ({model.provider}) - {model.description}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="text"
                      value={customModelName}
                      onChange={(e) => setCustomModelName(e.target.value)}
                      className="input"
                      placeholder="Enter custom model name (e.g., my-custom-model)"
                    />
                  )}

                  {selectedModelType === 'preset' && selectedModel && (
                    <div className="text-xs text-gray-600 dark:text-slate-400 bg-slate-100 dark:bg-slate-800 p-3 rounded">
                      <div className="font-medium text-gray-900 dark:text-slate-300 mb-1">{selectedModel.display_name}</div>
                      <div>Provider: {selectedModel.provider}</div>
                      <div>{selectedModel.description}</div>
                    </div>
                  )}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Enabled Tools</label>
                <div className="space-y-2">
                  {['sql', 'http', 'code', 'shell'].map((tool) => (
                    <label key={tool} className="flex items-center gap-2 text-slate-300">
                      <input
                        type="checkbox"
                        checked={formData.enabled_tools?.includes(tool) || false}
                        onChange={(e) => {
                          const tools = formData.enabled_tools || []
                          if (e.target.checked) {
                            setFormData({ ...formData, enabled_tools: [...tools, tool] })
                          } else {
                            setFormData({
                              ...formData,
                              enabled_tools: tools.filter((t) => t !== tool),
                            })
                          }
                        }}
                        className="w-4 h-4"
                      />
                      {tool.toUpperCase()}
                    </label>
                  ))}
                </div>
              </div>
            </div>

            <div className="p-6 border-t border-slate-200 dark:border-slate-800 flex justify-end gap-3">
              <button
                onClick={() => {
                  setShowCreateModal(false)
                  resetForm()
                }}
                className="btn btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateAgent}
                disabled={loading || !formData.name || (selectedModelType === 'custom' && !customModelName)}
                className="btn btn-primary"
              >
                {loading ? 'Creating...' : editingAgent ? 'Update' : 'Create'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

