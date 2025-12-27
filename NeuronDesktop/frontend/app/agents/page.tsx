'use client'

import { useState, useEffect } from 'react'
import { agentAPI, profilesAPI, type Profile, type Agent, type Model, type CreateAgentRequest } from '@/lib/api'
import ProfileSelector from '@/components/ProfileSelector'
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
        setSelectedProfile(response.data[0].id)
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
    if (!selectedProfile) return
    setLoading(true)
    try {
      const request: CreateAgentRequest = {
        ...formData,
        model_name: selectedModelType === 'custom' ? customModelName : formData.model_name,
      }
      await agentAPI.createAgent(selectedProfile, request)
      setShowCreateModal(false)
      resetForm()
      loadAgents()
    } catch (error) {
      console.error('Failed to create agent:', error)
      alert('Failed to create agent: ' + (error as Error).message)
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
    <div className="h-full overflow-auto bg-slate-950 p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold text-slate-100 mb-2">NeuronAgent Management</h1>
          <p className="text-slate-400">Create and manage AI agents with model selection</p>
        </div>
        <div className="flex items-center gap-4">
          <ProfileSelector
            profiles={profiles}
            selectedProfile={selectedProfile}
            onSelect={setSelectedProfile}
          />
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
          <p className="text-slate-400">Please select a profile to manage agents</p>
        </div>
      ) : loading ? (
        <div className="card text-center py-12">
          <p className="text-slate-400">Loading agents...</p>
        </div>
      ) : agents.length === 0 ? (
        <div className="card text-center py-12">
          <ChatBubbleLeftRightIcon className="w-16 h-16 text-slate-600 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-slate-200 mb-2">No Agents Yet</h3>
          <p className="text-slate-400 mb-6">Create your first agent to get started</p>
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
                  <h3 className="text-xl font-semibold text-slate-100 mb-1">{agent.name}</h3>
                  {agent.description && (
                    <p className="text-sm text-slate-400 mb-2">{agent.description}</p>
                  )}
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => openEditModal(agent)}
                    className="p-2 text-slate-400 hover:text-blue-400 hover:bg-slate-800 rounded"
                  >
                    <PencilIcon className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => handleDeleteAgent(agent.id)}
                    className="p-2 text-slate-400 hover:text-red-400 hover:bg-slate-800 rounded"
                  >
                    <TrashIcon className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2">
                  <CpuChipIcon className="w-4 h-4 text-blue-400" />
                  <span className="text-slate-300">Model:</span>
                  <span className="text-slate-100 font-medium">{agent.model_name || 'Not set'}</span>
                </div>
                {agent.enabled_tools && agent.enabled_tools.length > 0 && (
                  <div>
                    <span className="text-slate-300">Tools: </span>
                    <span className="text-slate-100">{agent.enabled_tools.join(', ')}</span>
                  </div>
                )}
                {agent.system_prompt && (
                  <div className="text-xs text-slate-400 truncate" title={agent.system_prompt}>
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
          <div className="bg-slate-900 rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-slate-800">
              <h2 className="text-2xl font-bold text-slate-100">
                {editingAgent ? 'Edit Agent' : 'Create New Agent'}
              </h2>
            </div>

            <div className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Name *</label>
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
                    <label className="flex items-center gap-2 text-slate-300">
                      <input
                        type="radio"
                        checked={selectedModelType === 'preset'}
                        onChange={() => setSelectedModelType('preset')}
                        className="w-4 h-4"
                      />
                      Preset Model
                    </label>
                    <label className="flex items-center gap-2 text-slate-300">
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
                    <div className="text-xs text-slate-400 bg-slate-800 p-3 rounded">
                      <div className="font-medium text-slate-300 mb-1">{selectedModel.display_name}</div>
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

            <div className="p-6 border-t border-slate-800 flex justify-end gap-3">
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

