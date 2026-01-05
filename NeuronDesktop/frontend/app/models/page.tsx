'use client'

import { useState, useEffect } from 'react'
import { profilesAPI, neurondbAPI } from '@/lib/api'
import { showSuccessToast, showErrorToast } from '@/lib/errors'
import { 
  PlusIcon, 
  KeyIcon, 
  EyeIcon, 
  EyeSlashIcon,
  TrashIcon,
  CheckIcon,
  XMarkIcon
} from '@/components/Icons'

interface Model {
  id: string
  name: string
  provider: string
  model_type: string
  api_key_set: boolean
  config: any
}

export default function ModelsPage() {
  const [profiles, setProfiles] = useState<any[]>([])
  const [selectedProfile, setSelectedProfile] = useState<string>('')
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(false)
  const [showAddModal, setShowAddModal] = useState(false)
  const [showKeyModal, setShowKeyModal] = useState(false)
  const [selectedModel, setSelectedModel] = useState<Model | null>(null)
  const [apiKey, setApiKey] = useState('')
  const [showKey, setShowKey] = useState(false)

  useEffect(() => {
    loadProfiles()
  }, [])

  useEffect(() => {
    if (selectedProfile) {
      loadModels()
    }
  }, [selectedProfile])

  const loadProfiles = async () => {
    try {
      const response = await profilesAPI.list()
      setProfiles(response.data)
      if (response.data.length > 0) {
        const defaultProfile = response.data.find((p: any) => p.is_default) || response.data[0]
        setSelectedProfile(defaultProfile.id)
      }
    } catch (error: any) {
      showErrorToast('Failed to load profiles: ' + (error.response?.data?.error || error.message))
    }
  }

  const loadModels = async () => {
    if (!selectedProfile) return
    setLoading(true)
    try {
      // This would call a models API endpoint
      // For now, we'll use a placeholder
      const response = await neurondbAPI.listModels(selectedProfile)
      setModels(response.data || [])
    } catch (error: any) {
      showErrorToast('Failed to load models: ' + (error.response?.data?.error || error.message))
    } finally {
      setLoading(false)
    }
  }

  const handleAddModel = async (modelData: any) => {
    setLoading(true)
    try {
      await neurondbAPI.addModel(selectedProfile, modelData)
      showSuccessToast('Model added successfully')
      setShowAddModal(false)
      loadModels()
    } catch (error: any) {
      showErrorToast('Failed to add model: ' + (error.response?.data?.error || error.message))
    } finally {
      setLoading(false)
    }
  }

  const handleSetApiKey = async () => {
    if (!selectedModel || !apiKey) return
    setLoading(true)
    try {
      await neurondbAPI.setModelKey(selectedProfile, selectedModel.name, apiKey)
      showSuccessToast('API key set successfully')
      setShowKeyModal(false)
      setApiKey('')
      loadModels()
    } catch (error: any) {
      showErrorToast('Failed to set API key: ' + (error.response?.data?.error || error.message))
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteModel = async (modelId: string) => {
    if (!confirm('Are you sure you want to delete this model?')) return
    setLoading(true)
    try {
      await neurondbAPI.deleteModel(selectedProfile, modelId)
      showSuccessToast('Model deleted successfully')
      loadModels()
    } catch (error: any) {
      showErrorToast('Failed to delete model: ' + (error.response?.data?.error || error.message))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-full bg-transparent p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">Model & Key Management</h1>
            <p className="text-slate-600 dark:text-slate-400 mt-1">Manage LLM models and API keys</p>
          </div>
          <button
            onClick={() => setShowAddModal(true)}
            className="btn btn-primary"
          >
            <PlusIcon className="w-5 h-5 mr-2" />
            Add Model
          </button>
        </div>

        {/* Profile Selector */}
        {profiles.length > 0 && (
          <div className="mb-6">
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Profile
            </label>
            <select
              value={selectedProfile}
              onChange={(e) => setSelectedProfile(e.target.value)}
              className="input"
            >
              {profiles.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.name} {profile.is_default && '(Default)'}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Models List */}
        {loading && models.length === 0 ? (
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mx-auto"></div>
            <p className="text-slate-600 dark:text-slate-400 mt-4">Loading models...</p>
          </div>
        ) : models.length === 0 ? (
          <div className="text-center py-12 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
            <p className="text-slate-600 dark:text-slate-400">No models configured</p>
            <button
              onClick={() => setShowAddModal(true)}
              className="btn btn-primary mt-4"
            >
              Add Your First Model
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {models.map((model) => (
              <div
                key={model.id}
                className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-6 hover:shadow-lg transition-shadow"
              >
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                      {model.name}
                    </h3>
                    <p className="text-sm text-slate-600 dark:text-slate-400">{model.provider}</p>
                  </div>
                  <button
                    onClick={() => handleDeleteModel(model.id)}
                    className="text-red-600 hover:text-red-700"
                  >
                    <TrashIcon className="w-5 h-5" />
                  </button>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-slate-600 dark:text-slate-400">API Key:</span>
                    <div className="flex items-center gap-2">
                      {model.api_key_set ? (
                        <span className="flex items-center text-green-600">
                          <CheckIcon className="w-4 h-4 mr-1" />
                          Set
                        </span>
                      ) : (
                        <span className="flex items-center text-red-600">
                          <XMarkIcon className="w-4 h-4 mr-1" />
                          Not Set
                        </span>
                      )}
                    </div>
                  </div>
                  
                  <button
                    onClick={() => {
                      setSelectedModel(model)
                      setShowKeyModal(true)
                    }}
                    className="btn btn-secondary w-full mt-4"
                  >
                    <KeyIcon className="w-4 h-4 mr-2" />
                    {model.api_key_set ? 'Update Key' : 'Set API Key'}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Add Model Modal */}
        {showAddModal && (
          <AddModelModal
            onClose={() => setShowAddModal(false)}
            onAdd={handleAddModel}
          />
        )}

        {/* Set API Key Modal */}
        {showKeyModal && selectedModel && (
          <SetApiKeyModal
            model={selectedModel}
            onClose={() => {
              setShowKeyModal(false)
              setSelectedModel(null)
              setApiKey('')
            }}
            onSave={handleSetApiKey}
            apiKey={apiKey}
            setApiKey={setApiKey}
            showKey={showKey}
            setShowKey={setShowKey}
          />
        )}
      </div>
    </div>
  )
}

function AddModelModal({ onClose, onAdd }: { onClose: () => void; onAdd: (data: any) => void }) {
  const [formData, setFormData] = useState({
    name: '',
    provider: 'openai',
    model_type: 'text',
    config: {},
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onAdd(formData)
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-slate-800 rounded-lg p-6 max-w-md w-full">
        <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-4">Add Model</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Model Name
            </label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              className="input w-full"
              placeholder="gpt-4"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Provider
            </label>
            <select
              value={formData.provider}
              onChange={(e) => setFormData({ ...formData, provider: e.target.value })}
              className="input w-full"
            >
              <option value="openai">OpenAI</option>
              <option value="anthropic">Anthropic</option>
              <option value="huggingface">HuggingFace</option>
              <option value="local">Local</option>
            </select>
          </div>
          <div className="flex gap-2">
            <button type="button" onClick={onClose} className="btn btn-secondary flex-1">
              Cancel
            </button>
            <button type="submit" className="btn btn-primary flex-1">
              Add Model
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

function SetApiKeyModal({
  model,
  onClose,
  onSave,
  apiKey,
  setApiKey,
  showKey,
  setShowKey,
}: {
  model: Model
  onClose: () => void
  onSave: () => void
  apiKey: string
  setApiKey: (key: string) => void
  showKey: boolean
  setShowKey: (show: boolean) => void
}) {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-slate-800 rounded-lg p-6 max-w-md w-full">
        <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-4">
          Set API Key for {model.name}
        </h2>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              API Key
            </label>
            <div className="relative">
              <input
                type={showKey ? 'text' : 'password'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="input w-full pr-10"
                placeholder="sk-..."
              />
              <button
                type="button"
                onClick={() => setShowKey(!showKey)}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-600 hover:text-slate-800"
              >
                {showKey ? <EyeSlashIcon className="w-5 h-5" /> : <EyeIcon className="w-5 h-5" />}
              </button>
            </div>
          </div>
          <div className="flex gap-2">
            <button type="button" onClick={onClose} className="btn btn-secondary flex-1">
              Cancel
            </button>
            <button type="button" onClick={onSave} className="btn btn-primary flex-1">
              Save
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
