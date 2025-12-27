'use client'

import { useState, useEffect } from 'react'
import { profilesAPI, neurondbAPI, type Profile, type CollectionInfo, type SearchRequest, type SearchResult } from '@/lib/api'
import ProfileSelector from '@/components/ProfileSelector'
import JSONViewer from '@/components/JSONViewer'
import { 
  MagnifyingGlassIcon,
  TableCellsIcon,
  ChartBarIcon,
  ArrowPathIcon
} from '@/components/Icons'

export default function NeuronDBPage() {
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [selectedProfile, setSelectedProfile] = useState<string>('')
  const [collections, setCollections] = useState<CollectionInfo[]>([])
  const [selectedCollection, setSelectedCollection] = useState<CollectionInfo | null>(null)
  const [queryText, setQueryText] = useState('')
  const [limit, setLimit] = useState(10)
  const [distanceType, setDistanceType] = useState<'l2' | 'cosine' | 'inner_product'>('cosine')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [loadingCollections, setLoadingCollections] = useState(false)

  useEffect(() => {
    loadProfiles()
  }, [])

  useEffect(() => {
    if (selectedProfile) {
      loadCollections()
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

  const loadCollections = async () => {
    if (!selectedProfile) return
    setLoadingCollections(true)
    try {
      const response = await neurondbAPI.listCollections(selectedProfile)
      setCollections(response.data)
      if (response.data.length > 0 && !selectedCollection) {
        setSelectedCollection(response.data[0])
      }
    } catch (error) {
      console.error('Failed to load collections:', error)
    } finally {
      setLoadingCollections(false)
    }
  }

  const handleSearch = async () => {
    if (!selectedProfile || !selectedCollection || !queryText) return
    
    setLoading(true)
    try {
      // For MVP, we'll need to generate embeddings on the backend
      // For now, this is a placeholder that shows the structure
      const request: SearchRequest = {
        collection: selectedCollection.name,
        schema: selectedCollection.schema,
        query_text: queryText,
        limit,
        distance_type: distanceType,
        query_vector: [], // This should be generated from query_text on backend
      }
      
      const response = await neurondbAPI.search(selectedProfile, request)
      setResults(response.data)
    } catch (error: any) {
      console.error('Search failed:', error)
      alert(error.response?.data?.error || error.message || 'Search failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="h-full flex flex-col bg-slate-800">
      {/* Header */}
      <div className="bg-slate-800 border-b border-slate-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-slate-100">NeuronDB Console</h1>
            <p className="text-sm text-slate-400 mt-1">Search collections and manage vector data</p>
          </div>
          <div className="w-64">
            <ProfileSelector
              profiles={profiles}
              selectedProfile={selectedProfile}
              onSelect={setSelectedProfile}
            />
          </div>
        </div>
      </div>

      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Collections Selector */}
        <div className="bg-slate-800 border-b border-slate-700 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div>
                <h2 className="text-lg font-semibold text-slate-100">Collections</h2>
                <p className="text-sm text-slate-400 mt-1">{collections.length} collections</p>
              </div>
              <select
                value={selectedCollection?.name || ''}
                onChange={(e) => {
                  const collection = collections.find(c => c.name === e.target.value)
                  setSelectedCollection(collection || null)
                }}
                className="input max-w-xs"
              >
                <option value="">Select a collection</option>
                {collections.map((collection) => (
                  <option key={collection.name} value={collection.name}>
                    {collection.name} ({collection.schema})
                  </option>
                ))}
              </select>
            </div>
            <button
              onClick={loadCollections}
              disabled={loadingCollections}
              className="p-2 text-slate-500 hover:text-slate-400 disabled:opacity-50"
            >
              <ArrowPathIcon className={`w-5 h-5 ${loadingCollections ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Search Form */}
          <div className="bg-slate-800 border-b border-slate-700 p-6">
            <h2 className="text-lg font-semibold text-slate-100 mb-4">Vector Search</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-200 mb-2">
                  Query Text
                </label>
                <textarea
                  value={queryText}
                  onChange={(e) => setQueryText(e.target.value)}
                  className="input"
                  rows={3}
                  placeholder="Enter search query..."
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-200 mb-2">
                    Limit: {limit}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="100"
                    value={limit}
                    onChange={(e) => setLimit(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-slate-200 mb-2">
                    Distance Metric
                  </label>
                  <select
                    value={distanceType}
                    onChange={(e) => setDistanceType(e.target.value as any)}
                    className="input"
                  >
                    <option value="cosine">Cosine</option>
                    <option value="l2">L2 (Euclidean)</option>
                    <option value="inner_product">Inner Product</option>
                  </select>
                </div>
              </div>
              
              <button
                onClick={handleSearch}
                disabled={loading || !queryText || !selectedCollection}
                className="btn btn-primary flex items-center gap-2"
              >
                <MagnifyingGlassIcon className="w-4 h-4" />
                {loading ? 'Searching...' : 'Search'}
              </button>
            </div>
          </div>

          {/* Results */}
          <div className="flex-1 overflow-y-auto p-6">
            {results.length === 0 && !loading && (
              <div className="text-center py-12">
                <MagnifyingGlassIcon className="w-12 h-12 text-slate-500 mx-auto mb-4" />
                <p className="text-slate-400">No results yet. Enter a query and click Search.</p>
              </div>
            )}
            
            {loading && (
              <div className="text-center py-12">
                <ArrowPathIcon className="w-8 h-8 text-blue-500 mx-auto mb-4 animate-spin" />
                <p className="text-slate-400">Searching...</p>
              </div>
            )}
            
            <div className="space-y-4">
              {results.map((result, idx) => (
                <div key={idx} className="card animate-fade-in">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm font-semibold text-slate-200">Result #{idx + 1}</span>
                        <span className="text-xs text-slate-400">ID: {String(result.id)}</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-4 text-sm">
                      <div className="text-right">
                        <div className="text-slate-400">Score</div>
                        <div className="font-semibold text-green-600">{result.score.toFixed(4)}</div>
                      </div>
                      <div className="text-right">
                        <div className="text-slate-400">Distance</div>
                        <div className="font-semibold text-slate-200">{result.distance.toFixed(4)}</div>
                      </div>
                    </div>
                  </div>
                  <JSONViewer data={result.data} title="Data" defaultExpanded={false} />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
