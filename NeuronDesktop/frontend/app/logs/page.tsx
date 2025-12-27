'use client'

import { useState } from 'react'
import { 
  DocumentTextIcon,
  FunnelIcon,
  CalendarIcon,
  CheckCircleIcon,
  XCircleIcon
} from '@/components/Icons'
import JSONViewer from '@/components/JSONViewer'

export default function LogsPage() {
  const [filter, setFilter] = useState({
    status: 'all',
    endpoint: '',
    dateRange: 'today',
  })

  // Mock data for now
  const logs = [
    {
      id: '1',
      timestamp: new Date(),
      method: 'POST',
      endpoint: '/api/v1/mcp/tools/call',
      statusCode: 200,
      duration: 145,
      requestBody: { name: 'vector_search', arguments: { query: 'test' } },
      responseBody: { result: 'success' },
    },
  ]

  return (
    <div className="h-full flex flex-col bg-slate-800">
      {/* Header */}
      <div className="bg-slate-800 border-b border-slate-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-slate-100">Request Logs</h1>
            <p className="text-sm text-slate-400 mt-1">View and inspect all API requests and responses</p>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-slate-800 border-b border-slate-700 px-6 py-4">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <FunnelIcon className="w-5 h-5 text-slate-500" />
            <span className="text-sm font-medium text-slate-200">Filters:</span>
          </div>
          
          <select
            value={filter.status}
            onChange={(e) => setFilter({ ...filter, status: e.target.value })}
            className="input w-40"
          >
            <option value="all">All Status</option>
            <option value="200">Success (200)</option>
            <option value="400">Client Error (4xx)</option>
            <option value="500">Server Error (5xx)</option>
          </select>
          
          <input
            type="text"
            value={filter.endpoint}
            onChange={(e) => setFilter({ ...filter, endpoint: e.target.value })}
            placeholder="Filter by endpoint..."
            className="input w-64"
          />
          
          <select
            value={filter.dateRange}
            onChange={(e) => setFilter({ ...filter, dateRange: e.target.value })}
            className="input w-40"
          >
            <option value="today">Today</option>
            <option value="week">This Week</option>
            <option value="month">This Month</option>
            <option value="all">All Time</option>
          </select>
        </div>
      </div>

      {/* Logs List */}
      <div className="flex-1 overflow-y-auto p-6">
        {logs.length === 0 ? (
          <div className="text-center py-12">
            <DocumentTextIcon className="w-12 h-12 text-slate-500 mx-auto mb-4" />
            <p className="text-slate-400">No logs available yet.</p>
            <p className="text-sm text-slate-500 mt-2">Logs will appear here as you make requests.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {logs.map((log) => (
              <div key={log.id} className="card">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    {log.statusCode >= 200 && log.statusCode < 300 ? (
                      <CheckCircleIcon className="w-5 h-5 text-green-500" />
                    ) : (
                      <XCircleIcon className="w-5 h-5 text-red-500" />
                    )}
                    <div>
                      <div className="flex items-center gap-2">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          log.method === 'GET' ? 'bg-blue-100 text-blue-800' :
                          log.method === 'POST' ? 'bg-green-100 text-green-800' :
                          log.method === 'PUT' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                          {log.method}
                        </span>
                        <span className="font-mono text-sm text-slate-100">{log.endpoint}</span>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          log.statusCode >= 200 && log.statusCode < 300 ? 'bg-green-100 text-green-800' :
                          log.statusCode >= 400 && log.statusCode < 500 ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                          {log.statusCode}
                        </span>
                      </div>
                      <div className="flex items-center gap-4 mt-2 text-xs text-slate-400">
                        <div className="flex items-center gap-1">
                          <CalendarIcon className="w-3 h-3" />
                          {log.timestamp.toLocaleString()}
                        </div>
                        <div>{log.duration}ms</div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h4 className="text-sm font-medium text-slate-200 mb-2">Request</h4>
                    <JSONViewer data={log.requestBody} defaultExpanded={false} />
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-slate-200 mb-2">Response</h4>
                    <JSONViewer data={log.responseBody} defaultExpanded={false} />
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
