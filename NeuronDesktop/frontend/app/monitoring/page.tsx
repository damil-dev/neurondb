'use client'

import { useState, useEffect, useRef } from 'react'
import { 
  CpuIcon, 
  ServerIcon, 
  HardDriveIcon,
  NetworkIcon,
  ActivityIcon,
  MemoryStickIcon
} from '@/components/Icons'

interface SystemMetrics {
  timestamp: string
  cpu: {
    usage_percent: number
    usage_per_core?: number[]
    count: number
    frequency?: number
  }
  memory: {
    total: number
    used: number
    available: number
    used_percent: number
  }
  disk: {
    total: number
    used: number
    free: number
    used_percent: number
  }
  network: {
    bytes_sent: number
    bytes_recv: number
    bytes_sent_rate?: number
    bytes_recv_rate?: number
  }
  process: {
    go_routines: number
    heap_alloc: number
    heap_sys: number
  }
}

const formatBytes = (bytes: number): string => {
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let size = bytes
  let unitIndex = 0
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024
    unitIndex++
  }
  return `${size.toFixed(2)} ${units[unitIndex]}`
}

const formatRate = (bytesPerSec: number): string => {
  return formatBytes(bytesPerSec) + '/s'
}

export default function MonitoringPage() {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null)
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8081/api/v1'
    const wsUrl = apiUrl.replace('http://', 'ws://').replace('https://', 'wss://')
    // System metrics endpoint doesn't require auth (public endpoint)
    const ws = new WebSocket(`${wsUrl}/system-metrics/ws`)

    ws.onopen = () => {
      setConnected(true)
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        setMetrics(data)
      } catch (error) {
        console.error('Failed to parse metrics:', error)
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setConnected(false)
    }

    ws.onclose = () => {
      setConnected(false)
    }

    wsRef.current = ws

    return () => {
      ws.close()
    }
  }, [])

  if (!metrics) {
    return (
      <div className="min-h-full bg-slate-950 p-8 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-slate-400">Connecting to metrics server...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-full bg-slate-950 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-slate-100 mb-2">System Monitoring</h1>
              <p className="text-slate-400">Real-time system metrics and performance monitoring</p>
            </div>
            <div className={`px-4 py-2 rounded-lg ${connected ? 'bg-emerald-900/30 text-emerald-300 border border-emerald-700/50' : 'bg-red-900/30 text-red-300 border border-red-700/50'}`}>
              {connected ? '● Connected' : '● Disconnected'}
            </div>
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {/* CPU Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center">
                  <CpuIcon className="w-6 h-6 text-blue-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-slate-100">CPU</h3>
                  <p className="text-sm text-slate-400">{metrics.cpu.count} cores</p>
                </div>
              </div>
            </div>
            <div className="mb-4">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-slate-400">Usage</span>
                <span className="text-slate-200 font-semibold">{metrics.cpu.usage_percent.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-3">
                <div
                  className="bg-gradient-to-r from-blue-500 to-blue-600 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${metrics.cpu.usage_percent}%` }}
                ></div>
              </div>
            </div>
            {metrics.cpu.usage_per_core && metrics.cpu.usage_per_core.length > 0 && (
              <div className="mt-4 pt-4 border-t border-slate-800">
                <p className="text-xs text-slate-400 mb-2">Per Core Usage</p>
                <div className="grid grid-cols-2 gap-2">
                  {metrics.cpu.usage_per_core.slice(0, 4).map((usage, idx) => (
                    <div key={idx} className="flex items-center gap-2">
                      <div className="flex-1 bg-slate-800 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${usage}%` }}
                        ></div>
                      </div>
                      <span className="text-xs text-slate-400 w-10 text-right">{usage.toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Memory Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-purple-500/20 rounded-lg flex items-center justify-center">
                  <MemoryStickIcon className="w-6 h-6 text-purple-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-slate-100">Memory</h3>
                  <p className="text-sm text-slate-400">{formatBytes(metrics.memory.total)}</p>
                </div>
              </div>
            </div>
            <div className="mb-4">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-slate-400">Used</span>
                <span className="text-slate-200 font-semibold">{formatBytes(metrics.memory.used)}</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-3">
                <div
                  className="bg-gradient-to-r from-purple-500 to-purple-600 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${metrics.memory.used_percent}%` }}
                ></div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-slate-400">Available</p>
                <p className="text-slate-200 font-semibold">{formatBytes(metrics.memory.available)}</p>
              </div>
              <div>
                <p className="text-slate-400">Usage</p>
                <p className="text-slate-200 font-semibold">{metrics.memory.used_percent.toFixed(1)}%</p>
              </div>
            </div>
          </div>

          {/* Disk Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-green-500/20 rounded-lg flex items-center justify-center">
                  <HardDriveIcon className="w-6 h-6 text-green-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-slate-100">Disk</h3>
                  <p className="text-sm text-slate-400">{formatBytes(metrics.disk.total)}</p>
                </div>
              </div>
            </div>
            <div className="mb-4">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-slate-400">Used</span>
                <span className="text-slate-200 font-semibold">{formatBytes(metrics.disk.used)}</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-3">
                <div
                  className="bg-gradient-to-r from-green-500 to-green-600 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${metrics.disk.used_percent}%` }}
                ></div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-slate-400">Free</p>
                <p className="text-slate-200 font-semibold">{formatBytes(metrics.disk.free)}</p>
              </div>
              <div>
                <p className="text-slate-400">Usage</p>
                <p className="text-slate-200 font-semibold">{metrics.disk.used_percent.toFixed(1)}%</p>
              </div>
            </div>
          </div>

          {/* Network Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-amber-500/20 rounded-lg flex items-center justify-center">
                  <NetworkIcon className="w-6 h-6 text-amber-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-slate-100">Network</h3>
                  <p className="text-sm text-slate-400">Traffic</p>
                </div>
              </div>
            </div>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-slate-400">Sent</span>
                  <span className="text-slate-200 font-semibold">{formatBytes(metrics.network.bytes_sent)}</span>
                </div>
                {metrics.network.bytes_sent_rate !== undefined && (
                  <p className="text-xs text-slate-500">{formatRate(metrics.network.bytes_sent_rate)}</p>
                )}
              </div>
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-slate-400">Received</span>
                  <span className="text-slate-200 font-semibold">{formatBytes(metrics.network.bytes_recv)}</span>
                </div>
                {metrics.network.bytes_recv_rate !== undefined && (
                  <p className="text-xs text-slate-500">{formatRate(metrics.network.bytes_recv_rate)}</p>
                )}
              </div>
            </div>
          </div>

          {/* Process Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-indigo-500/20 rounded-lg flex items-center justify-center">
                  <ActivityIcon className="w-6 h-6 text-indigo-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-slate-100">Process</h3>
                  <p className="text-sm text-slate-400">Go Runtime</p>
                </div>
              </div>
            </div>
            <div className="space-y-3">
              <div>
                <p className="text-sm text-slate-400 mb-1">Goroutines</p>
                <p className="text-xl font-semibold text-slate-100">{metrics.process.go_routines.toLocaleString()}</p>
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-slate-400">Heap Alloc</p>
                  <p className="text-slate-200 font-semibold">{formatBytes(metrics.process.heap_alloc)}</p>
                </div>
                <div>
                  <p className="text-slate-400">Heap Sys</p>
                  <p className="text-slate-200 font-semibold">{formatBytes(metrics.process.heap_sys)}</p>
                </div>
              </div>
            </div>
          </div>

          {/* System Info Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-slate-700 rounded-lg flex items-center justify-center">
                  <ServerIcon className="w-6 h-6 text-slate-300" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-slate-100">System</h3>
                  <p className="text-sm text-slate-400">Information</p>
                </div>
              </div>
            </div>
            <div className="space-y-3 text-sm">
              <div>
                <p className="text-slate-400">Last Update</p>
                <p className="text-slate-200 font-semibold">
                  {new Date(metrics.timestamp).toLocaleTimeString()}
                </p>
              </div>
              {metrics.cpu.frequency && (
                <div>
                  <p className="text-slate-400">CPU Frequency</p>
                  <p className="text-slate-200 font-semibold">{metrics.cpu.frequency.toFixed(0)} MHz</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

