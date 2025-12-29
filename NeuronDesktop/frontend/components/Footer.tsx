'use client'

import { useEffect, useState } from 'react'
import { ActivityIcon } from './Icons'

export default function Footer() {
  const [apiStatus, setApiStatus] = useState<'online' | 'offline' | 'checking'>('checking')

  useEffect(() => {
    const checkApiStatus = async () => {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8081'
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), 2000)
        
        const response = await fetch(`${apiUrl}/health`, {
          method: 'GET',
          signal: controller.signal,
        })
        clearTimeout(timeoutId)
        
        if (response.ok) {
          setApiStatus('online')
        } else {
          setApiStatus('offline')
        }
      } catch (error) {
        setApiStatus('offline')
      }
    }

    checkApiStatus()
    const interval = setInterval(checkApiStatus, 30000) // Check every 30 seconds

    return () => clearInterval(interval)
  }, [])

  const getStatusColor = () => {
    switch (apiStatus) {
      case 'online':
        return 'bg-green-500'
      case 'offline':
        return 'bg-red-500'
      default:
        return 'bg-yellow-500'
    }
  }

  const getStatusText = () => {
    switch (apiStatus) {
      case 'online':
        return 'Online'
      case 'offline':
        return 'Offline'
      default:
        return 'Checking...'
    }
  }

  return (
    <footer className="h-10 bg-slate-100 dark:bg-slate-900 border-t border-slate-200 dark:border-slate-800 flex items-center justify-between px-4 text-xs text-gray-700 dark:text-slate-400 select-none">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <span className="font-medium text-gray-900 dark:text-slate-100">NeuronDesktop</span>
          <span className="text-gray-700 dark:text-slate-500">v1.0.0</span>
        </div>
        <span className="text-gray-600 dark:text-slate-600">•</span>
        <span className="text-gray-700 dark:text-slate-300">NeuronDB PostgreSQL AI Factory</span>
      </div>
      
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <ActivityIcon className={`w-3 h-3 ${apiStatus === 'online' ? 'text-green-600 dark:text-green-400' : apiStatus === 'offline' ? 'text-red-600 dark:text-red-400' : 'text-yellow-600 dark:text-yellow-400'}`} />
          <span className={apiStatus === 'online' ? 'text-green-600 dark:text-green-400' : apiStatus === 'offline' ? 'text-red-600 dark:text-red-400' : 'text-yellow-600 dark:text-yellow-400'}>
            API {getStatusText()}
          </span>
        </div>
        <span className="text-gray-600 dark:text-slate-600">•</span>
        <span className="text-gray-700 dark:text-slate-300">© 2024 NeuronDB</span>
      </div>
    </footer>
  )
}
