'use client'

import { useEffect, useState } from 'react'
import { toastManager, type Toast } from '@/lib/toast'
import { CheckCircleIcon, XCircleIcon, ExclamationTriangleIcon, InformationCircleIcon, XMarkIcon } from '@/components/Icons'

export default function ToastContainer() {
  const [toasts, setToasts] = useState<Toast[]>([])

  useEffect(() => {
    const unsubscribe = toastManager.subscribe(setToasts)
    return unsubscribe
  }, [])

  if (toasts.length === 0) return null

  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-3 max-w-md pointer-events-none">
      {toasts.map((toast, index) => (
        <div key={toast.id} className="pointer-events-auto" style={{ animationDelay: `${index * 50}ms` }}>
          <ToastItem toast={toast} />
        </div>
      ))}
    </div>
  )
}

function ToastItem({ toast }: { toast: Toast }) {
  const [isVisible, setIsVisible] = useState(true)
  const [isExiting, setIsExiting] = useState(false)

  useEffect(() => {
    setIsVisible(true)
    return () => setIsVisible(false)
  }, [])

  const handleClose = () => {
    setIsExiting(true)
    setTimeout(() => {
      setIsVisible(false)
      toastManager.remove(toast.id)
    }, 300)
  }

  if (!isVisible) return null

  const getIcon = () => {
    const iconClass = "w-6 h-6"
    switch (toast.type) {
      case 'success':
        return <CheckCircleIcon className={`${iconClass} text-green-500 animate-scale-in`} />
      case 'error':
        return <XCircleIcon className={`${iconClass} text-red-500 animate-scale-in`} />
      case 'warning':
        return <ExclamationTriangleIcon className={`${iconClass} text-yellow-500 animate-scale-in`} />
      case 'info':
        return <InformationCircleIcon className={`${iconClass} text-blue-500 animate-scale-in`} />
    }
  }

  const getBgColor = () => {
    switch (toast.type) {
      case 'success':
        return 'bg-green-50/95 dark:bg-green-900/30 border-green-300 dark:border-green-700'
      case 'error':
        return 'bg-red-50/95 dark:bg-red-900/30 border-red-300 dark:border-red-700'
      case 'warning':
        return 'bg-yellow-50/95 dark:bg-yellow-900/30 border-yellow-300 dark:border-yellow-700'
      case 'info':
        return 'bg-blue-50/95 dark:bg-blue-900/30 border-blue-300 dark:border-blue-700'
    }
  }

  const getTextColor = () => {
    switch (toast.type) {
      case 'success':
        return 'text-green-800 dark:text-green-200'
      case 'error':
        return 'text-red-800 dark:text-red-200'
      case 'warning':
        return 'text-yellow-800 dark:text-yellow-200'
      case 'info':
        return 'text-blue-800 dark:text-blue-200'
    }
  }

  const getGlowColor = () => {
    switch (toast.type) {
      case 'success':
        return 'shadow-green-500/20'
      case 'error':
        return 'shadow-red-500/20'
      case 'warning':
        return 'shadow-yellow-500/20'
      case 'info':
        return 'shadow-blue-500/20'
    }
  }

  return (
    <div
      className={`
        ${getBgColor()} ${getTextColor()} 
        border-2 rounded-xl shadow-2xl ${getGlowColor()}
        backdrop-blur-xl
        p-4 flex items-start gap-3 
        ${isExiting ? 'animate-slide-in-right opacity-0 scale-95' : 'animate-slide-in-right'}
        transition-all duration-300
        hover:scale-[1.02] hover:shadow-2xl
        min-w-[320px] max-w-md
      `}
    >
      <div className="flex-shrink-0 mt-0.5 animate-fade-in">{getIcon()}</div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold break-words leading-relaxed">{toast.message}</p>
      </div>
      <button
        onClick={handleClose}
        className="flex-shrink-0 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 hover:scale-110 transition-all duration-200 rounded-lg p-1 hover:bg-black/5 dark:hover:bg-white/5"
        aria-label="Close"
      >
        <XMarkIcon className="w-4 h-4" />
      </button>
    </div>
  )
}






