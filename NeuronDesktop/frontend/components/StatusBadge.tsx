'use client'

interface StatusBadgeProps {
  status: 'connected' | 'disconnected' | 'error' | 'loading' | 'connecting'
  label?: string
}

export default function StatusBadge({ status, label }: StatusBadgeProps) {
  const config = {
    connected: {
      bg: 'bg-emerald-900/30',
      text: 'text-emerald-300',
      dot: 'bg-emerald-500',
      border: 'border-emerald-700/50',
      defaultLabel: 'Connected',
    },
    disconnected: {
      bg: 'bg-slate-800',
      text: 'text-slate-400',
      dot: 'bg-slate-500',
      border: 'border-slate-700',
      defaultLabel: 'Disconnected',
    },
    error: {
      bg: 'bg-red-900/30',
      text: 'text-red-300',
      dot: 'bg-red-500',
      border: 'border-red-700/50',
      defaultLabel: 'Error',
    },
    loading: {
      bg: 'bg-amber-900/30',
      text: 'text-amber-300',
      dot: 'bg-amber-500',
      border: 'border-amber-700/50',
      defaultLabel: 'Connecting...',
    },
    connecting: {
      bg: 'bg-amber-900/30',
      text: 'text-amber-300',
      dot: 'bg-amber-500',
      border: 'border-amber-700/50',
      defaultLabel: 'Connecting...',
    },
  }

  const style = config[status]

  return (
    <div className={`inline-flex items-center px-3 py-1.5 rounded-full text-sm font-semibold border ${style.bg} ${style.text} ${style.border}`}>
      <span className={`w-2 h-2 rounded-full mr-2 ${style.dot} ${status === 'loading' || status === 'connecting' ? 'animate-pulse' : ''}`}></span>
      {label || style.defaultLabel}
    </div>
  )
}

