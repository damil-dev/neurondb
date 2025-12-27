'use client'

import { useState } from 'react'

interface JSONViewerProps {
  data: any
  title?: string
  defaultExpanded?: boolean
}

export default function JSONViewer({ data, title, defaultExpanded = false }: JSONViewerProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded)
  const [copied, setCopied] = useState(false)

  const jsonString = JSON.stringify(data, null, 2)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(jsonString)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="border border-slate-700 rounded-lg overflow-hidden bg-slate-900">
      {(title || !defaultExpanded) && (
        <div className="flex items-center justify-between px-4 py-2 bg-slate-800 border-b border-slate-700">
          {title && <h3 className="text-sm font-medium text-slate-200">{title}</h3>}
          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="text-xs text-slate-400 hover:text-slate-100 px-2 py-1 rounded hover:bg-slate-800"
            >
              {copied ? 'Copied!' : 'Copy'}
            </button>
            {!defaultExpanded && (
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="text-xs text-slate-400 hover:text-slate-100 px-2 py-1 rounded hover:bg-slate-800"
              >
                {isExpanded ? 'Collapse' : 'Expand'}
              </button>
            )}
          </div>
        </div>
      )}
      {isExpanded && (
        <pre className="p-4 bg-slate-950 text-slate-200 text-xs overflow-x-auto max-h-96 overflow-y-auto font-mono">
          {jsonString}
        </pre>
      )}
    </div>
  )
}

