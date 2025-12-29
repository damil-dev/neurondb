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
    <div className="border border-slate-300 rounded-lg overflow-hidden bg-white">
      {(title || !defaultExpanded) && (
        <div className="flex items-center justify-between px-4 py-2 bg-slate-100 border-b border-slate-300">
          {title && <h3 className="text-sm font-medium text-gray-900">{title}</h3>}
          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="text-xs text-gray-700 hover:text-gray-900 px-2 py-1 rounded hover:bg-slate-200"
            >
              {copied ? 'Copied!' : 'Copy'}
            </button>
            {!defaultExpanded && (
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="text-xs text-gray-700 hover:text-gray-900 px-2 py-1 rounded hover:bg-slate-200"
              >
                {isExpanded ? 'Collapse' : 'Expand'}
              </button>
            )}
          </div>
        </div>
      )}
      {isExpanded && (
        <pre className="p-4 bg-slate-50 text-gray-800 text-xs overflow-x-auto max-h-96 overflow-y-auto font-mono">
          {jsonString}
        </pre>
      )}
    </div>
  )
}

