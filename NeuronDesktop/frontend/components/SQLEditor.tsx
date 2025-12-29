'use client'

import { useEffect, useState } from 'react'
import CodeMirror from '@uiw/react-codemirror'
import { sql } from '@codemirror/lang-sql'
import { oneDark } from '@codemirror/theme-one-dark'
import { EditorView } from '@codemirror/view'

interface SQLEditorProps {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  className?: string
  minHeight?: string
}

export default function SQLEditor({
  value,
  onChange,
  placeholder,
  className = '',
  minHeight = '300px',
}: SQLEditorProps) {
  const [isDark, setIsDark] = useState(false)

  // Detect dark mode from document
  useEffect(() => {
    const checkDarkMode = () => {
      setIsDark(document.documentElement.classList.contains('dark'))
    }
    
    checkDarkMode()
    
    // Watch for dark mode changes
    const observer = new MutationObserver(checkDarkMode)
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class'],
    })
    
    return () => observer.disconnect()
  }, [])

  // PostgreSQL-specific SQL configuration
  const extensions = [
    sql({
      dialect: 'postgresql', // Use PostgreSQL dialect for proper syntax highlighting
      upperCaseKeywords: false, // Keep keywords in their original case
    }),
    EditorView.lineWrapping, // Enable line wrapping
    EditorView.theme({
      '&': {
        fontSize: '14px',
        fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
      },
      '.cm-content': {
        minHeight: minHeight,
        padding: '16px',
      },
      '.cm-editor': {
        borderRadius: '8px',
        border: '1px solid',
        borderColor: isDark ? 'rgb(51, 65, 85)' : 'rgb(203, 213, 225)',
        backgroundColor: isDark ? 'rgb(15, 23, 42)' : 'rgb(255, 255, 255)',
      },
      '.cm-scroller': {
        overflow: 'auto',
      },
      '.cm-focused': {
        outline: 'none',
      },
      '.cm-editor.cm-focused': {
        borderColor: isDark ? 'rgb(59, 130, 246)' : 'rgb(59, 130, 246)',
        boxShadow: isDark 
          ? '0 0 0 3px rgba(59, 130, 246, 0.1)' 
          : '0 0 0 3px rgba(59, 130, 246, 0.1)',
      },
    }, { dark: isDark }),
  ]

  return (
    <div className={className}>
      <CodeMirror
        value={value}
        onChange={onChange}
        extensions={extensions}
        placeholder={placeholder}
        theme={isDark ? oneDark : undefined}
        basicSetup={{
          lineNumbers: true,
          foldGutter: true,
          dropCursor: false,
          allowMultipleSelections: false,
          indentOnInput: true,
          bracketMatching: true,
          closeBrackets: true,
          autocompletion: true,
          highlightSelectionMatches: true,
          searchKeymap: true,
          defaultKeymap: true,
          historyKeymap: true,
          foldKeymap: true,
        }}
        style={{
          fontSize: '14px',
          minHeight: minHeight,
        }}
      />
    </div>
  )
}

