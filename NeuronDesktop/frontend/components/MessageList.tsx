'use client'

import { useEffect, useRef } from 'react'
import { Message } from '@/lib/api'
import ChatMessage from './ChatMessage'
import MarkdownContent from './MarkdownContent'

interface MessageListProps {
  messages: Message[]
  streamingContent?: string
  isStreaming?: boolean
}

export default function MessageList({ messages, streamingContent, isStreaming }: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, streamingContent])

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.length === 0 && !streamingContent && (
        <div className="flex items-center justify-center h-full">
          <div className="text-center">
            <p className="text-slate-400 text-lg mb-2">Start a conversation</p>
            <p className="text-slate-500 text-sm">Send a message to begin chatting with the agent</p>
          </div>
        </div>
      )}
      
      {messages.map((message) => (
        <ChatMessage key={message.id} message={message} />
      ))}
      
      {isStreaming && streamingContent && (
        <div className="flex justify-start mb-4">
          <div className="max-w-[80%] order-1">
            <div className="rounded-lg px-4 py-3 bg-slate-800 text-slate-100 border border-slate-700">
              <MarkdownContent content={streamingContent} />
              <span className="inline-block w-2 h-4 bg-slate-400 ml-1 animate-pulse" />
            </div>
          </div>
          <div className="order-2 mr-3 flex-shrink-0">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center">
              <span className="text-white text-sm font-bold">AI</span>
            </div>
          </div>
        </div>
      )}
      
      <div ref={messagesEndRef} />
    </div>
  )
}

