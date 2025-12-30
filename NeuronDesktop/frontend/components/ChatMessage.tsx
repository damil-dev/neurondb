'use client'

import { Message } from '@/lib/api'
import MarkdownContent from './MarkdownContent'

interface ChatMessageProps {
  message: Message
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user'
  const timestamp = message.created_at ? new Date(message.created_at) : new Date()
  const timeString = timestamp.toLocaleTimeString('en-US', { 
    hour: '2-digit', 
    minute: '2-digit',
    hour12: false 
  })

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-[80%] ${isUser ? 'order-2' : 'order-1'}`}>
        <div
          className={`rounded-lg px-4 py-3 ${
            isUser
              ? 'bg-blue-600 text-white'
              : 'bg-slate-800 text-slate-100 border border-slate-700'
          }`}
        >
          {isUser ? (
            <div className="whitespace-pre-wrap break-words">{message.content}</div>
          ) : (
            <MarkdownContent content={message.content} />
          )}
        </div>
        <div className={`text-xs text-slate-400 mt-1 ${isUser ? 'text-right' : 'text-left'}`}>
          {timeString}
        </div>
      </div>
      {!isUser && (
        <div className="order-2 mr-3 flex-shrink-0">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center">
            <span className="text-white text-sm font-bold">AI</span>
          </div>
        </div>
      )}
      {isUser && (
        <div className="order-1 ml-3 flex-shrink-0">
          <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center">
            <span className="text-slate-300 text-sm font-bold">You</span>
          </div>
        </div>
      )}
    </div>
  )
}

