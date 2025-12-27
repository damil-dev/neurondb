'use client'

import Link from 'next/link'
import {
  SparklesIcon,
  ChatBubbleLeftRightIcon,
  DatabaseIcon,
  DocumentTextIcon
} from '@/components/Icons'

const features = [
  {
    name: 'MCP Console',
    description: 'Interact with MCP servers, test tools, and view responses in real-time',
    href: '/mcp',
    icon: ChatBubbleLeftRightIcon,
    color: 'bg-blue-500',
  },
  {
    name: 'NeuronDB Console',
    description: 'Search collections, view indexes, and manage vector data',
    href: '/neurondb',
    icon: DatabaseIcon,
    color: 'bg-green-500',
  },
  {
    name: 'Logs & Inspector',
    description: 'View request logs and inspect tool calls with detailed analytics',
    href: '/logs',
    icon: DocumentTextIcon,
    color: 'bg-purple-500',
  },
]

export default function Home() {
  return (
    <div className="min-h-full bg-slate-950">
      {/* Hero Section */}
      <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border-b border-slate-800">
        <div className="max-w-7xl mx-auto px-8 py-20">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
              <SparklesIcon className="w-7 h-7 text-white" />
            </div>
            <h1 className="text-5xl font-bold text-slate-100 tracking-tight">NeuronDesktop</h1>
          </div>
          <p className="text-xl text-slate-300 mb-10 max-w-3xl leading-relaxed">
            Unified web interface for MCP servers, NeuronDB, and NeuronAgent. 
            Everything you need to build and manage AI applications in one place.
          </p>
          <div className="flex gap-4">
            <Link
              href="/mcp"
              className="px-8 py-4 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition-all duration-200 shadow-lg shadow-blue-500/20 hover:shadow-xl hover:shadow-blue-500/30 hover:-translate-y-0.5"
            >
              Get Started
            </Link>
            <Link
              href="/settings"
              className="px-8 py-4 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg font-semibold transition-all duration-200 border border-slate-700 hover:border-slate-600"
            >
              Configure
            </Link>
          </div>
        </div>
      </div>

      {/* Features Grid */}
      <div className="max-w-7xl mx-auto px-8 py-16">
        <h2 className="text-3xl font-bold text-slate-100 mb-12">Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {features.map((feature) => {
            const Icon = feature.icon
            return (
              <Link
                key={feature.name}
                href={feature.href}
                className="group bg-slate-900 border border-slate-800 rounded-xl p-6 hover:border-slate-700 transition-all duration-200 hover:-translate-y-1 hover:shadow-2xl hover:shadow-slate-900/50"
              >
                <div className={`w-14 h-14 ${feature.color} rounded-xl flex items-center justify-center mb-5 group-hover:scale-110 transition-transform shadow-lg`}>
                  <Icon className="w-7 h-7 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-slate-100 mb-2">{feature.name}</h3>
                <p className="text-slate-400 leading-relaxed">{feature.description}</p>
              </Link>
            )
          })}
        </div>
      </div>
    </div>
  )
}
