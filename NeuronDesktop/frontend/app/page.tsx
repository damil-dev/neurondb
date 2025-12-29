'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { factoryAPI } from '@/lib/api'
import {
  SparklesIcon,
  ChatBubbleLeftRightIcon,
  DatabaseIcon,
  DocumentTextIcon,
  CpuChipIcon,
  ActivityIcon,
  WrenchScrewdriverIcon,
} from '@/components/Icons'

const features = [
  {
    name: 'MCP Console',
    description: 'Interact with MCP servers, test tools, and view responses in real-time',
    href: '/mcp',
    icon: ChatBubbleLeftRightIcon,
    color: 'from-blue-500 to-cyan-500',
  },
  {
    name: 'NeuronDB Console',
    description: 'Search collections, view indexes, and manage vector data',
    href: '/neurondb',
    icon: DatabaseIcon,
    color: 'from-green-500 to-emerald-500',
  },
  {
    name: 'Agents',
    description: 'Manage and interact with AI agents',
    href: '/agents',
    icon: CpuChipIcon,
    color: 'from-purple-500 to-pink-500',
  },
  {
    name: 'Factory Console',
    description: 'Installation and system monitoring',
    href: '/setup',
    icon: WrenchScrewdriverIcon,
    color: 'from-orange-500 to-red-500',
  },
  {
    name: 'Monitoring',
    description: 'System metrics and performance monitoring',
    href: '/monitoring',
    icon: ActivityIcon,
    color: 'from-indigo-500 to-purple-500',
  },
  {
    name: 'Logs & Inspector',
    description: 'View request logs and inspect tool calls with detailed analytics',
    href: '/logs',
    icon: DocumentTextIcon,
    color: 'from-yellow-500 to-orange-500',
  },
]

export default function Home() {
  const router = useRouter()
  const [checking, setChecking] = useState(true)

  useEffect(() => {
    // Check if setup is complete
    factoryAPI.getSetupState()
      .then((response) => {
        if (!response.data.setup_complete) {
          router.push('/setup')
        } else {
          setChecking(false)
        }
      })
      .catch(() => {
        // If API call fails, show homepage (might be first run)
        setChecking(false)
      })
  }, [router])

  if (checking) {
    return (
      <div className="min-h-full bg-transparent flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mx-auto mb-4"></div>
          <p className="text-slate-700">Loading...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-full bg-transparent">
      {/* Hero Section */}
      <div className="bg-slate-100 border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-8 py-24">
          <div className="flex items-center justify-center gap-4 mb-8">
            <div className="w-20 h-20 bg-gradient-to-br from-purple-500 via-purple-600 to-indigo-600 rounded-2xl flex items-center justify-center shadow-2xl shadow-purple-500/30">
              <SparklesIcon className="w-10 h-10 text-white" />
            </div>
            <div>
              <h1 className="text-6xl font-bold bg-gradient-to-r from-purple-600 via-purple-500 to-indigo-600 bg-clip-text text-transparent">
                NeuronDesktop
              </h1>
              <p className="text-lg text-slate-700 mt-2">NeuronDB PostgreSQL AI Factory</p>
            </div>
          </div>
          <p className="text-xl text-slate-700 mb-10 max-w-3xl mx-auto text-center leading-relaxed">
            Everything you need to build and manage AI applications in one place.
            Integrated tools for MCP servers, NeuronDB, and NeuronAgent.
          </p>
        </div>
      </div>

      {/* Features Grid */}
      <div className="max-w-7xl mx-auto px-8 py-20">
        <h2 className="text-3xl font-bold text-slate-800 mb-12 text-center">Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature) => {
            const Icon = feature.icon
            return (
              <Link
                key={feature.name}
                href={feature.href}
                className="group relative bg-white border border-slate-200 rounded-xl p-6 hover:border-slate-300 transition-all duration-300 hover:-translate-y-1 hover:shadow-xl hover:shadow-slate-200/50"
              >
                <div className={`w-14 h-14 bg-gradient-to-br ${feature.color} rounded-xl flex items-center justify-center mb-5 group-hover:scale-110 transition-transform shadow-lg`}>
                  <Icon className="w-7 h-7 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-slate-800 mb-2">{feature.name}</h3>
                <p className="text-slate-700 leading-relaxed">{feature.description}</p>
                <div className="mt-4 flex items-center text-purple-600 text-sm font-medium opacity-0 group-hover:opacity-100 transition-opacity">
                  Explore →
                </div>
              </Link>
            )
          })}
        </div>
      </div>
    </div>
  )
}
