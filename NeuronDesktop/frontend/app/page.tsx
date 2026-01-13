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
    // Add timeout to prevent infinite loading
    const timeoutId = setTimeout(() => {
      console.warn('Home: Setup check timed out, showing homepage')
      setChecking(false)
    }, 10000) // 10 second max wait
    
    // Check if setup is complete
    factoryAPI.getSetupState()
      .then((response) => {
        clearTimeout(timeoutId)
        if (!response.data.setup_complete) {
          router.push('/setup')
        } else {
          setChecking(false)
        }
      })
      .catch((error) => {
        clearTimeout(timeoutId)
        // If API call fails, show homepage (might be first run)
        console.warn('Home: Setup check failed, showing homepage', error)
        setChecking(false)
      })
  }, [router])

  if (checking) {
    return (
      <div className="min-h-full bg-transparent flex items-center justify-center">
        <div className="text-center animate-fade-in">
          <div className="relative w-16 h-16 mx-auto mb-4">
            <div className="absolute inset-0 border-4 border-purple-200 dark:border-purple-900 rounded-full"></div>
            <div className="absolute inset-0 border-4 border-transparent border-t-purple-600 rounded-full animate-spin"></div>
          </div>
          <p className="text-slate-700 dark:text-slate-300 font-medium">Loading...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-full bg-transparent">
      {/* Hero Section */}
      <div className="relative bg-gradient-to-br from-purple-50/50 via-indigo-50/50 to-pink-50/50 dark:from-slate-900 dark:via-slate-950 dark:to-slate-900 border-b border-slate-200/50 dark:border-slate-800/50 overflow-hidden">
        {/* Animated background elements */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-40 -right-40 w-64 h-64 sm:w-80 sm:h-80 bg-purple-300/20 dark:bg-purple-500/10 rounded-full blur-3xl animate-float"></div>
          <div className="absolute -bottom-40 -left-40 w-64 h-64 sm:w-80 sm:h-80 bg-indigo-300/20 dark:bg-indigo-500/10 rounded-full blur-3xl animate-float" style={{ animationDelay: '1s' }}></div>
        </div>
        
        <div className="relative w-full max-w-7xl mx-auto px-4 sm:px-6 md:px-8 lg:px-10 xl:px-12 py-12 sm:py-16 md:py-20 lg:py-24">
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 sm:gap-6 mb-6 sm:mb-8 animate-fade-in-up">
            <div className="w-16 h-16 sm:w-20 sm:h-20 md:w-24 md:h-24 bg-gradient-to-br from-purple-500 via-purple-600 to-indigo-600 rounded-xl sm:rounded-2xl flex items-center justify-center shadow-2xl shadow-purple-500/40 animate-pulse-glow group hover:scale-110 transition-transform duration-300 flex-shrink-0">
              <SparklesIcon className="w-8 h-8 sm:w-10 sm:h-10 md:w-12 md:h-12 text-white group-hover:animate-spin" />
            </div>
            <div className="text-center sm:text-left">
              <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold bg-gradient-to-r from-purple-600 via-purple-500 to-indigo-600 bg-clip-text text-transparent animate-gradient">
                NeuronDesktop
              </h1>
              <p className="text-sm sm:text-base md:text-lg text-slate-700 dark:text-slate-300 mt-1 sm:mt-2 font-medium">NeuronDB PostgreSQL AI Factory</p>
            </div>
          </div>
          <p className="text-base sm:text-lg md:text-xl text-slate-700 dark:text-slate-300 mb-6 sm:mb-8 md:mb-10 max-w-3xl mx-auto text-center leading-relaxed animate-fade-in-up px-4" style={{ animationDelay: '0.1s' }}>
            Everything you need to build and manage AI applications in one place.
            Integrated tools for MCP servers, NeuronDB, and NeuronAgent.
          </p>
        </div>
      </div>

      {/* Features Grid */}
      <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 md:px-8 lg:px-10 xl:px-12 py-12 sm:py-16 md:py-20">
        <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold text-slate-800 dark:text-slate-100 mb-3 sm:mb-4 text-center animate-fade-in-up px-4">Features</h2>
        <p className="text-sm sm:text-base text-slate-600 dark:text-slate-400 text-center mb-8 sm:mb-10 md:mb-12 animate-fade-in-up px-4" style={{ animationDelay: '0.1s' }}>
          Powerful tools to supercharge your AI development workflow
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-5 md:gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon
            return (
              <Link
                key={feature.name}
                href={feature.href}
                className="group relative bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border border-slate-200/50 dark:border-slate-700/50 rounded-xl sm:rounded-2xl p-4 sm:p-5 md:p-6 hover:border-purple-300 dark:hover:border-purple-700 transition-all duration-300 hover:-translate-y-1 sm:hover:-translate-y-2 hover:shadow-xl sm:hover:shadow-2xl hover:shadow-purple-500/20 card-interactive animate-fade-in-up overflow-hidden"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                {/* Gradient overlay on hover */}
                <div className={`absolute inset-0 bg-gradient-to-br ${feature.color} opacity-0 group-hover:opacity-5 transition-opacity duration-300`}></div>
                
                {/* Icon */}
                <div className={`relative w-12 h-12 sm:w-14 sm:h-14 md:w-16 md:h-16 bg-gradient-to-br ${feature.color} rounded-lg sm:rounded-xl flex items-center justify-center mb-3 sm:mb-4 md:mb-5 group-hover:scale-110 group-hover:rotate-3 transition-all duration-300 shadow-lg group-hover:shadow-xl`}>
                  <Icon className="w-6 h-6 sm:w-7 sm:h-7 md:w-8 md:h-8 text-white" />
                  <div className="absolute inset-0 bg-white/20 rounded-lg sm:rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                </div>
                
                {/* Content */}
                <h3 className="text-lg sm:text-xl font-bold text-slate-800 dark:text-slate-100 mb-1.5 sm:mb-2 group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors">
                  {feature.name}
                </h3>
                <p className="text-sm sm:text-base text-slate-600 dark:text-slate-400 leading-relaxed mb-3 sm:mb-4">
                  {feature.description}
                </p>
                
                {/* CTA Arrow */}
                <div className="flex items-center text-purple-600 dark:text-purple-400 text-xs sm:text-sm font-semibold opacity-0 group-hover:opacity-100 transition-all duration-300 group-hover:translate-x-1">
                  <span>Explore</span>
                  <span className="ml-2 group-hover:translate-x-1 transition-transform duration-300">→</span>
                </div>
                
                {/* Shine effect */}
                <div className="absolute inset-0 -translate-x-full group-hover:translate-x-full transition-transform duration-1000 bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>
              </Link>
            )
          })}
        </div>
      </div>
    </div>
  )
}
