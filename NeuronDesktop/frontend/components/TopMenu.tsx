'use client'

import { useState } from 'react'
import { useRouter, usePathname } from 'next/navigation'
import { removeAuthToken } from '@/lib/auth'
import Link from 'next/link'
import {
  HomeIcon,
  Cog6ToothIcon,
  ArrowPathIcon,
  InformationCircleIcon,
  SparklesIcon,
  ChatBubbleLeftRightIcon,
  DatabaseIcon,
  CpuChipIcon,
  ActivityIcon,
} from './Icons'

export default function TopMenu() {
  const router = useRouter()
  const pathname = usePathname()

  const navigation = [
    { name: 'Home', href: '/', icon: HomeIcon },
    { name: 'MCP Console', href: '/mcp', icon: ChatBubbleLeftRightIcon },
    { name: 'NeuronDB', href: '/neurondb', icon: DatabaseIcon },
    { name: 'Agents', href: '/agents', icon: CpuChipIcon },
    { name: 'Monitoring', href: '/monitoring', icon: ActivityIcon },
    { name: 'Settings', href: '/settings', icon: Cog6ToothIcon },
  ]

  const handleLogout = () => {
    removeAuthToken()
    router.push('/login')
  }

  const isActive = (href: string) => {
    if (href === '/') {
      return pathname === '/'
    }
    return pathname?.startsWith(href)
  }

  return (
    <nav className="h-12 sm:h-14 bg-white/80 dark:bg-slate-950/80 backdrop-blur-xl border-b border-slate-200/50 dark:border-slate-800/50 shadow-sm sticky top-0 z-40">
      <div className="w-full px-3 sm:px-4 md:px-6 lg:px-8 h-full flex items-center justify-between gap-2 sm:gap-4">
        {/* Logo/Brand */}
        <Link href="/" className="flex items-center gap-2 sm:gap-3 group min-w-0 flex-shrink-0">
          <div className="w-9 h-9 sm:w-10 sm:h-10 lg:w-11 lg:h-11 flex-shrink-0 bg-gradient-to-br from-purple-500 via-purple-600 to-indigo-600 rounded-lg sm:rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/30 group-hover:shadow-purple-500/50 transition-all duration-300 group-hover:scale-110 group-hover:rotate-3">
            <SparklesIcon className="w-5 h-5 sm:w-6 sm:h-6 text-white group-hover:animate-pulse" />
          </div>
          <div className="flex flex-col min-w-0 hidden xs:flex">
            <span className="text-base sm:text-lg font-bold bg-gradient-to-r from-purple-600 via-purple-500 to-indigo-600 bg-clip-text text-transparent group-hover:from-purple-500 group-hover:to-indigo-500 transition-all duration-300 truncate">
              NeuronDesktop
            </span>
            <span className="text-[9px] sm:text-[10px] text-gray-600 dark:text-slate-400 leading-tight font-medium truncate">NeuronDB PostgreSQL AI Factory</span>
          </div>
        </Link>

        {/* Center Navigation */}
        <div className="hidden md:flex items-center gap-1 bg-slate-50/50 dark:bg-slate-900/50 rounded-xl p-1 backdrop-blur-sm overflow-x-auto flex-1 justify-center mx-2">
          {navigation.map((item) => {
            const Icon = item.icon
            const active = isActive(item.href)
            return (
              <Link
                key={item.name}
                href={item.href}
                className={`
                  relative px-3 lg:px-4 py-1.5 lg:py-2 rounded-lg text-xs lg:text-sm font-medium transition-all duration-300
                  flex items-center gap-1.5 lg:gap-2 group whitespace-nowrap flex-shrink-0
                  ${
                    active
                      ? 'text-white bg-gradient-to-r from-purple-600 to-indigo-600 shadow-lg shadow-purple-500/30 scale-105'
                      : 'text-gray-700 dark:text-slate-300 hover:text-purple-600 dark:hover:text-purple-400 hover:bg-white/50 dark:hover:bg-slate-800/50'
                  }
                `}
              >
                <Icon className={`w-3.5 h-3.5 lg:w-4 lg:h-4 transition-transform duration-300 flex-shrink-0 ${active ? 'text-white scale-110' : 'text-gray-600 dark:text-slate-400 group-hover:scale-110'}`} />
                <span className="relative z-10 hidden lg:inline">{item.name}</span>
                {active && (
                  <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-indigo-600 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 blur-sm -z-0"></div>
                )}
              </Link>
            )
          })}
        </div>

        {/* Right Actions */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => window.location.reload()}
            className="p-2.5 text-gray-700 dark:text-slate-400 hover:text-purple-600 dark:hover:text-purple-400 hover:bg-purple-50 dark:hover:bg-purple-500/10 rounded-lg transition-all duration-300 hover:scale-110 hover:rotate-180 group"
            title="Refresh"
          >
            <ArrowPathIcon className="w-5 h-5 group-hover:animate-spin" />
          </button>
          <div className="relative group">
            <button
              className="p-2.5 text-gray-700 dark:text-slate-400 hover:text-purple-600 dark:hover:text-purple-400 hover:bg-purple-50 dark:hover:bg-purple-500/10 rounded-lg transition-all duration-300 hover:scale-110"
              title="More options"
            >
              <InformationCircleIcon className="w-5 h-5" />
            </button>
            <div className="absolute right-0 top-full mt-2 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-300 z-50 animate-scale-in">
              <div className="bg-white/95 dark:bg-slate-800/95 backdrop-blur-xl border border-slate-200/50 dark:border-slate-700/50 rounded-xl shadow-2xl py-2 min-w-[180px] overflow-hidden">
                <button
                  onClick={() => alert('NeuronDesktop - NeuronDB PostgreSQL AI Factory\nVersion 2.0.0')}
                  className="w-full text-left px-4 py-2.5 hover:bg-gradient-to-r hover:from-purple-50 hover:to-indigo-50 dark:hover:from-purple-500/10 dark:hover:to-indigo-500/10 text-gray-700 dark:text-slate-200 text-sm font-medium transition-all duration-200 hover:translate-x-1"
                >
                  About
                </button>
                <div className="border-t border-slate-200 dark:border-slate-700 my-1"></div>
                <button
                  onClick={handleLogout}
                  className="w-full text-left px-4 py-2.5 hover:bg-red-50 dark:hover:bg-red-500/10 text-red-600 dark:text-red-400 text-sm font-medium transition-all duration-200 hover:translate-x-1"
                >
                  Logout
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </nav>
  )
}
