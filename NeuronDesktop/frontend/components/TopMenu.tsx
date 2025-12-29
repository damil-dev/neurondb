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
    <nav className="h-12 bg-slate-100 dark:bg-slate-950 border-b border-slate-200 dark:border-slate-800">
      <div className="w-full px-4 h-full flex items-center justify-between">
        {/* Logo/Brand */}
        <Link href="/" className="flex items-center gap-3 group">
          <div className="w-10 h-10 bg-gradient-to-br from-purple-500 via-purple-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/20 group-hover:shadow-purple-500/40 transition-all duration-300 group-hover:scale-105">
            <SparklesIcon className="w-6 h-6 text-white" />
          </div>
          <div className="flex flex-col">
            <span className="text-lg font-bold bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent">
              NeuronDesktop
            </span>
            <span className="text-[10px] text-gray-700 dark:text-slate-400 leading-tight">NeuronDB PostgreSQL AI Factory</span>
          </div>
        </Link>

        {/* Center Navigation */}
        <div className="flex items-center gap-1">
          {navigation.map((item) => {
            const Icon = item.icon
            const active = isActive(item.href)
            return (
              <Link
                key={item.name}
                href={item.href}
                className={`
                  relative px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200
                  flex items-center gap-2
                  ${
                    active
                      ? 'text-purple-600 dark:text-purple-400 bg-purple-50 dark:bg-purple-500/10 shadow-sm'
                      : 'text-gray-700 dark:text-slate-300 hover:text-gray-900 dark:hover:text-white hover:bg-slate-50 dark:hover:bg-slate-800/50'
                  }
                `}
              >
                <Icon className={`w-4 h-4 ${active ? 'text-purple-600 dark:text-purple-400' : 'text-gray-700 dark:text-slate-400'}`} />
                <span>{item.name}</span>
                {active && (
                  <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-1 h-1 bg-purple-600 dark:bg-purple-400 rounded-full" />
                )}
              </Link>
            )
          })}
        </div>

        {/* Right Actions */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => window.location.reload()}
            className="p-2 text-gray-700 dark:text-slate-400 hover:text-gray-900 dark:hover:text-white hover:bg-slate-50 dark:hover:bg-slate-800/50 rounded-lg transition-colors"
            title="Refresh"
          >
            <ArrowPathIcon className="w-5 h-5" />
          </button>
          <div className="relative group">
            <button
              className="p-2 text-gray-700 dark:text-slate-400 hover:text-gray-900 dark:hover:text-white hover:bg-slate-50 dark:hover:bg-slate-800/50 rounded-lg transition-colors"
              title="More options"
            >
              <InformationCircleIcon className="w-5 h-5" />
            </button>
            <div className="absolute right-0 top-full mt-2 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50">
              <div className="bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg shadow-xl py-1 min-w-[160px]">
                <button
                  onClick={() => alert('NeuronDesktop - NeuronDB PostgreSQL AI Factory\nVersion 1.0.0')}
                  className="w-full text-left px-4 py-2 hover:bg-slate-50 dark:hover:bg-slate-700 text-gray-700 dark:text-slate-200 text-sm"
                >
                  About
                </button>
                <div className="border-t border-slate-200 dark:border-slate-700 my-1"></div>
                <button
                  onClick={handleLogout}
                  className="w-full text-left px-4 py-2 hover:bg-slate-50 dark:hover:bg-slate-700 text-red-600 dark:text-red-400 text-sm"
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
