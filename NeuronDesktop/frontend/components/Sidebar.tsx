'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useSidebar } from '@/contexts/SidebarContext'
import {
  HomeIcon,
  ChatBubbleLeftRightIcon,
  ChatIcon,
  DatabaseIcon,
  DocumentTextIcon,
  Cog6ToothIcon,
  ActivityIcon,
  CpuChipIcon,
  SparklesIcon,
  Bars3Icon,
  XMarkIcon,
  WrenchScrewdriverIcon,
  ChartBarIcon
} from '@/components/Icons'

const navigation = [
  { name: 'Home', href: '/', icon: HomeIcon },
  { name: 'Dashboard', href: '/dashboard', icon: ChartBarIcon },
  { name: 'Factory', href: '/setup', icon: WrenchScrewdriverIcon },
  { name: 'Chat', href: '/chat', icon: ChatIcon },
  { name: 'MCP Console', href: '/mcp', icon: ChatBubbleLeftRightIcon },
  { name: 'NeuronDB', href: '/neurondb', icon: DatabaseIcon },
  { name: 'Agents', href: '/agents', icon: CpuChipIcon },
  { name: 'Models', href: '/models', icon: SparklesIcon },
  { name: 'Monitoring', href: '/monitoring', icon: ActivityIcon },
  { name: 'Logs', href: '/logs', icon: DocumentTextIcon },
  { name: 'Settings', href: '/settings', icon: Cog6ToothIcon },
]

export default function Sidebar() {
  const pathname = usePathname()
  const { isOpen, close } = useSidebar()

  return (
    <>
      {/* Overlay for mobile */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 lg:hidden animate-fade-in"
          onClick={close}
        />
      )}
      
      {/* Sidebar */}
      <div className={`
        fixed left-0 top-10 h-[calc(100vh-2.5rem)] w-64 sm:w-72 lg:w-64 xl:w-72
        bg-slate-900/95 dark:bg-slate-950/95 backdrop-blur-xl border-r border-slate-700/50 
        flex flex-col shadow-2xl z-50 
        transform transition-transform duration-300 ease-out
        ${isOpen 
          ? 'translate-x-0' 
          : '-translate-x-full'
        }
        lg:static lg:top-0 lg:h-screen lg:z-auto lg:transform-none lg:transition-none
        ${isOpen ? 'lg:flex' : 'lg:hidden'}
      `}>
        <div className="p-4 sm:p-5 lg:p-6 border-b border-slate-700/50 bg-gradient-to-r from-purple-500/10 to-indigo-500/10">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2 sm:gap-3 group min-w-0 flex-1">
              <div className="w-9 h-9 sm:w-10 sm:h-10 lg:w-10 lg:h-10 flex-shrink-0 bg-gradient-to-br from-purple-500 via-purple-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/30 group-hover:shadow-purple-500/50 transition-all duration-300 group-hover:scale-110">
                <span className="text-white font-bold text-xs sm:text-sm">N</span>
              </div>
              <div className="min-w-0 flex-1">
                <h1 className="text-lg sm:text-xl font-bold bg-gradient-to-r from-purple-400 to-indigo-400 bg-clip-text text-transparent truncate">NeuronDesktop</h1>
                <p className="text-[10px] sm:text-xs text-slate-400 mt-0.5 font-medium truncate">NeuronDB PostgreSQL AI Factory</p>
              </div>
            </div>
            <button
              onClick={close}
              className="lg:hidden p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition-all duration-200 hover:scale-110"
              aria-label="Close sidebar"
            >
              <XMarkIcon className="w-5 h-5" />
            </button>
          </div>
        </div>
        
        <nav className="flex-1 p-3 sm:p-4 lg:p-4 space-y-1 overflow-y-auto">
          {navigation.map((item, index) => {
            const isActive = pathname === item.href
            const Icon = item.icon
            return (
              <Link
                key={item.name}
                href={item.href}
                onClick={() => {
                  // Close sidebar on mobile when navigating
                  if (window.innerWidth < 1024) {
                    close()
                  }
                }}
                className={`
                  flex items-center px-3 sm:px-4 py-2.5 sm:py-3 rounded-xl transition-all duration-300 group relative
                  animate-fade-in-up
                  ${isActive 
                    ? 'bg-gradient-to-r from-purple-600 to-indigo-600 text-white font-semibold shadow-lg shadow-purple-500/30 scale-[1.02]' 
                    : 'text-slate-300 hover:bg-slate-800/50 hover:text-white hover:translate-x-1'
                  }
                `}
                style={{ animationDelay: `${index * 30}ms` }}
              >
                {isActive && (
                  <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-white rounded-r-full"></div>
                )}
                <Icon className={`w-4 h-4 sm:w-5 sm:h-5 mr-2 sm:mr-3 flex-shrink-0 transition-transform duration-300 ${isActive ? 'text-white scale-110' : 'text-slate-400 group-hover:text-slate-200 group-hover:scale-110'}`} />
                <span className="relative z-10 text-sm sm:text-base truncate flex-1">{item.name}</span>
                {!isActive && (
                  <div className="ml-auto opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                    <div className="w-1.5 h-1.5 bg-purple-500 rounded-full"></div>
                  </div>
                )}
              </Link>
            )
          })}
        </nav>
        
        <div className="p-4 border-t border-slate-700/50 bg-gradient-to-r from-slate-900/50 to-slate-800/50 backdrop-blur-sm">
          <div className="text-xs text-slate-400">
            <div className="font-semibold text-slate-300 mb-1 flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              Version 2.0.0
            </div>
            <div className="text-slate-500">© 2024 NeuronDB</div>
          </div>
        </div>
      </div>
    </>
  )
}

