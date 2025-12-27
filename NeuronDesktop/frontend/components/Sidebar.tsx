'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useSidebar } from '@/contexts/SidebarContext'
import {
  HomeIcon,
  ChatBubbleLeftRightIcon,
  DatabaseIcon,
  DocumentTextIcon,
  Cog6ToothIcon,
  ActivityIcon,
  CpuChipIcon,
  SparklesIcon,
  Bars3Icon,
  XMarkIcon
} from '@/components/Icons'

const navigation = [
  { name: 'Home', href: '/', icon: HomeIcon },
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
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={close}
        />
      )}
      
      {/* Sidebar */}
      <div className={`
        fixed left-0 top-0 h-screen w-64 bg-[#252525] border-r border-[#333333] flex flex-col shadow-2xl z-50 
        transform transition-transform duration-300 ease-in-out
        ${isOpen 
          ? 'translate-x-0' 
          : '-translate-x-full'
        }
        lg:static lg:z-auto lg:transform-none lg:transition-none
        ${isOpen ? 'lg:flex' : 'lg:hidden'}
      `}>
        <div className="p-6 border-b border-[#333333]">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-gradient-to-br from-[#8b5cf6] to-[#6366f1] rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">N</span>
              </div>
              <h1 className="text-xl font-bold text-[#e0e0e0]">NeuronDesktop</h1>
            </div>
            <button
              onClick={close}
              className="lg:hidden p-2 text-[#999999] hover:text-[#e0e0e0] hover:bg-[#2d2d2d] rounded-lg transition-colors"
              aria-label="Close sidebar"
            >
              <XMarkIcon className="w-5 h-5" />
            </button>
          </div>
          <p className="text-xs text-[#999999] mt-1 font-medium">Unified AI Platform</p>
        </div>
        
        <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
          {navigation.map((item) => {
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
                  flex items-center px-4 py-3 rounded-lg transition-all duration-200 group
                  ${isActive 
                    ? 'bg-[#8b5cf6] text-white font-semibold shadow-lg shadow-[#8b5cf6]/20' 
                    : 'text-[#c8c8c8] hover:bg-[#2d2d2d] hover:text-white'
                  }
                `}
              >
                <Icon className={`w-5 h-5 mr-3 ${isActive ? 'text-white' : 'text-[#999999] group-hover:text-[#e0e0e0]'}`} />
                <span>{item.name}</span>
              </Link>
            )
          })}
        </nav>
        
        <div className="p-4 border-t border-[#333333] bg-[#1a1a1a]/50">
          <div className="text-xs text-[#999999]">
            <div className="font-medium text-[#c8c8c8]">Version 1.0.0</div>
            <div className="mt-1">© 2024 NeuronDB</div>
          </div>
        </div>
      </div>
    </>
  )
}

