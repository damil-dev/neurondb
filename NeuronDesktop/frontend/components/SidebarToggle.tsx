'use client'

import { useSidebar } from '@/contexts/SidebarContext'
import { Bars3Icon } from '@/components/Icons'
import { useEffect, useState } from 'react'

export default function SidebarToggle() {
  const { isOpen, toggle } = useSidebar()
  const [isDesktop, setIsDesktop] = useState(false)

  useEffect(() => {
    const checkDesktop = () => {
      setIsDesktop(window.innerWidth >= 1024)
    }
    
    checkDesktop()
    window.addEventListener('resize', checkDesktop)
    return () => window.removeEventListener('resize', checkDesktop)
  }, [])

  // Only show toggle button on mobile, or on desktop when sidebar is closed
  if (isDesktop && isOpen) {
    return null
  }

  return (
    <button
      onClick={toggle}
      className={`
        fixed top-4 z-50 p-2 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white rounded-lg shadow-lg 
        transition-all duration-300 border border-slate-700
        ${isOpen && !isDesktop ? 'left-[272px]' : 'left-4'}
        lg:hidden
      `}
      aria-label={isOpen ? 'Hide sidebar' : 'Show sidebar'}
    >
      <Bars3Icon className="w-6 h-6" />
    </button>
  )
}
