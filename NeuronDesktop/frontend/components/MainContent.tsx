'use client'

import { useSidebar } from '@/contexts/SidebarContext'
import { useEffect, useState } from 'react'

export default function MainContent({ children }: { children: React.ReactNode }) {
  const { isOpen } = useSidebar()
  const [isDesktop, setIsDesktop] = useState(false)

  useEffect(() => {
    const checkDesktop = () => {
      setIsDesktop(window.innerWidth >= 1024)
    }
    
    checkDesktop()
    window.addEventListener('resize', checkDesktop)
    return () => window.removeEventListener('resize', checkDesktop)
  }, [])

  return (
    <main
      className={`
        flex-1 overflow-auto bg-slate-50 transition-all duration-300 ease-in-out
        ${isOpen && isDesktop ? 'lg:ml-64' : 'lg:ml-0'}
      `}
    >
      <div className="max-w-7xl mx-auto w-full px-6">
        {children}
      </div>
    </main>
  )
}

