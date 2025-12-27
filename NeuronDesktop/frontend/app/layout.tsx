import type { Metadata } from 'next'
import { SidebarProvider } from '@/contexts/SidebarContext'
import Sidebar from '@/components/Sidebar'
import SidebarToggle from '@/components/SidebarToggle'
import MainContent from '@/components/MainContent'
import './globals.css'

export const metadata: Metadata = {
  title: 'NeuronDesktop - Unified AI Platform',
  description: 'Unified interface for MCP servers, NeuronDB, and NeuronAgent',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="bg-[#1a1a1a] text-[#e0e0e0]">
        <SidebarProvider>
          <div className="flex h-screen bg-[#1a1a1a] relative">
            <Sidebar />
            <SidebarToggle />
            <MainContent>
              {children}
            </MainContent>
          </div>
        </SidebarProvider>
      </body>
    </html>
  )
}
