'use client'

import { ThemeProvider as ThemeProviderImpl } from '@/contexts/ThemeContext'

export default function ThemeProvider({ children }: { children: React.ReactNode }) {
  return <ThemeProviderImpl>{children}</ThemeProviderImpl>
}





