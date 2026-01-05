import { expect, afterEach, vi } from 'vitest'
import { cleanup } from '@testing-library/react'
import * as matchers from '@testing-library/jest-dom/matchers'

// Extend Vitest's expect with jest-dom matchers
expect.extend(matchers)

// Cleanup after each test
afterEach(() => {
  cleanup()
})

// Mock Next.js router
vi.mock('next/navigation', () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
  }),
  usePathname: () => '/',
  useSearchParams: () => new URLSearchParams(),
}))

// Mock API client
vi.mock('@/lib/api', () => ({
  profilesAPI: {
    list: vi.fn(),
    get: vi.fn(),
    create: vi.fn(),
    update: vi.fn(),
    delete: vi.fn(),
  },
  agentAPI: {
    list: vi.fn(),
    create: vi.fn(),
    health: vi.fn(),
  },
  neurondbAPI: {
    search: vi.fn(),
    testConnection: vi.fn(),
    listModels: vi.fn(),
  },
  factoryAPI: {
    getSetupState: vi.fn(),
  },
}))

