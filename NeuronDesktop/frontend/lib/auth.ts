// Authentication utilities for NeuronDesktop

const API_KEY_STORAGE_KEY = 'neurondesk_api_key'

export function getAPIKey(): string | null {
  if (typeof window === 'undefined') return null
  return localStorage.getItem(API_KEY_STORAGE_KEY)
}

export function setAPIKey(key: string): void {
  if (typeof window === 'undefined') return
  localStorage.setItem(API_KEY_STORAGE_KEY, key)
}

export function removeAPIKey(): void {
  if (typeof window === 'undefined') return
  localStorage.removeItem(API_KEY_STORAGE_KEY)
}

export function getAuthHeaders(): Record<string, string> {
  const apiKey = getAPIKey()
  if (!apiKey) return {}
  
  return {
    'Authorization': `Bearer ${apiKey}`,
  }
}

