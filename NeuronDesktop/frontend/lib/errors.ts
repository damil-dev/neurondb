import { AxiosError } from 'axios'
import { toastManager } from './toast'

/**
 * Extracts a user-friendly error message from an error object
 */
export function getErrorMessage(error: unknown): string {
  // Handle Axios errors (API errors)
  if (error instanceof Error && 'response' in error) {
    const axiosError = error as AxiosError<any>
    
    if (axiosError.response) {
      const { status, data } = axiosError.response
      
      // Try to extract error message from response body
      if (data) {
        if (typeof data === 'string') {
          return data
        }
        if (data.error) {
          return data.error
        }
        if (data.message) {
          return data.message
        }
        // Check for validation errors
        if (data.errors && Array.isArray(data.errors)) {
          return data.errors.join(', ')
        }
      }
      
      // Provide helpful messages for HTTP status codes
      switch (status) {
        case 400:
          return 'Invalid request. Please check your input and try again.'
        case 401:
          return 'Authentication failed. Please check your username and password.'
        case 403:
          return 'Access denied. You do not have permission to perform this action.'
        case 404:
          return 'The requested resource was not found.'
        case 409:
          return 'A conflict occurred. This resource may already exist.'
        case 422:
          return 'Validation failed. Please check your input.'
        case 429:
          return 'Too many requests. Please wait a moment and try again.'
        case 500:
          return 'Server error. Please try again later.'
        case 502:
          return 'Bad gateway. The server is temporarily unavailable.'
        case 503:
          return 'Service unavailable. The server is temporarily down.'
        case 504:
          return 'Gateway timeout. The server took too long to respond.'
        default:
          return `Request failed with status ${status}. Please try again.`
      }
    }
    
    // Network errors (no response received)
    if (axiosError.request) {
      if (axiosError.code === 'ECONNREFUSED') {
        return 'Connection refused. Please check if the API server is running.'
      }
      if (axiosError.code === 'ETIMEDOUT') {
        return 'Request timeout. The server took too long to respond.'
      }
      if (axiosError.code === 'ENOTFOUND' || axiosError.code === 'ERR_NAME_NOT_RESOLVED') {
        return 'Could not resolve server address. Please check your network connection.'
      }
      if (axiosError.code === 'ERR_NETWORK') {
        return 'Network error. Please check your internet connection and try again.'
      }
      return 'Network error. Please check your connection and try again.'
    }
    
    // Other Axios errors
    if (axiosError.message) {
      return axiosError.message
    }
  }
  
  // Handle standard Error objects
  if (error instanceof Error) {
    return error.message
  }
  
  // Handle string errors
  if (typeof error === 'string') {
    return error
  }
  
  // Fallback for unknown error types
  return 'An unexpected error occurred. Please try again.'
}

/**
 * Shows an error toast notification
 */
export function showErrorToast(error: unknown, customMessage?: string) {
  const message = customMessage || getErrorMessage(error)
  toastManager.error(message)
}

/**
 * Shows a success toast notification
 */
export function showSuccessToast(message: string) {
  toastManager.success(message)
}

/**
 * Shows a warning toast notification
 */
export function showWarningToast(message: string) {
  toastManager.warning(message)
}

/**
 * Shows an info toast notification
 */
export function showInfoToast(message: string) {
  toastManager.info(message)
}

/**
 * Extracts detailed error information for debugging
 */
export function getErrorDetails(error: unknown): {
  message: string
  code?: string
  status?: number
  details?: any
} {
  const result: {
    message: string
    code?: string
    status?: number
    details?: any
  } = {
    message: getErrorMessage(error),
  }
  
  if (error instanceof Error && 'response' in error) {
    const axiosError = error as AxiosError
    if (axiosError.response) {
      result.status = axiosError.response.status
      result.details = axiosError.response.data
    }
    if (axiosError.code) {
      result.code = axiosError.code
    }
  }
  
  return result
}

