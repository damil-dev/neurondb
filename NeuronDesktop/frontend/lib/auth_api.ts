import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8081/api/v1'

// Create a separate axios instance for auth endpoints that don't require tokens
// But still use the same error interceptor
const authApi = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
  withCredentials: true, // Enable cookies for session-based auth
})

// Add error response interceptor for better error messages (same as main api)
authApi.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      const data = error.response.data as any
      if (data?.error) {
        error.message = data.error
      } else if (data?.message) {
        error.message = data.message
      } else if (error.response.status === 401) {
        error.message = 'Authentication failed. Please check your username and password.'
      } else if (error.response.status === 403) {
        error.message = 'Access denied.'
      } else if (error.response.status >= 500) {
        error.message = 'Server error. Please try again later.'
      }
    } else if (error.request) {
      if (error.code === 'ECONNREFUSED' || error.message.includes('ECONNREFUSED')) {
        error.message = 'Cannot connect to server. Please check if the API server is running on ' + API_BASE_URL
      } else if (error.code === 'ETIMEDOUT' || error.code === 'ECONNABORTED') {
        error.message = 'Request timeout. The server took too long to respond.'
      } else {
        error.message = 'Network error. Please check your connection and try again.'
      }
    }
    return Promise.reject(error)
  }
)

export interface LoginRequest {
  username: string
  password: string
}

export interface RegisterRequest {
  username: string
  password: string
  neurondb_dsn?: string
}

export interface AuthResponse {
  token: string
  user_id: string
  username: string
  profile_id?: string // Profile ID if login matched a profile
}

export const authAPI = {
  login: (credentials: LoginRequest) => 
    authApi.post<AuthResponse>('/auth/login', credentials),
  
  register: (credentials: RegisterRequest) => 
    authApi.post<AuthResponse>('/auth/register', credentials),
  
  getCurrentUser: () => 
    authApi.get('/auth/me'),
  
  // OIDC endpoints
  startOIDC: () => 
    authApi.get<{ auth_url: string; state: string }>('/auth/oidc/start'),
  
  refreshToken: () => 
    authApi.post<{ session_id: string; user_id: string }>('/auth/refresh'),
  
  logout: () => 
    authApi.post('/auth/logout'),
}

