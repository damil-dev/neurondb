'use client'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl'
  className?: string
  variant?: 'default' | 'purple' | 'gradient'
}

export default function LoadingSpinner({ 
  size = 'md', 
  className = '',
  variant = 'gradient'
}: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
    xl: 'w-16 h-16',
  }

  const variantClasses = {
    default: 'border-gray-200 dark:border-gray-700 border-t-blue-600 dark:border-t-blue-400',
    purple: 'border-purple-200 dark:border-purple-900 border-t-purple-600 dark:border-t-purple-400',
    gradient: 'border-transparent border-t-purple-600 dark:border-t-purple-400',
  }

  return (
    <div className={`flex items-center justify-center ${className}`}>
      <div className="relative" role="status" aria-label="Loading">
        {/* Outer ring */}
        <div
          className={`${sizeClasses[size]} ${variantClasses[variant]} border-4 rounded-full animate-spin`}
        >
          <span className="sr-only">Loading...</span>
        </div>
        {/* Inner glow effect for gradient variant */}
        {variant === 'gradient' && (
          <div
            className={`absolute inset-0 ${sizeClasses[size]} bg-gradient-to-r from-purple-500 to-indigo-500 rounded-full opacity-20 blur-sm animate-pulse`}
          />
        )}
      </div>
    </div>
  )
}






