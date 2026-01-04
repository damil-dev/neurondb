'use client'

interface SkeletonProps {
  className?: string
  lines?: number
  width?: string
  height?: string
}

export function Skeleton({ className = '', lines = 1, width, height }: SkeletonProps) {
  const baseClasses = 'animate-pulse bg-gray-200 dark:bg-gray-700 rounded'

  if (lines > 1) {
    return (
      <div className={`space-y-2 ${className}`}>
        {Array.from({ length: lines }).map((_, i) => (
          <div
            key={i}
            className={`${baseClasses} ${i === lines - 1 && width ? '' : 'w-full'}`}
            style={{
              width: i === lines - 1 ? width || '75%' : '100%',
              height: height || '1rem',
            }}
          />
        ))}
      </div>
    )
  }

  return (
    <div
      className={`${baseClasses} ${className}`}
      style={{
        width: width || '100%',
        height: height || '1rem',
      }}
    />
  )
}

export function SkeletonCard() {
  return (
    <div className="card animate-pulse">
      <div className="space-y-4">
        <Skeleton height="1.5rem" width="60%" />
        <Skeleton lines={3} />
        <div className="flex gap-2">
          <Skeleton width="80px" height="2rem" />
          <Skeleton width="80px" height="2rem" />
        </div>
      </div>
    </div>
  )
}

export function SkeletonList({ count = 3 }: { count?: number }) {
  return (
    <div className="space-y-4">
      {Array.from({ length: count }).map((_, i) => (
        <SkeletonCard key={i} />
      ))}
    </div>
  )
}

export function SkeletonTable({ rows = 5, cols = 4 }: { rows?: number; cols?: number }) {
  return (
    <div className="card animate-pulse">
      <div className="space-y-3">
        {/* Header */}
        <div className="flex gap-4">
          {Array.from({ length: cols }).map((_, i) => (
            <Skeleton key={i} width="25%" height="1.5rem" />
          ))}
        </div>
        {/* Rows */}
        {Array.from({ length: rows }).map((_, rowIdx) => (
          <div key={rowIdx} className="flex gap-4">
            {Array.from({ length: cols }).map((_, colIdx) => (
              <Skeleton key={colIdx} width="25%" height="1rem" />
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

