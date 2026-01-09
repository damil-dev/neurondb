'use client'

import { useState, useCallback } from 'react'
import GridLayout, { Layout } from 'react-grid-layout'
import { useTheme } from '@/contexts/ThemeContext'

interface GridItem {
  i: string
  x: number
  y: number
  w: number
  h: number
  minW?: number
  minH?: number
  maxW?: number
  maxH?: number
  static?: boolean
}

interface DraggableGridProps {
  items: Array<{
    id: string
    content: React.ReactNode
    defaultLayout: GridItem
  }>
  cols?: number
  rowHeight?: number
  onLayoutChange?: (layout: Layout[]) => void
  className?: string
}

export default function DraggableGrid({
  items,
  cols = 12,
  rowHeight = 100,
  onLayoutChange,
  className = '',
}: DraggableGridProps) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'

  const [layout, setLayout] = useState<Layout[]>(
    items.map((item) => ({
      i: item.id,
      x: item.defaultLayout.x,
      y: item.defaultLayout.y,
      w: item.defaultLayout.w,
      h: item.defaultLayout.h,
      minW: item.defaultLayout.minW,
      minH: item.defaultLayout.minH,
      maxW: item.defaultLayout.maxW,
      maxH: item.defaultLayout.maxH,
      static: item.defaultLayout.static,
    }))
  )

  const handleLayoutChange = useCallback(
    (newLayout: Layout[]) => {
      setLayout(newLayout)
      onLayoutChange?.(newLayout)
    },
    [onLayoutChange]
  )

  return (
    <div className={className}>
      <GridLayout
        className={`layout ${isDark ? 'dark' : ''}`}
        layout={layout}
        onLayoutChange={handleLayoutChange}
        cols={cols}
        rowHeight={rowHeight}
        width={1200}
        isDraggable={true}
        isResizable={true}
        draggableHandle=".drag-handle"
        style={{
          backgroundColor: 'transparent',
        }}
      >
        {items.map((item) => (
          <div
            key={item.id}
            className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-4 shadow-sm"
          >
            {item.content}
          </div>
        ))}
      </GridLayout>
      <style jsx global>{`
        .react-grid-layout {
          position: relative;
        }
        .react-grid-item {
          transition: all 200ms ease;
          transition-property: left, top, width, height;
        }
        .react-grid-item.cssTransforms {
          transition-property: transform, width, height;
        }
        .react-grid-item.resizing {
          transition: none;
          z-index: 1;
          will-change: width, height;
        }
        .react-grid-item.react-draggable-dragging {
          transition: none;
          z-index: 3;
          will-change: transform;
        }
        .react-grid-item.dropping {
          visibility: hidden;
        }
        .react-grid-item.react-grid-placeholder {
          background: ${isDark ? 'rgba(139, 92, 246, 0.2)' : 'rgba(139, 92, 246, 0.1)'};
          opacity: 0.2;
          transition-duration: 100ms;
          z-index: 2;
          -webkit-user-select: none;
          -moz-user-select: none;
          -ms-user-select: none;
          -o-user-select: none;
          user-select: none;
        }
        .react-grid-item > .react-resizable-handle {
          position: absolute;
          width: 20px;
          height: 20px;
        }
        .react-grid-item > .react-resizable-handle::after {
          content: "";
          position: absolute;
          right: 3px;
          bottom: 3px;
          width: 5px;
          height: 5px;
          border-right: 2px solid ${isDark ? 'rgb(148, 163, 184)' : 'rgb(71, 85, 105)'};
          border-bottom: 2px solid ${isDark ? 'rgb(148, 163, 184)' : 'rgb(71, 85, 105)'};
        }
      `}</style>
    </div>
  )
}

