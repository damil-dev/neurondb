'use client'

import { Profile } from '@/lib/api'
import { ChevronDownIcon } from '@/components/Icons'
import { useState } from 'react'

interface ProfileSelectorProps {
  profiles: Profile[]
  selectedProfile: string
  onSelect: (profileId: string) => void
}

export default function ProfileSelector({ profiles, selectedProfile, onSelect }: ProfileSelectorProps) {
  const [isOpen, setIsOpen] = useState(false)
  const selected = profiles.find(p => p.id === selectedProfile)

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg hover:border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-slate-900 transition-all duration-200"
      >
        <span className="text-sm font-medium text-slate-200">
          {selected ? selected.name : 'Select Profile...'}
        </span>
        <ChevronDownIcon className={`w-4 h-4 text-slate-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>
      
      {isOpen && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setIsOpen(false)}></div>
          <div className="absolute z-20 w-full mt-1 bg-slate-800 border border-slate-700 rounded-lg shadow-2xl max-h-60 overflow-auto">
            {profiles.map((profile) => (
              <button
                key={profile.id}
                onClick={() => {
                  onSelect(profile.id)
                  setIsOpen(false)
                }}
                className={`w-full text-left px-4 py-2 hover:bg-slate-700 transition-colors ${
                  profile.id === selectedProfile ? 'bg-blue-600 text-white' : 'text-slate-200'
                }`}
              >
                <div className="font-medium">{profile.name}</div>
                <div className={`text-xs ${profile.id === selectedProfile ? 'text-blue-200' : 'text-slate-400'}`}>
                  {profile.neurondb_dsn.split('@')[1] || profile.neurondb_dsn}
                </div>
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

