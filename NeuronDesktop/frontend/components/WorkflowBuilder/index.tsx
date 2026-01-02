'use client'

import { useState, useCallback } from 'react'
import {
  PlusIcon,
  TrashIcon,
  CheckCircleIcon,
  XCircleIcon,
} from '@/components/Icons'

interface WorkflowNode {
  id: string
  type: 'tool' | 'agent' | 'conditional' | 'approval' | 'http'
  name: string
  config: Record<string, any>
  depends_on?: string[]
}

interface WorkflowBuilderProps {
  workflow?: { steps: WorkflowNode[] }
  onChange?: (workflow: { steps: WorkflowNode[] }) => void
  onSave?: (workflow: { steps: WorkflowNode[] }) => void
  readOnly?: boolean
}

export default function WorkflowBuilder({
  workflow,
  onChange,
  onSave,
  readOnly = false,
}: WorkflowBuilderProps) {
  const [steps, setSteps] = useState<WorkflowNode[]>(
    workflow?.steps || []
  )
  const [selectedStep, setSelectedStep] = useState<string | null>(null)
  const [editingStep, setEditingStep] = useState<WorkflowNode | null>(null)

  const handleAddStep = useCallback(() => {
    const newStep: WorkflowNode = {
      id: `step_${Date.now()}`,
      type: 'tool',
      name: 'New Step',
      config: {},
    }
    const newSteps = [...steps, newStep]
    setSteps(newSteps)
    if (onChange) onChange({ steps: newSteps })
    setSelectedStep(newStep.id)
    setEditingStep(newStep)
  }, [steps, onChange])

  const handleDeleteStep = useCallback(
    (stepId: string) => {
      const newSteps = steps.filter((s) => s.id !== stepId)
      // Remove dependencies on deleted step
      newSteps.forEach((step) => {
        if (step.depends_on) {
          step.depends_on = step.depends_on.filter((dep) => dep !== stepId)
        }
      })
      setSteps(newSteps)
      if (onChange) onChange({ steps: newSteps })
      if (selectedStep === stepId) {
        setSelectedStep(null)
        setEditingStep(null)
      }
    },
    [steps, onChange, selectedStep]
  )

  const handleUpdateStep = useCallback(
    (step: WorkflowNode) => {
      const newSteps = steps.map((s) => (s.id === step.id ? step : s))
      setSteps(newSteps)
      if (onChange) onChange({ steps: newSteps })
      setEditingStep(null)
    },
    [steps, onChange]
  )

  const handleSelectStep = useCallback((step: WorkflowNode) => {
    if (readOnly) return
    setSelectedStep(step.id)
    setEditingStep(step)
  }, [readOnly])

  const handleSave = useCallback(() => {
    if (onSave) {
      onSave({ steps })
    }
  }, [steps, onSave])

  const validateWorkflow = useCallback(() => {
    const errors: string[] = []
    const stepIds = new Set(steps.map((s) => s.id))

    steps.forEach((step) => {
      if (!step.name || step.name.trim() === '') {
        errors.push(`Step ${step.id} has no name`)
      }

      if (step.depends_on) {
        step.depends_on.forEach((dep) => {
          if (!stepIds.has(dep)) {
            errors.push(`Step ${step.id} depends on non-existent step: ${dep}`)
          }
        })
      }
    })

    // Check for cycles (simple check)
    const visited = new Set<string>()
    const recStack = new Set<string>()

    const hasCycle = (stepId: string): boolean => {
      if (recStack.has(stepId)) return true
      if (visited.has(stepId)) return false

      visited.add(stepId)
      recStack.add(stepId)

      const step = steps.find((s) => s.id === stepId)
      if (step?.depends_on) {
        for (const dep of step.depends_on) {
          if (hasCycle(dep)) return true
        }
      }

      recStack.delete(stepId)
      return false
    }

    for (const step of steps) {
      if (hasCycle(step.id)) {
        errors.push(`Workflow contains a cycle involving step ${step.id}`)
        break
      }
    }

    return errors
  }, [steps])

  const validationErrors = validateWorkflow()
  const isValid = validationErrors.length === 0

  return (
    <div className="h-full flex flex-col">
      {/* Toolbar */}
      {!readOnly && (
        <div className="border-b border-gray-200 dark:border-slate-700 p-4 flex items-center justify-between bg-white dark:bg-slate-900">
          <div className="flex items-center gap-3">
            <button
              onClick={handleAddStep}
              className="btn btn-primary flex items-center gap-2"
            >
              <PlusIcon className="w-5 h-5" />
              Add Step
            </button>
            {onSave && (
              <button
                onClick={handleSave}
                disabled={!isValid}
                className="btn btn-secondary flex items-center gap-2"
              >
                <CheckCircleIcon className="w-5 h-5" />
                Save Workflow
              </button>
            )}
          </div>
          <div className="flex items-center gap-2">
            {isValid ? (
              <span className="text-sm text-green-600 dark:text-green-400 flex items-center gap-1">
                <CheckCircleIcon className="w-4 h-4" />
                Valid
              </span>
            ) : (
              <span className="text-sm text-red-600 dark:text-red-400 flex items-center gap-1">
                <XCircleIcon className="w-4 h-4" />
                {validationErrors.length} error(s)
              </span>
            )}
          </div>
        </div>
      )}

      {/* Validation Errors */}
      {validationErrors.length > 0 && (
        <div className="bg-red-50 dark:bg-red-900/20 border-b border-red-200 dark:border-red-800 p-3">
          <div className="text-sm text-red-800 dark:text-red-200">
            <strong>Validation Errors:</strong>
            <ul className="list-disc list-inside mt-1 space-y-1">
              {validationErrors.map((error, idx) => (
                <li key={idx}>{error}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Workflow Visualization */}
      <div className="flex-1 overflow-auto p-6 bg-gray-50 dark:bg-slate-800">
        {steps.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <p className="text-gray-500 dark:text-slate-400 mb-4">
                No workflow steps yet
              </p>
              {!readOnly && (
                <button onClick={handleAddStep} className="btn btn-primary">
                  Add First Step
                </button>
              )}
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {steps.map((step, index) => {
              const dependencies = step.depends_on || []
              const isSelected = selectedStep === step.id

              return (
                <div key={step.id} className="relative">
                  {/* Dependency Lines */}
                  {dependencies.length > 0 && (
                    <div className="absolute left-0 top-0 bottom-0 w-8 border-l-2 border-dashed border-blue-400" />
                  )}

                  {/* Step Card */}
                  <div
                    onClick={() => handleSelectStep(step)}
                    className={`ml-8 card cursor-pointer transition-all ${
                      isSelected
                        ? 'ring-2 ring-blue-500 dark:ring-blue-400 shadow-lg'
                        : 'hover:shadow-md'
                    } ${readOnly ? 'cursor-default' : ''}`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <span className="flex items-center justify-center w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 font-semibold">
                            {index + 1}
                          </span>
                          <div>
                            <h3 className="font-semibold text-gray-900 dark:text-slate-100">
                              {step.name}
                            </h3>
                            <span className="text-xs px-2 py-1 bg-gray-100 dark:bg-slate-700 text-gray-600 dark:text-slate-300 rounded">
                              {step.type}
                            </span>
                          </div>
                        </div>

                        {dependencies.length > 0 && (
                          <div className="mt-2 text-xs text-gray-500 dark:text-slate-400">
                            Depends on: {dependencies.join(', ')}
                          </div>
                        )}

                        {step.config && Object.keys(step.config).length > 0 && (
                          <div className="mt-2 text-xs text-gray-600 dark:text-slate-400 bg-gray-50 dark:bg-slate-800 p-2 rounded">
                            {JSON.stringify(step.config, null, 2).substring(0, 100)}
                            {JSON.stringify(step.config, null, 2).length > 100 && '...'}
                          </div>
                        )}
                      </div>

                      {!readOnly && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleDeleteStep(step.id)
                          }}
                          className="p-2 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded"
                        >
                          <TrashIcon className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Step Editor */}
      {editingStep && !readOnly && (
        <StepEditor
          step={editingStep}
          allSteps={steps}
          onSave={handleUpdateStep}
          onCancel={() => {
            setEditingStep(null)
            setSelectedStep(null)
          }}
        />
      )}
    </div>
  )
}

interface StepEditorProps {
  step: WorkflowNode
  allSteps: WorkflowNode[]
  onSave: (step: WorkflowNode) => void
  onCancel: () => void
}

function StepEditor({ step, allSteps, onSave, onCancel }: StepEditorProps) {
  const [editedStep, setEditedStep] = useState<WorkflowNode>({ ...step })

  const handleSave = () => {
    onSave(editedStep)
  }

  const availableSteps = allSteps.filter((s) => s.id !== step.id)

  return (
    <div className="border-t border-gray-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-4">
      <div className="max-w-4xl mx-auto space-y-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-slate-100">
            Edit Step
          </h3>
          <div className="flex gap-2">
            <button onClick={onCancel} className="btn btn-secondary">
              Cancel
            </button>
            <button onClick={handleSave} className="btn btn-primary">
              Save
            </button>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
              Step Name *
            </label>
            <input
              type="text"
              value={editedStep.name}
              onChange={(e) =>
                setEditedStep({ ...editedStep, name: e.target.value })
              }
              className="input"
              placeholder="Step name"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
              Step Type *
            </label>
            <select
              value={editedStep.type}
              onChange={(e) =>
                setEditedStep({
                  ...editedStep,
                  type: e.target.value as WorkflowNode['type'],
                })
              }
              className="input"
            >
              <option value="tool">Tool</option>
              <option value="agent">Agent</option>
              <option value="conditional">Conditional</option>
              <option value="approval">Approval</option>
              <option value="http">HTTP</option>
            </select>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
            Dependencies (select steps this depends on)
          </label>
          <div className="space-y-2 border rounded-lg p-3 max-h-40 overflow-y-auto">
            {availableSteps.length === 0 ? (
              <p className="text-sm text-gray-500 dark:text-slate-400">
                No other steps available
              </p>
            ) : (
              availableSteps.map((s) => {
                const dependsOn = editedStep.depends_on || []
                const isChecked = dependsOn.includes(s.id)
                return (
                  <label
                    key={s.id}
                    className="flex items-center gap-2 cursor-pointer hover:bg-gray-50 dark:hover:bg-slate-800 p-2 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={isChecked}
                      onChange={(e) => {
                        const newDependsOn = e.target.checked
                          ? [...dependsOn, s.id]
                          : dependsOn.filter((id) => id !== s.id)
                        setEditedStep({
                          ...editedStep,
                          depends_on: newDependsOn,
                        })
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-900 dark:text-slate-100">
                      {s.name} ({s.type})
                    </span>
                  </label>
                )
              })
            )}
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
            Configuration (JSON)
          </label>
          <textarea
            value={JSON.stringify(editedStep.config || {}, null, 2)}
            onChange={(e) => {
              try {
                const config = JSON.parse(e.target.value)
                setEditedStep({ ...editedStep, config })
              } catch {
                // Invalid JSON, keep as is for now
              }
            }}
            className="input font-mono text-sm min-h-[150px]"
            placeholder='{\n  "key": "value"\n}'
          />
        </div>
      </div>
    </div>
  )
}

