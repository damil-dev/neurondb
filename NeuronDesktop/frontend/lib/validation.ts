// Form validation utilities with real-time feedback
import React from 'react'

export interface ValidationRule {
  validate: (value: any, formData?: any) => boolean | string
  message: string
}

export interface FieldValidation {
  field: string
  rules: ValidationRule[]
  required?: boolean
}

export interface ValidationResult {
  isValid: boolean
  errors: Record<string, string>
}

// Common validation rules
export const rules = {
  required: (message = 'This field is required'): ValidationRule => ({
    validate: (value) => {
      if (value === null || value === undefined || value === '') {
        return message
      }
      if (typeof value === 'string' && value.trim() === '') {
        return message
      }
      if (Array.isArray(value) && value.length === 0) {
        return message
      }
      return true
    },
    message,
  }),

  minLength: (min: number, message?: string): ValidationRule => ({
    validate: (value) => {
      if (!value) return true // Let required rule handle empty values
      const length = typeof value === 'string' ? value.length : Array.isArray(value) ? value.length : 0
      if (length < min) {
        return message || `Must be at least ${min} characters`
      }
      return true
    },
    message: message || `Must be at least ${min} characters`,
  }),

  maxLength: (max: number, message?: string): ValidationRule => ({
    validate: (value) => {
      if (!value) return true
      const length = typeof value === 'string' ? value.length : Array.isArray(value) ? value.length : 0
      if (length > max) {
        return message || `Must be no more than ${max} characters`
      }
      return true
    },
    message: message || `Must be no more than ${max} characters`,
  }),

  email: (message = 'Invalid email address'): ValidationRule => ({
    validate: (value) => {
      if (!value) return true
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
      if (!emailRegex.test(value)) {
        return message
      }
      return true
    },
    message,
  }),

  url: (message = 'Invalid URL'): ValidationRule => ({
    validate: (value) => {
      if (!value) return true
      try {
        new URL(value)
        return true
      } catch {
        return message
      }
    },
    message,
  }),

  pattern: (regex: RegExp, message: string): ValidationRule => ({
    validate: (value) => {
      if (!value) return true
      if (!regex.test(value)) {
        return message
      }
      return true
    },
    message,
  }),

  custom: (validator: (value: any, formData?: any) => boolean | string, message: string): ValidationRule => ({
    validate: validator,
    message,
  }),
}

// Validate a single field
export function validateField(
  field: string,
  value: any,
  fieldValidation: FieldValidation,
  formData?: any
): string | null {
  // Check required first
  if (fieldValidation.required !== false && (value === null || value === undefined || value === '')) {
    const requiredRule = fieldValidation.rules.find(r => r.message.includes('required'))
    if (requiredRule) {
      const result = requiredRule.validate(value, formData)
      if (result !== true) {
        return typeof result === 'string' ? result : requiredRule.message
      }
    } else {
      return 'This field is required'
    }
  }

  // Run all validation rules
  for (const rule of fieldValidation.rules) {
    const result = rule.validate(value, formData)
    if (result !== true) {
      return typeof result === 'string' ? result : rule.message
    }
  }

  return null
}

// Validate entire form
export function validateForm(
  formData: Record<string, any>,
  validations: FieldValidation[]
): ValidationResult {
  const errors: Record<string, string> = {}

  for (const fieldValidation of validations) {
    const value = formData[fieldValidation.field]
    const error = validateField(fieldValidation.field, value, fieldValidation, formData)
    if (error) {
      errors[fieldValidation.field] = error
    }
  }

  return {
    isValid: Object.keys(errors).length === 0,
    errors,
  }
}

// React hook for form validation
export function useFormValidation<T extends Record<string, any>>(
  initialData: T,
  validations: FieldValidation[]
) {
  const [formData, setFormData] = React.useState<T>(initialData)
  const [errors, setErrors] = React.useState<Record<string, string>>({})
  const [touched, setTouched] = React.useState<Record<string, boolean>>({})

  const validate = React.useCallback(() => {
    const result = validateForm(formData, validations)
    setErrors(result.errors)
    return result.isValid
  }, [formData, validations])

  const validateFieldValue = React.useCallback(
    (field: string) => {
      const fieldValidation = validations.find(v => v.field === field)
      if (!fieldValidation) return

      const error = validateField(field, formData[field], fieldValidation, formData)
      setErrors(prev => {
        if (error) {
          return { ...prev, [field]: error }
        } else {
          const { [field]: _, ...rest } = prev
          return rest
        }
      })
    },
    [formData, validations]
  )

  const setFieldValue = React.useCallback((field: string, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }))
    // Validate on change if field has been touched
    if (touched[field]) {
      setTimeout(() => validateFieldValue(field), 0)
    }
  }, [touched, validateFieldValue])

  const setFieldTouched = React.useCallback((field: string) => {
    setTouched(prev => ({ ...prev, [field]: true }))
    validateFieldValue(field)
  }, [validateFieldValue])

  const reset = React.useCallback(() => {
    setFormData(initialData)
    setErrors({})
    setTouched({})
  }, [initialData])

  return {
    formData,
    errors,
    touched,
    setFieldValue,
    setFieldTouched,
    validate,
    validateField: validateFieldValue,
    reset,
    isValid: Object.keys(errors).length === 0 && Object.keys(touched).length > 0,
  }
}

