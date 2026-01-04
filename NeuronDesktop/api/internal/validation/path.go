package validation

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

/* ValidateFilePath validates a file path for safety */
func ValidateFilePath(path, fieldName string) error {
	if path == "" {
		return fmt.Errorf("%s cannot be empty", fieldName)
	}
	
	path = strings.TrimSpace(path)
	
	/* Check for path traversal attempts */
	if strings.Contains(path, "..") {
		return fmt.Errorf("%s contains path traversal attempt: %s", fieldName, path)
	}
	
	/* Check for absolute paths (if you want to restrict to relative) */
	/* This is optional - remove if absolute paths are allowed */
	if filepath.IsAbs(path) {
		/* Allow absolute paths but validate they exist and are readable */
		if _, err := os.Stat(path); err != nil {
			return fmt.Errorf("%s points to non-existent file: %w", fieldName, err)
		}
	}
	
	/* Check for null bytes */
	if strings.Contains(path, "\x00") {
		return fmt.Errorf("%s contains null byte", fieldName)
	}
	
	return nil
}

/* ValidateFilePathRequired validates a file path and ensures it's not empty */
func ValidateFilePathRequired(path, fieldName string) error {
	if path == "" {
		return fmt.Errorf("%s is required and cannot be empty", fieldName)
	}
	return ValidateFilePath(path, fieldName)
}

/* ValidateExecutablePath validates that a path points to an executable file */
func ValidateExecutablePath(path, fieldName string) error {
	if err := ValidateFilePath(path, fieldName); err != nil {
		return err
	}
	
	/* Check if file exists and is executable */
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("%s points to non-existent file: %w", fieldName, err)
	}
	
	if info.IsDir() {
		return fmt.Errorf("%s points to a directory, not a file: %s", fieldName, path)
	}
	
	/* Check if file is executable (Unix) */
	if info.Mode().Perm()&0111 == 0 {
		return fmt.Errorf("%s is not executable: %s", fieldName, path)
	}
	
	return nil
}

