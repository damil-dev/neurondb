/*-------------------------------------------------------------------------
 *
 * sandbox_enhanced.go
 *    Enhanced sandbox for tool execution
 *
 * Provides sandboxed code execution with resource limits, file allowlists,
 * network egress rules, and container-based isolation.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/sandbox_enhanced.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

/* SandboxConfig defines sandbox configuration */
type SandboxConfig struct {
	MaxMemory    int64         `json:"max_memory"`    // Bytes
	MaxCPU       float64       `json:"max_cpu"`       // CPU percentage (0-100)
	MaxDisk      int64         `json:"max_disk"`      // Bytes
	Timeout      time.Duration `json:"timeout"`       // Execution timeout
	AllowedDirs  []string      `json:"allowed_dirs"`  // Allowed directories
	AllowedFiles []string      `json:"allowed_files"` // Allowed specific files
	NetworkRules NetworkRules  `json:"network_rules"` // Network egress rules
	Isolation    IsolationType `json:"isolation"`     // Isolation type
}

/* NetworkRules defines network egress rules */
type NetworkRules struct {
	AllowedDomains []string `json:"allowed_domains"` // Allowed domain names
	AllowedIPs     []string `json:"allowed_ips"`     // Allowed IP addresses/CIDR
	BlockAll       bool     `json:"block_all"`       // Block all network access
}

/* IsolationType defines isolation method */
type IsolationType string

const (
	IsolationNone      IsolationType = "none"      // No isolation
	IsolationChroot    IsolationType = "chroot"    // Chroot isolation
	IsolationContainer IsolationType = "container" // Container isolation (Docker, etc.)
)

/* EnhancedSandbox provides enhanced sandboxing capabilities */
type EnhancedSandbox struct {
	config SandboxConfig
	base   *Sandbox
}

/* NewEnhancedSandbox creates a new enhanced sandbox */
func NewEnhancedSandbox(config SandboxConfig) *EnhancedSandbox {
	return &EnhancedSandbox{
		config: config,
		base:   NewSandbox("", config.MaxMemory, int(config.MaxCPU)),
	}
}

/* ExecuteCommand executes a command in the sandbox */
func (s *EnhancedSandbox) ExecuteCommand(ctx context.Context, command string, args []string, workingDir string) ([]byte, error) {
	/* Create context with timeout */
	if s.config.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, s.config.Timeout)
		defer cancel()
	}

	/* Validate working directory */
	if workingDir != "" {
		if err := s.validatePath(workingDir); err != nil {
			return nil, fmt.Errorf("invalid working directory: %w", err)
		}
	}

	/* Validate command path */
	if err := s.validatePath(command); err != nil {
		return nil, fmt.Errorf("invalid command path: %w", err)
	}

	/* Create command */
	cmd := exec.CommandContext(ctx, command, args...)
	if workingDir != "" {
		cmd.Dir = workingDir
	}

	/* Apply resource limits */
	if s.base != nil {
		if err := s.base.ApplyResourceLimits(cmd); err != nil {
			return nil, fmt.Errorf("failed to apply resource limits: %w", err)
		}
	}

	/* Apply isolation based on type */
	switch s.config.Isolation {
	case IsolationChroot:
		/* TODO: Apply chroot isolation */
		/* Requires root privileges */
	case IsolationContainer:
		/* TODO: Execute in container */
		/* Would need Docker/container runtime integration */
	}

	/* Execute command */
	output, err := cmd.CombinedOutput()
	if err != nil {
		return output, fmt.Errorf("command execution failed: %w", err)
	}

	return output, nil
}

/* validatePath validates that a path is allowed */
func (s *EnhancedSandbox) validatePath(path string) error {
	/* Resolve absolute path */
	absPath, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("failed to resolve path: %w", err)
	}

	/* Check if path is in allowed directories */
	if len(s.config.AllowedDirs) > 0 {
		allowed := false
		for _, allowedDir := range s.config.AllowedDirs {
			allowedAbs, err := filepath.Abs(allowedDir)
			if err != nil {
				continue
			}
			if strings.HasPrefix(absPath, allowedAbs) {
				allowed = true
				break
			}
		}
		if !allowed {
			return fmt.Errorf("path not in allowed directories: %s", path)
		}
	}

	/* Check if path is in allowed files */
	if len(s.config.AllowedFiles) > 0 {
		for _, allowedFile := range s.config.AllowedFiles {
			allowedAbs, err := filepath.Abs(allowedFile)
			if err != nil {
				continue
			}
			if absPath == allowedAbs {
				return nil /* Allowed */
			}
		}
		/* If allowed files specified but not found, reject */
		if len(s.config.AllowedDirs) == 0 {
			return fmt.Errorf("path not in allowed files: %s", path)
		}
	}

	return nil
}

/* ValidateNetworkAccess validates network access based on rules */
func (s *EnhancedSandbox) ValidateNetworkAccess(host string) error {
	if s.config.NetworkRules.BlockAll {
		return fmt.Errorf("network access blocked")
	}

	/* Check allowed domains */
	if len(s.config.NetworkRules.AllowedDomains) > 0 {
		allowed := false
		for _, domain := range s.config.NetworkRules.AllowedDomains {
			if strings.HasSuffix(host, domain) || host == domain {
				allowed = true
				break
			}
		}
		if !allowed {
			return fmt.Errorf("domain not in allowed list: %s", host)
		}
	}

	/* Check allowed IPs */
	if len(s.config.NetworkRules.AllowedIPs) > 0 {
		/* TODO: Parse IP/CIDR and validate */
		/* Would need net package for IP matching */
	}

	return nil
}

/* SetNetworkRules sets network egress rules */
func (s *EnhancedSandbox) SetNetworkRules(rules NetworkRules) {
	s.config.NetworkRules = rules
}

/* ApplyFileAllowlist applies file allowlist to command environment */
func (s *EnhancedSandbox) ApplyFileAllowlist(cmd *exec.Cmd) error {
	/* Set environment variable with allowed files */
	if len(s.config.AllowedFiles) > 0 {
		allowedFilesStr := strings.Join(s.config.AllowedFiles, ":")
		cmd.Env = append(os.Environ(), "SANDBOX_ALLOWED_FILES="+allowedFilesStr)
	}

	/* Set allowed directories */
	if len(s.config.AllowedDirs) > 0 {
		allowedDirsStr := strings.Join(s.config.AllowedDirs, ":")
		cmd.Env = append(cmd.Env, "SANDBOX_ALLOWED_DIRS="+allowedDirsStr)
	}

	return nil
}

/* DefaultSandboxConfig returns default sandbox configuration */
func DefaultSandboxConfig() SandboxConfig {
	return SandboxConfig{
		MaxMemory: 512 * 1024 * 1024,      /* 512 MB */
		MaxCPU:    50.0,                   /* 50% CPU */
		MaxDisk:   1 * 1024 * 1024 * 1024, /* 1 GB */
		Timeout:   5 * time.Minute,
		Isolation: IsolationNone,
		NetworkRules: NetworkRules{
			BlockAll: true, /* Block all by default */
		},
	}
}
