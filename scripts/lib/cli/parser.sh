#!/usr/bin/env bash
#-------------------------------------------------------------------------
# parser.sh - Argument parsing engine
#-------------------------------------------------------------------------
# Provides a flexible argument parsing system that automatically handles
# --help, --version, and --verbose flags, while allowing scripts to register
# custom options. Self-sufficient - no dependencies.
#
# Copyright (c) 2024-2025, neurondb, Inc.
#-------------------------------------------------------------------------

# Prevent multiple sourcing
[[ -n "${CLI_PARSER_LOADED:-}" ]] && return 0
CLI_PARSER_LOADED=1

#=========================================================================
# UTILITY FUNCTIONS (self-contained)
#=========================================================================

# Get script name from path
_get_script_name() {
    local script_path="${1:-${BASH_SOURCE[1]:-$0}}"
    basename "$script_path" 2>/dev/null || echo "script"
}

# Get product version
_get_version() {
    echo "2.0.0"
}

# Show version information
_show_version() {
    local script_name="${1:-$(_get_script_name)}"
    echo "$script_name version $(_get_version)"
    exit 0
}

# Set verbose mode
_set_verbose() {
    export CLI_VERBOSE="${1:-1}"
}

#=========================================================================
# PARSER STATE
#=========================================================================

declare -A CLI_OPTIONS
declare -a CLI_POSITIONAL_ARGS

#=========================================================================
# PARSER FUNCTIONS
#=========================================================================

# Parse CLI arguments
# Usage: parse_cli_args "$@"
# Automatically handles: --help, --version, --verbose
parse_cli_args() {
    local script_name
    script_name=$(_get_script_name)
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                # Call show_help if defined, otherwise show default
                if declare -f show_help >/dev/null 2>&1; then
                    show_help
                else
                    echo "Usage: $script_name [OPTIONS]"
                    echo "Use --help for more information"
                fi
                exit 0
                ;;
            -V|--version)
                _show_version "$script_name"
                ;;
            -v|--verbose)
                _set_verbose 1
                shift
                ;;
            --)
                # End of options
                shift
                CLI_POSITIONAL_ARGS+=("$@")
                break
                ;;
            -*)
                # Unknown option - let script handle it
                CLI_POSITIONAL_ARGS+=("$1")
                shift
                ;;
            *)
                # Positional argument
                CLI_POSITIONAL_ARGS+=("$1")
                shift
                ;;
        esac
    done
}

# Get remaining positional arguments
# Usage: ARGS=($(get_cli_args))
get_cli_args() {
    printf '%s\n' "${CLI_POSITIONAL_ARGS[@]}"
}

# Check if option was provided
# Usage: if has_cli_option "verbose"; then ...; fi
has_cli_option() {
    local option="$1"
    [[ -n "${CLI_OPTIONS[$option]:-}" ]]
}

# Get option value
# Usage: VALUE=$(get_cli_option "key")
get_cli_option() {
    local option="$1"
    echo "${CLI_OPTIONS[$option]:-}"
}

# Set option value (for custom options)
# Usage: set_cli_option "key" "value"
set_cli_option() {
    local key="$1"
    local value="$2"
    CLI_OPTIONS["$key"]="$value"
}
