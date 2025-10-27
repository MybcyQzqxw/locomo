#!/bin/bash
# Common setup script for all LoCoMo evaluation scripts
# This file contains shared initialization logic that should be sourced by all environment configuration files

# ============================================
# Auto-detect project root and change to it
# Works whether script is run from project root or scripts/ directory
# ============================================
# Get the directory of the calling script
if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
else
    SCRIPT_DIR="$(pwd)"
fi

# Check if we're in the scripts directory
if [[ "$SCRIPT_DIR" == */scripts ]]; then
    # If in scripts directory, move to parent (project root)
    PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
else
    # Otherwise assume we're already in project root
    PROJECT_ROOT="$SCRIPT_DIR"
fi

# Change to project root directory
cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"

# ============================================
# Conda Environment Setup
# ============================================
# Conda environment name
# Change this if your environment name is not 'locomo'
CONDA_ENV_NAME=locomo

# Activate conda environment
# Note: You need to run 'conda init bash' first
if [ -n "$CONDA_ENV_NAME" ]; then
    echo "Activating conda environment: $CONDA_ENV_NAME"
    # Initialize conda if not already initialized
    eval "$(conda shell.bash hook)"
    conda activate $CONDA_ENV_NAME
    
    if [ $? -eq 0 ]; then
        echo "Successfully activated environment: $CONDA_ENV_NAME"
        echo "Python path: $(which python)"
        echo "Python version: $(python --version)"
    else
        echo "Warning: Unable to activate conda environment $CONDA_ENV_NAME"
        echo "Using system default Python"
    fi
fi
