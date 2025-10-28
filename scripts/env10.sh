#!/bin/bash
# Main environment configuration for LoCoMo project
# Uses locomo dataset (full version)

# Source common initialization
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ============================================
# Dataset and Output Configuration
# ============================================
# Save generated outputs to this location
OUT_DIR=./outputs

# Save embeddings to this location
EMB_DIR=./outputs

# Path to LoCoMo data file
DATA_FILE_PATH=./data/locomo10.json

# Filenames for different outputs
QA_OUTPUT_FILE=locomo10_qa.json
OBS_OUTPUT_FILE=locomo10_observation.json
SESS_SUMM_OUTPUT_FILE=locomo10_session_summary.json

# Path to folder containing prompts and in-context examples
PROMPT_DIR=./prompt_examples
