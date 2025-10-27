#!/bin/bash
# Test environment configuration for LoCoMo project
# Uses locomo1_5qa dataset (5 QA samples for quick testing)

# Source common initialization
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ============================================
# Dataset and Output Configuration
# ============================================
# Save generated outputs to this location
OUT_DIR=./outputs

# Save embeddings to this location
EMB_DIR=./outputs

# Path to LoCoMo data file (5 QA samples for quick testing)
DATA_FILE_PATH=./data/locomo1_5qa.json

# Filenames for different outputs
QA_OUTPUT_FILE=locomo1_5qa_qa.json
OBS_OUTPUT_FILE=locomo1_5qa_observation.json
SESS_SUMM_OUTPUT_FILE=locomo1_5qa_session_summary.json

# Path to folder containing prompts and in-context examples
PROMPT_DIR=./prompt_examples

# ============================================
# API Keys Configuration
# ============================================

# path to LoCoMo data file
DATA_FILE_PATH=./data/locomo1.json

# filenames for different outputs
QA_OUTPUT_FILE=locomo1_qa.json
OBS_OUTPUT_FILE=locomo1_observation.json
SESS_SUMM_OUTPUT_FILE=locomo1_session_summary.json

# path to folder containing prompts and in-context examples
PROMPT_DIR=./prompt_examples

# OpenAI API Key
export OPENAI_API_KEY=

# Google API Key
export GOOGLE_API_KEY=

# Anthropic API Key
export ANTHROPIC_API_KEY=

# HuggingFace Token
export HF_TOKEN=
