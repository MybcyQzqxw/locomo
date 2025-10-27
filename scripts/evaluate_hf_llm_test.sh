#!/bin/bash
# HuggingFace model evaluation script for Ubuntu/Linux (Pure LLM without RAG)

# sets necessary environment variables and activates conda environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/env1_5qa.sh"

# Override output directory for pure HF LLM results
OUT_DIR=./outputs/hf_llm
# Note: EMB_DIR not needed for non-RAG mode

# run models
for MODEL in mistral-instruct-7b-32k-v2; do
    python3 task_eval/evaluate_qa.py \
        --data-file $DATA_FILE_PATH --out-file $OUT_DIR/$QA_OUTPUT_FILE \
        --model $MODEL --use-4bit --batch-size 1
done
