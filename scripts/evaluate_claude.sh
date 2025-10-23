#!/bin/bash
# Claude model evaluation script for Ubuntu/Linux

# sets necessary environment variables and activates conda environment
source scripts/env.sh

# Evaluate Claude-Sonnet
python3 task_eval/evaluate_qa.py \
    --data-file $DATA_FILE_PATH --out-file $OUT_DIR/$QA_OUTPUT_FILE \
    --model claude-sonnet --batch-size 10
