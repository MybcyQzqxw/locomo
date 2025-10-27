#!/bin/bash
# HuggingFace model evaluation script for Ubuntu/Linux

# sets necessary environment variables and activates conda environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/env.sh"


# run models
for MODEL in mistral-instruct-7b-32k-v2; do
    python3 task_eval/evaluate_qa.py \
        --data-file $DATA_FILE_PATH --out-file $OUT_DIR/$QA_OUTPUT_FILE \
        --model $MODEL --use-4bit --batch-size 1
done
