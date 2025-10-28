#!/bin/bash
# GPT evaluation script for Ubuntu/Linux

# sets necessary environment variables and activates conda environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/env10.sh"

# Evaluate gpt-4-turbo
python3 task_eval/evaluate_qa.py \
    --data-file $DATA_FILE_PATH --out-file $OUT_DIR/$QA_OUTPUT_FILE \
    --model gpt-4-turbo --batch-size 20

# Evaluate gpt-3.5-turbo under different context lengths
for MODEL in gpt-3.5-turbo-4k gpt-3.5-turbo-8k gpt-3.5-turbo-12k gpt-3.5-turbo-16k; do
    python3 task_eval/evaluate_qa.py \
        --data-file $DATA_FILE_PATH --out-file $OUT_DIR/$QA_OUTPUT_FILE \
        --model $MODEL --batch-size 10
done
