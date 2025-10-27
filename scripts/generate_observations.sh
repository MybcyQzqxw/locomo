#!/bin/bash
# Observation generation script for Ubuntu/Linux

# sets necessary environment variables and activates conda environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/env.sh"

# gets observations using gpt-3.5-turbo and extract DRAGON embeddings for RAG database
python task_eval/get_facts.py --data-file $DATA_FILE_PATH --out-file $OUT_DIR/$OBS_OUTPUT_FILE \
    --prompt-dir $PROMPT_DIR --emb-dir $EMB_DIR --use-date --overwrite