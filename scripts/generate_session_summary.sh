#!/bin/bash
# Session summary generation script for Ubuntu/Linux

# sets necessary environment variables and activates conda environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/env10.sh"

# gets session summaries using gpt-3.5-turbo and extract DRAGON embeddings for RAG database
python task_eval/get_session_summary.py --data-file $DATA_FILE_PATH --out-file $OUT_DIR/$SESS_SUMM_OUTPUT_FILE \
    --prompt-dir $PROMPT_DIR --emb-dir $EMB_DIR --use-date