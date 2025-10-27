#!/bin/bash
# RAG-based HuggingFace local model evaluation script for Ubuntu/Linux (TEST VERSION)

# sets necessary environment variables and activates conda environment
# Use BASH_SOURCE to get the directory of this script, then source the env file
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/env1.sh"

# Override output directories for RAG-based HF LLM results
OUT_DIR=./outputs/rag_hf_llm
EMB_DIR=./outputs/rag_hf_llm/embeddings

# Evaluate local HuggingFace model under different RAG conditions
# You can change the MODEL variable to test different models

# Model selection: mistral-instruct-7b-32k-v2, gemma-7b-it, llama2-chat, llama3-chat-70b, etc.
MODEL="mistral-instruct-7b-32k-v2"

# dialog as database
for TOP_K in 5 10 25 50; do
    python3 task_eval/evaluate_qa.py \
        --data-file $DATA_FILE_PATH --out-file $OUT_DIR/$QA_OUTPUT_FILE \
        --model $MODEL --batch-size 1 --use-rag --retriever dragon --top-k $TOP_K \
        --emb-dir $EMB_DIR --rag-mode dialog --use-4bit
done

# # observation as database
# for TOP_K in 5 10 25 50; do
#     python3 task_eval/evaluate_qa.py \
#         --data-file $DATA_FILE_PATH --out-file $OUT_DIR/$QA_OUTPUT_FILE \
#         --model $MODEL --batch-size 1 --use-rag --retriever dragon --top-k $TOP_K \
#         --emb-dir $EMB_DIR --rag-mode observation --use-4bit
# done

# # summary as database
# for TOP_K in 2 5 10; do
#     python3 task_eval/evaluate_qa.py \
#         --data-file $DATA_FILE_PATH --out-file $OUT_DIR/$QA_OUTPUT_FILE \
#         --model $MODEL --batch-size 1 --use-rag --retriever dragon --top-k $TOP_K \
#         --emb-dir $EMB_DIR --rag-mode summary --use-4bit
# done
