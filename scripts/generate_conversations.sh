#!/bin/bash
# Conversation generation script for Ubuntu/Linux

# sets necessary environment variables and activates conda environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/env.sh"

python3 generative_agents/generate_conversations.py \
    --out-dir ./data/multimodal_dialog/example/ \
    --prompt-dir ./prompt_examples \
    --events --session --summary --num-sessions 3 \
    --persona --blip-caption \
    --num-days 90 --num-events 10 --max-turns-per-session 20 --num-events-per-session 1
