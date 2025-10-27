#!/bin/bash
# Environment setup script for LoCoMo project

# Conda 环境名称
# 如果你的环境名不是 locomo，请修改这里
CONDA_ENV_NAME=locomo

# 激活 conda 环境
# 注意：首次使用需要先运行 conda init bash
if [ -n "$CONDA_ENV_NAME" ]; then
    echo "正在激活 conda 环境: $CONDA_ENV_NAME"
    # 初始化 conda（如果尚未初始化）
    eval "$(conda shell.bash hook)"
    conda activate $CONDA_ENV_NAME
    
    if [ $? -eq 0 ]; then
        echo "成功激活环境: $CONDA_ENV_NAME"
        echo "Python 路径: $(which python)"
        echo "Python 版本: $(python --version)"
    else
        echo "警告: 无法激活 conda 环境 $CONDA_ENV_NAME"
        echo "将使用系统默认 Python"
    fi
fi

# save generated outputs to this location
OUT_DIR=./outputs

# save embeddings to this location
EMB_DIR=./outputs

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
