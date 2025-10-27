# Windows PowerShell script for HuggingFace model evaluation
# Equivalent to scripts/evaluate_hf_llm.sh

# 激活 conda 环境（确保使用正确的 Python 解释器）
# 注意：需要先运行 conda init powershell
Write-Host "Activating conda environment: locomo" -ForegroundColor Cyan
conda activate locomo

# Set environment variables
$OUT_DIR = "./outputs"
$EMB_DIR = "./outputs"
$DATA_FILE_PATH = "./data/locomo1.json"
$QA_OUTPUT_FILE = "locomo1_qa.json"
$OBS_OUTPUT_FILE = "locomo1_observation.json"
$SESS_SUMM_OUTPUT_FILE = "locomo1_session_summary.json"
$PROMPT_DIR = "./prompt_examples"

# Optional: Set API Keys if needed
# $env:OPENAI_API_KEY = "your_key_here"
# $env:GOOGLE_API_KEY = "your_key_here"
# $env:ANTHROPIC_API_KEY = "your_key_here"
# $env:HF_TOKEN = "your_token_here"

# Create output directory if it does not exist
if (-not (Test-Path $OUT_DIR)) {
    New-Item -ItemType Directory -Path $OUT_DIR | Out-Null
    Write-Host "Created output directory: $OUT_DIR" -ForegroundColor Green
}

# List of models to evaluate
$MODELS = @(
    "mistral-instruct-7b-32k-v2"
    # "llama3-chat-70b"  # Uncomment to use other models
    # "gemma-7b-it"
)

# Loop through each model
foreach ($MODEL in $MODELS) {
    Write-Host "`n========================================" -ForegroundColor Yellow
    Write-Host "Evaluating model: $MODEL" -ForegroundColor Yellow
    Write-Host "========================================`n" -ForegroundColor Yellow
    
    python task_eval/evaluate_qa.py `
        --data-file $DATA_FILE_PATH `
        --out-file "$OUT_DIR/$QA_OUTPUT_FILE" `
        --model $MODEL `
        --use-4bit `
        --batch-size 1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nModel $MODEL evaluation completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "`nModel $MODEL evaluation failed! Exit code: $LASTEXITCODE" -ForegroundColor Red
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "All model evaluations completed!" -ForegroundColor Cyan
Write-Host "Results saved in: $OUT_DIR/$QA_OUTPUT_FILE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
