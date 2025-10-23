# Windows PowerShell 版本的 HuggingFace 模型评估脚本
# 对应 scripts/evaluate_hf_llm.sh

# 设置环境变量
$OUT_DIR = "./outputs"
$EMB_DIR = "./outputs"
$DATA_FILE_PATH = "./data/locomo10.json"
$QA_OUTPUT_FILE = "locomo10_qa.json"
$OBS_OUTPUT_FILE = "locomo10_observation.json"
$SESS_SUMM_OUTPUT_FILE = "locomo10_session_summary.json"
$PROMPT_DIR = "./prompt_examples"

# 可选：设置 API Keys（如果需要）
# $env:OPENAI_API_KEY = "your_key_here"
# $env:GOOGLE_API_KEY = "your_key_here"
# $env:ANTHROPIC_API_KEY = "your_key_here"
# $env:HF_TOKEN = "your_token_here"

# 确保输出目录存在
if (-not (Test-Path $OUT_DIR)) {
    New-Item -ItemType Directory -Path $OUT_DIR | Out-Null
    Write-Host "创建输出目录: $OUT_DIR" -ForegroundColor Green
}

# 激活 conda 环境
Write-Host "激活 locomo 环境..." -ForegroundColor Cyan
conda activate locomo

# 要评估的模型列表
$MODELS = @(
    "mistral-instruct-7b-32k-v2"
    # "llama3-chat-70b"  # 取消注释以使用其他模型
    # "gemma-7b-it"
)

# 循环运行每个模型
foreach ($MODEL in $MODELS) {
    Write-Host "`n========================================" -ForegroundColor Yellow
    Write-Host "正在评估模型: $MODEL" -ForegroundColor Yellow
    Write-Host "========================================`n" -ForegroundColor Yellow
    
    python task_eval/evaluate_qa.py `
        --data-file $DATA_FILE_PATH `
        --out-file "$OUT_DIR/$QA_OUTPUT_FILE" `
        --model $MODEL `
        --use-4bit `
        --batch-size 1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ 模型 $MODEL 评估完成！" -ForegroundColor Green
    } else {
        Write-Host "`n❌ 模型 $MODEL 评估失败！退出代码: $LASTEXITCODE" -ForegroundColor Red
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "所有模型评估完成！" -ForegroundColor Cyan
Write-Host "结果保存在: $OUT_DIR/$QA_OUTPUT_FILE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
