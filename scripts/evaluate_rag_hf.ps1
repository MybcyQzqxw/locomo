# Windows PowerShell script for RAG evaluation using local HuggingFace models
# This runs LoCoMo QA with retrieval-augmented generation (RAG) for HF models like Mistral/LLaMA/Gemma.

# Activate conda environment
Write-Host "Activating conda environment: locomo" -ForegroundColor Cyan
conda activate locomo

# Paths and parameters
$OUT_DIR = "./outputs"
$EMB_DIR = "./outputs"           # store/reuse embeddings here
$DATA_FILE_PATH = "./data/locomo1.json"
$QA_OUTPUT_FILE = "locomo1_qa_rag_hf.json"
$RAG_MODE = "dialog"              # options: dialog | observation | summary
$TOP_K = 5
$RETRIEVER = "contriever"        # options: contriever | dpr | dragon | openai

# Choose local HF models here
$MODELS = @(
    "mistral-instruct-7b-32k-v2"
    # add more models if desired, e.g. "mistral-7b-8k", "llama2-chat"
)

# Ensure outputs directory exists
if (-not (Test-Path $OUT_DIR)) {
    New-Item -ItemType Directory -Path $OUT_DIR | Out-Null
    Write-Host "Created output directory: $OUT_DIR" -ForegroundColor Green
}

foreach ($MODEL in $MODELS) {
    Write-Host "`n========================================" -ForegroundColor Yellow
    Write-Host "RAG evaluating HF model: $MODEL" -ForegroundColor Yellow
    Write-Host "RAG mode: $RAG_MODE | top-k: $TOP_K | retriever: $RETRIEVER" -ForegroundColor Yellow
    Write-Host "========================================`n" -ForegroundColor Yellow

    python task_eval/evaluate_qa.py `
        --data-file $DATA_FILE_PATH `
        --out-file "$OUT_DIR/$QA_OUTPUT_FILE" `
        --model $MODEL `
        --use-4bit `
        --use-rag `
        --rag-mode $RAG_MODE `
        --emb-dir $EMB_DIR `
        --top-k $TOP_K `
        --retriever $RETRIEVER `
        --batch-size 1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nModel $MODEL RAG evaluation completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "`nModel $MODEL RAG evaluation failed! Exit code: $LASTEXITCODE" -ForegroundColor Red
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "All RAG evaluations completed!" -ForegroundColor Cyan
Write-Host "Results saved in: $OUT_DIR/$QA_OUTPUT_FILE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
