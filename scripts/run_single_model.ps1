# Quick script to run a single model evaluation
# Usage: .\scripts\run_single_model.ps1 -Model "mistral-instruct-7b-32k-v2"

param(
    [string]$Model = "mistral-instruct-7b-32k-v2",
    [string]$DataFile = "./data/locomo10.json",
    [string]$OutDir = "./outputs",
    [string]$OutFile = "locomo10_qa.json",
    [switch]$Use4Bit = $true,
    [int]$BatchSize = 1,
    [string]$HFToken = ""
)

# Set HF Token if provided
if ($HFToken -ne "") {
    $env:HF_TOKEN = $HFToken
    Write-Host "HF_TOKEN has been set" -ForegroundColor Green
}

# Create output directory if it does not exist
if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
    Write-Host "Created output directory: $OutDir" -ForegroundColor Green
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Evaluation Configuration:" -ForegroundColor Cyan
Write-Host "  Model: $Model" -ForegroundColor White
Write-Host "  Data File: $DataFile" -ForegroundColor White
Write-Host "  Output Directory: $OutDir" -ForegroundColor White
Write-Host "  4-bit Quantization: $Use4Bit" -ForegroundColor White
Write-Host "  Batch Size: $BatchSize" -ForegroundColor White
Write-Host "========================================`n" -ForegroundColor Cyan

# Build command arguments
$args = @(
    "task_eval/evaluate_qa.py",
    "--data-file", $DataFile,
    "--out-file", "$OutDir/$OutFile",
    "--model", $Model,
    "--batch-size", $BatchSize
)

if ($Use4Bit) {
    $args += "--use-4bit"
}

# Run evaluation
Write-Host "Starting evaluation..." -ForegroundColor Yellow
python @args

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nEvaluation completed successfully!" -ForegroundColor Green
    Write-Host "Results saved in: $OutDir/$OutFile" -ForegroundColor Green
} else {
    Write-Host "`nEvaluation failed! Exit code: $LASTEXITCODE" -ForegroundColor Red
}
