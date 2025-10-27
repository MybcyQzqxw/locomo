# Windows PowerShell Scripts

This directory contains PowerShell scripts for running LoCoMo evaluations on Windows systems.

## Files

- `evaluate_hf_llm.ps1` - Evaluates HuggingFace models on locomo10 dataset
- `evaluate_hf_llm_1.ps1` - Evaluates HuggingFace models on locomo1 dataset

## Prerequisites

1. Install Anaconda or Miniconda for Windows
2. Initialize PowerShell for conda: `conda init powershell`
3. Create and activate the locomo environment
4. Install required packages: `pip install -r requirements.txt`

## Usage

```powershell
# Run in PowerShell
cd path\to\locomo
.\scripts\windows\evaluate_hf_llm.ps1
```

## Note

These scripts are Windows equivalents of the bash scripts in the parent `scripts/` directory. For Linux/macOS users, please use the `.sh` scripts instead.
