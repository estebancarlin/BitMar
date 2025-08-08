# Complete BabyLM Evaluation Pipeline Setup Script (PowerShell)
# Run this script from your desired directory (e.g., d:\BabyLM)

Write-Host "🚀 Setting up BabyLM Evaluation Pipelines..." -ForegroundColor Green

# Create base directory
if (!(Test-Path "BabyLM")) {
    New-Item -ItemType Directory -Name "BabyLM"
}
Set-Location "BabyLM"

Write-Host "📥 Cloning evaluation pipelines..." -ForegroundColor Blue

# Clone Pipeline 2025
if (!(Test-Path "evaluation-pipeline-2025")) {
    git clone https://github.com/babylm/evaluation-pipeline-2025.git
    Write-Host "✅ Cloned evaluation-pipeline-2025" -ForegroundColor Green
} else {
    Write-Host "✅ evaluation-pipeline-2025 already exists" -ForegroundColor Green
}

# Clone Pipeline 2024
if (!(Test-Path "evaluation-pipeline-2024")) {
    git clone https://github.com/babylm/evaluation-pipeline-2024.git
    Write-Host "✅ Cloned evaluation-pipeline-2024" -ForegroundColor Green
} else {
    Write-Host "✅ evaluation-pipeline-2024 already exists" -ForegroundColor Green
}

Write-Host "📦 Installing dependencies..." -ForegroundColor Blue

# Install Pipeline 2025 dependencies
Set-Location "evaluation-pipeline-2025"
pip install -r requirements.txt
pip install transformers torch scikit-learn numpy pandas statsmodels datasets wandb nltk
Set-Location ".."

# Install Pipeline 2024 dependencies
Set-Location "evaluation-pipeline-2024"
pip install -e .
pip install minicons
pip install --upgrade accelerate
Set-Location ".."

Write-Host "🤗 Setting up HuggingFace access..." -ForegroundColor Magenta
Write-Host "Please run: huggingface-cli login"
Write-Host "Then request access to:"
Write-Host "- https://huggingface.co/datasets/facebook/winoground"
Write-Host "- https://huggingface.co/datasets/ewok-core/ewok-core-1.0"

Write-Host "📊 Downloading evaluation data..." -ForegroundColor Blue

# Download Pipeline 2025 evaluation data (full_eval)
Write-Host "Downloading Pipeline 2025 full evaluation data..." -ForegroundColor Yellow
Set-Location "evaluation-pipeline-2025"
if (!(Test-Path "evaluation_data")) {
    New-Item -ItemType Directory -Name "evaluation_data"
}
Set-Location "evaluation_data"

# Download the full_eval.zip from OSF
Write-Host "📥 Downloading full_eval.zip from OSF..." -ForegroundColor Yellow
$url = "https://files.osf.io/v1/resources/ryjfm/providers/osfstorage/6819fcae32b1521c270a7df8/?zip="
$output = "full_eval.zip"

try {
    Invoke-WebRequest -Uri $url -OutFile $output -UseBasicParsing
    Write-Host "✅ Downloaded full_eval.zip" -ForegroundColor Green
    
    # Extract the evaluation data
    Write-Host "📦 Extracting full_eval.zip..." -ForegroundColor Yellow
    Expand-Archive -Path $output -DestinationPath "." -Force
    Write-Host "✅ Pipeline 2025 evaluation data extracted" -ForegroundColor Green
    
    # Check if extraction was successful
    if (Test-Path "full_eval") {
        Write-Host "✅ Found full_eval directory with the following structure:" -ForegroundColor Green
        Write-Host "   - blimp_filtered/"
        Write-Host "   - supplement_filtered/"  
        Write-Host "   - ewok_filtered/"
        Write-Host "   - entity_tracking/"
        Write-Host "   - glue_filtered/"
        Write-Host "   - winoground_filtered/"
        Write-Host "   - vqa_filtered/"
        Write-Host "   - wug_adj_nominalization/"
        Write-Host "   - wug_past_tense/"
        Write-Host "   - comps/"
        Write-Host "   - reading/"
        Write-Host "   - cdi_childes/"
    } else {
        Write-Host "❌ full_eval directory not found after extraction" -ForegroundColor Red
    }
    
    # Clean up zip file
    Remove-Item $output
    
} catch {
    Write-Host "❌ Failed to download or extract evaluation data: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please download manually from: https://files.osf.io/v1/resources/ryjfm/providers/osfstorage/6819fcae32b1521c270a7df8/?zip=" -ForegroundColor Yellow
}

Set-Location ".."
Set-Location ".."

# Download Pipeline 2024 evaluation data
Write-Host "📥 Setting up Pipeline 2024 evaluation data..." -ForegroundColor Blue
Set-Location "evaluation-pipeline-2024"
if (!(Test-Path "evaluation_data")) {
    New-Item -ItemType Directory -Name "evaluation_data"
}

Write-Host "Please download evaluation data from: https://osf.io/ad7qg/" -ForegroundColor Yellow
Write-Host "Extract and place in: $((Get-Location).Path)\evaluation_data" -ForegroundColor Yellow

Set-Location ".."

Write-Host "🗂️ Setting up EWoK data..." -ForegroundColor Blue
Set-Location "evaluation-pipeline-2025"
try {
    python -m evaluation_pipeline.ewok.dl_and_filter
    Write-Host "✅ EWoK data setup completed" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Failed to setup EWoK data: $($_.Exception.Message)" -ForegroundColor Yellow
}
Set-Location ".."

Write-Host "🗂️ Setting up DevBench data..." -ForegroundColor Blue
Set-Location "evaluation-pipeline-2024"
if (Test-Path "devbench\download_data.sh") {
    # For Windows, we need to run this through WSL or Git Bash
    Write-Host "DevBench script found. Please run manually:" -ForegroundColor Yellow
    Write-Host "bash devbench/download_data.sh" -ForegroundColor Yellow
} else {
    Write-Host "⚠️ devbench/download_data.sh not found, skipping DevBench setup" -ForegroundColor Yellow
}
Set-Location ".."

Write-Host "✅ Setup script completed!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "1. Ensure HuggingFace CLI is logged in: huggingface-cli login"
Write-Host "2. Request access to restricted datasets:"
Write-Host "   - https://huggingface.co/datasets/facebook/winoground"
Write-Host "   - https://huggingface.co/datasets/ewok-core/ewok-core-1.0"
Write-Host "3. Validate setup:"
Write-Host "   cd BitMar"
Write-Host "   python validate_evaluation_setup.py"
Write-Host "4. Start training with evaluation:"
Write-Host "   python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml"
