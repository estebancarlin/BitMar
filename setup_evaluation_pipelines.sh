#!/bin/bash
# Complete BabyLM Evaluation Pipeline Setup Script
# Run this script from your desired directory (e.g., d:\BabyLM)

echo "🚀 Setting up BabyLM Evaluation Pipelines..."

# Create base directory
mkdir -p BabyLM
cd BabyLM

echo "📥 Cloning evaluation pipelines..."

# Clone Pipeline 2025
if [ ! -d "evaluation-pipeline-2025" ]; then
    git clone https://github.com/babylm/evaluation-pipeline-2025.git
    echo "✅ Cloned evaluation-pipeline-2025"
else
    echo "✅ evaluation-pipeline-2025 already exists"
fi

# Clone Pipeline 2024
if [ ! -d "evaluation-pipeline-2024" ]; then
    git clone https://github.com/babylm/evaluation-pipeline-2024.git
    echo "✅ Cloned evaluation-pipeline-2024"
else
    echo "✅ evaluation-pipeline-2024 already exists"
fi

echo "📦 Installing dependencies..."

# Install Pipeline 2025 dependencies
cd evaluation-pipeline-2025
pip install -r requirements.txt
pip install transformers torch scikit-learn numpy pandas statsmodels datasets wandb nltk
cd ..

# Install Pipeline 2024 dependencies
cd evaluation-pipeline-2024
pip install -e .
pip install minicons
pip install --upgrade accelerate
cd ..

echo "🤗 Setting up HuggingFace access..."
echo "Please run: huggingface-cli login"
echo "Then request access to:"
echo "- https://huggingface.co/datasets/facebook/winoground"
echo "- https://huggingface.co/datasets/ewok-core/ewok-core-1.0"

echo "📊 Downloading evaluation data..."

# Download Pipeline 2025 evaluation data (full_eval)
echo "Downloading Pipeline 2025 full evaluation data..."
cd evaluation-pipeline-2025
mkdir -p evaluation_data
cd evaluation_data

# Download the full_eval.zip from OSF
echo "📥 Downloading full_eval.zip from OSF..."
curl -L -o full_eval.zip "https://files.osf.io/v1/resources/ryjfm/providers/osfstorage/6819fcae32b1521c270a7df8/?zip="

# Extract the evaluation data
echo "📦 Extracting full_eval.zip..."
unzip -q full_eval.zip
echo "✅ Pipeline 2025 evaluation data extracted"

# Check if extraction was successful
if [ -d "full_eval" ]; then
    echo "✅ Found full_eval directory with the following structure:"
    echo "   - blimp_filtered/"
    echo "   - supplement_filtered/"  
    echo "   - ewok_filtered/"
    echo "   - entity_tracking/"
    echo "   - glue_filtered/"
    echo "   - winoground_filtered/"
    echo "   - vqa_filtered/"
    echo "   - wug_adj_nominalization/"
    echo "   - wug_past_tense/"
    echo "   - comps/"
    echo "   - reading/"
    echo "   - cdi_childes/"
else
    echo "❌ full_eval directory not found after extraction"
fi

cd ../..

# Download Pipeline 2024 evaluation data
echo "📥 Downloading Pipeline 2024 evaluation data..."
cd evaluation-pipeline-2024
mkdir -p evaluation_data
cd evaluation_data

echo "Please download evaluation data from: https://osf.io/ad7qg/"
echo "Extract and place in: $(pwd)"

cd ../..

echo "🗂️  Setting up EWoK data..."
cd evaluation-pipeline-2025
python -m evaluation_pipeline.ewok.dl_and_filter
cd ..

echo "🗂️  Setting up DevBench data..."
cd evaluation-pipeline-2024
if [ -f "devbench/download_data.sh" ]; then
    bash devbench/download_data.sh
else
    echo "⚠️  devbench/download_data.sh not found, skipping DevBench setup"
fi
cd ..

echo "✅ Setup script completed!"
echo ""
echo "📋 Next steps:"
echo "1. Ensure HuggingFace CLI is logged in: huggingface-cli login"
echo "2. Request access to restricted datasets:"
echo "   - https://huggingface.co/datasets/facebook/winoground"
echo "   - https://huggingface.co/datasets/ewok-core/ewok-core-1.0"
echo "3. Validate setup:"
echo "   cd BitMar"
echo "   python validate_evaluation_setup.py"
echo "4. Start training with evaluation:"
echo "   python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml"
