#!/bin/bash
set -e

echo "Setting up BabyLM 2025 evaluation environment..."

# Check if we're in the right directory structure
if [ ! -d "../evaluation-pipeline-2025" ]; then
    echo "Error: evaluation-pipeline-2025 directory not found at expected location"
    echo "Please ensure this script is run from evals-2025/setup/"
    exit 1
fi

# Install requirements from evaluation pipeline
cd ../evaluation-pipeline-2025
echo "Installing evaluation pipeline requirements..."
pip install -r requirements.txt

# Install additional requirements for our scripts
cd ../evals-2025
if [ -f "requirements.txt" ]; then
    echo "Installing additional requirements..."
    pip install -r requirements.txt
fi

echo "Environment setup complete!"
echo "You can now run evaluation scripts using python -m evaluation_pipeline.* commands from evaluation-pipeline-2025 root."