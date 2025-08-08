#!/bin/bash
set -e

echo "Setting up BabyLM 2025 evaluation environment..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALS_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$EVALS_DIR")"

# Check if we can find evaluation-pipeline-2025
EVAL_PIPELINE_2025="$PROJECT_ROOT/evaluation-pipeline-2025"

if [ ! -d "$EVAL_PIPELINE_2025" ]; then
    echo "Error: evaluation-pipeline-2025 directory not found at: $EVAL_PIPELINE_2025"
    echo "Project structure should be:"
    echo "BitMar/"
    echo "├── evals-2025/"
    echo "└── evaluation-pipeline-2025/"
    echo ""
    echo "Current script location: $SCRIPT_DIR"
    echo "Looking for pipeline at: $EVAL_PIPELINE_2025"
    exit 1
fi

# Install requirements from evaluation pipeline
cd "$EVAL_PIPELINE_2025"
echo "Installing evaluation pipeline requirements..."
pip install -r requirements.txt

# Install additional requirements for our scripts
cd "$EVALS_DIR"
if [ -f "requirements.txt" ]; then
    echo "Installing additional requirements..."
    pip install -r requirements.txt
fi

echo "Environment setup complete!"
echo "You can now run evaluation scripts using python -m evaluation_pipeline.* commands from $EVAL_PIPELINE_2025"