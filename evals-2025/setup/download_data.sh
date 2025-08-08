#!/bin/bash
set -e

echo "Setting up BabyLM 2025 evaluation data..."

# Check if evaluation-pipeline-2025 exists
if [ ! -d "../evaluation-pipeline-2025" ]; then
    echo "Error: evaluation-pipeline-2025 directory not found"
    exit 1
fi

cd ../evaluation-pipeline-2025

echo "Please download evaluation_data from: https://osf.io/ryjfm/"
echo "Extract the contents to: $(pwd)/evaluation_data/"
echo ""
echo "The directory structure should look like:"
echo "evaluation-pipeline-2025/"
echo "├── evaluation_data/"
echo "│   ├── blimp/"
echo "│   ├── ewok/"
echo "│   ├── glue/"
echo "│   └── ..."

# Check if data directory exists
if [ ! -d "evaluation_data" ]; then
    echo "Creating evaluation_data directory..."
    mkdir -p evaluation_data
    echo "Please download and extract evaluation data to this directory."
else
    echo "evaluation_data directory found."
fi

# Download EWoK data if not present
echo "Checking for EWoK data..."
if [ ! -d "evaluation_data/ewok" ]; then
    echo "EWoK data not found. Please run:"
    echo "python -m evaluation_pipeline.ewok.dl_and_filter"
    echo "after downloading the main evaluation data."
fi

echo "Data setup instructions complete!"