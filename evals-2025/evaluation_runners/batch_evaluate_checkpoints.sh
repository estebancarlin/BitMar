#!/bin/bash

# Batch evaluate all checkpoints for BabyLM submission

MODEL_PATH=$1
ARCHITECTURE=${2:-"causal"}
EVAL_TYPE=${3:-"fast"}  # fast or full

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <model_path> [architecture] [eval_type]"
    echo "Example: $0 huggingface.co/username/model-name causal fast"
    exit 1
fi

echo "Starting batch evaluation of checkpoints..."
echo "Model: $MODEL_PATH"
echo "Architecture: $ARCHITECTURE" 
echo "Evaluation type: $EVAL_TYPE"

# Define checkpoints based on track
CHECKPOINTS_1_10M="chck_1M chck_2M chck_3M chck_4M chck_5M chck_6M chck_7M chck_8M chck_9M chck_10M"
CHECKPOINTS_20_100M="chck_20M chck_30M chck_40M chck_50M chck_60M chck_70M chck_80M chck_90M chck_100M"
CHECKPOINTS_200_1000M="chck_200M chck_300M chck_400M chck_500M chck_600M chck_700M chck_800M chck_900M chck_1000M"

ALL_CHECKPOINTS="$CHECKPOINTS_1_10M $CHECKPOINTS_20_100M"

# Add large checkpoints if not strict-small track
read -p "Include 200M-1000M checkpoints? (y/N): " include_large
if [[ $include_large =~ ^[Yy]$ ]]; then
    ALL_CHECKPOINTS="$ALL_CHECKPOINTS $CHECKPOINTS_200_1000M"
fi

# Navigate to evaluation pipeline directory
cd ../evaluation-pipeline-2025 || {
    echo "Error: Could not find evaluation-pipeline-2025 directory"
    exit 1
}

# Counter for progress tracking
total_checkpoints=$(echo $ALL_CHECKPOINTS | wc -w)
current=0

echo "Will evaluate $total_checkpoints checkpoints"

# Evaluate each checkpoint
for checkpoint in $ALL_CHECKPOINTS; do
    current=$((current + 1))
    echo ""
    echo "[$current/$total_checkpoints] Evaluating $checkpoint..."
    
    if [ "$EVAL_TYPE" = "fast" ]; then
        ./eval_zero_shot_fast.sh "$MODEL_PATH" "$checkpoint" "$ARCHITECTURE"
    else
        ./eval_zero_shot.sh "$MODEL_PATH" "$ARCHITECTURE" "$checkpoint"
    fi
    
    if [ $? -eq 0 ]; then
        echo "✓ $checkpoint evaluation completed"
    else
        echo "✗ $checkpoint evaluation failed"
        read -p "Continue with remaining checkpoints? (y/N): " continue_eval
        if [[ ! $continue_eval =~ ^[Yy]$ ]]; then
            echo "Evaluation stopped by user"
            exit 1
        fi
    fi
done

echo ""
echo "Batch evaluation completed!"
echo "Results can be found in the results/ directory"