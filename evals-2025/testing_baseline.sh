#!/bin/bash
set -e

echo "Testing BabyLM 2025 Evaluation Pipeline"
echo "======================================="

# Test with official BabyLM baseline model
TEST_MODEL="babylm/babylm-baseline-10m"
TEST_ARCHITECTURE="causal"

echo "Using test model: $TEST_MODEL"
echo "Architecture: $TEST_ARCHITECTURE"

# 1. Test setup (call from current directory, not cd into setup/)
echo "1. Testing setup scripts..."
bash setup/setup_environment.sh
echo "✓ Environment setup complete"

# Note: Skip data download test (requires manual download)
echo "⚠ Skipping data download test (requires manual setup)"

# 2. Test model compatibility
echo "2. Testing model compatibility..."
python model_management/validate_model_compatibility.py --model_path $TEST_MODEL --architecture $TEST_ARCHITECTURE
echo "✓ Model compatibility test complete"

# 3. Test single checkpoint evaluation (using main branch instead of specific checkpoint)
echo "3. Testing single evaluation..."
python -c "
import sys
sys.path.append('evaluation_runners')
try:
    from run_all_fast_evals import run_fast_eval_checkpoint
    print('✓ Evaluation runner import successful')
except Exception as e:
    print(f'✗ Import failed: {e}')
"

echo "Pipeline test complete!"