#!/bin/bash
set -e

echo "BabyLM Pipeline Quick Test"
echo "========================="

# Make scripts executable
chmod +x setup/*.sh
chmod +x evaluation_runners/*.sh

# 1. Test Python imports
echo "1. Testing Python imports..."
python -c "
import sys
sys.path.append('model_management')
sys.path.append('evaluation_runners')
sys.path.append('results_processing')
sys.path.append('analysis')

try:
    from validate_model_compatibility import test_model_loading
    from collate_all_results import load_results_from_directory
    from performance_analysis import plot_learning_curves
    print('✓ All imports successful')
except Exception as e:
    print(f'✗ Import failed: {e}')
    exit(1)
"

# 2. Test with mock data
echo "2. Testing with mock data..."
python testing_mock_data.py

# 3. Test model validation (with small model)
echo "3. Testing model validation..."
python testing_small_models.py

echo "✓ Quick test completed!"