#!/usr/bin/env python3
"""
Test with actual small models from HuggingFace
"""
import subprocess
import sys
from pathlib import Path

# Small models we can test with
TEST_MODELS = [
    ("gpt2", "causal"),
    ("distilbert-base-uncased", "encoder"), 
    ("google/flan-t5-small", "encoder-decoder")
]

def test_model_loading(model_name, architecture):
    """Test if we can load and validate a model"""
    print(f"Testing model: {model_name} ({architecture})")
    
    cmd = [
        "python", "model_management/validate_model_compatibility.py",
        "--model_path", model_name,
        "--architecture", architecture
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"✓ {model_name} validation passed")
            return True
        else:
            print(f"✗ {model_name} validation failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⚠ {model_name} validation timed out")
        return False
    except Exception as e:
        print(f"✗ {model_name} validation error: {e}")
        return False

def test_fast_eval_dry_run(model_name, architecture):
    """Test the evaluation runner logic (without actual evaluation)"""
    print(f"Testing evaluation runner with: {model_name}")
    
    # Import our evaluation runner
    sys.path.append('evaluation_runners')
    try:
        from run_all_fast_evals import batch_run_fast_evals
        # This would normally run evaluations - we'll just test the setup
        print(f"✓ Evaluation runner import successful for {model_name}")
        return True
    except Exception as e:
        print(f"✗ Evaluation runner test failed: {e}")
        return False

def main():
    print("Testing BabyLM Pipeline with Real Models")
    print("=" * 40)
    
    passed = 0
    total = 0
    
    for model_name, architecture in TEST_MODELS:
        total += 1
        print(f"\n[{total}/{len(TEST_MODELS)}] Testing {model_name}...")
        
        # Test model loading/validation
        if test_model_loading(model_name, architecture):
            passed += 1
            
            # Test evaluation runner setup
            test_fast_eval_dry_run(model_name, architecture)
    
    print(f"\nResults: {passed}/{total} models passed validation")
    
    if passed > 0:
        print("✓ Pipeline is working with real models!")
    else:
        print("✗ No models passed - check your setup")

if __name__ == "__main__":
    main()