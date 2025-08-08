#!/usr/bin/env python3
"""
Test script for tinyBenchmarks integration with BitMar model evaluation.
This script validates that the tinyBenchmarks package is properly integrated
and can evaluate the BitMar model on tiny benchmark datasets.
"""

import os
import sys
import logging
import torch
import time
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_tinybenchmarks_availability():
    """Test if tinyBenchmarks package is available"""
    try:
        import tinyBenchmarks as tb
        logger.info("✅ tinyBenchmarks package imported successfully")
        
        # Check available datasets
        available_datasets = ['tinyMMLU', 'tinyTruthfulQA', 'tinyGSM8K', 
                            'tinyWinogrande', 'tinyARC', 'tinyHellaSwag', 'tinyAlpacaEval']
        
        logger.info(f"📊 Available tinyBenchmarks datasets: {available_datasets}")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import tinyBenchmarks: {e}")
        logger.info("💡 Install with: pip install tinyBenchmarks")
        return False

def test_huggingface_datasets():
    """Test if HuggingFace community datasets are accessible"""
    try:
        from datasets import load_dataset
        
        # Test loading one of the tiny datasets
        logger.info("🔍 Testing HuggingFace community dataset access...")
        
        # Try to load a small sample
        dataset = load_dataset("felipemaiapolo/tinyMMLU", split="test")
        logger.info(f"✅ Successfully loaded tinyMMLU with {len(dataset)} samples")
        
        # Show a sample
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"📝 Sample question: {sample.get('question', 'N/A')[:100]}...")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load HuggingFace datasets: {e}")
        return False

def test_benchmark_evaluator():
    """Test the BenchmarkEvaluator class with tinyBenchmarks"""
    try:
        from benchmark_evaluator import BenchmarkEvaluator
        
        logger.info("🧪 Testing BenchmarkEvaluator with tinyBenchmarks...")
        
        # Create mock model and tokenizer for testing
        class MockModel:
            def __init__(self):
                self.device = 'cpu'
                self.eval_mode = True
            
            def eval(self):
                self.eval_mode = True
                
            def train(self):
                self.eval_mode = False
                
            def to(self, device):
                self.device = device
                return self
                
            def generate(self, **kwargs):
                # Mock generation
                input_ids = kwargs.get('input_ids')
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
                
                # Generate dummy tokens
                new_tokens = torch.randint(1000, 2000, (batch_size, 10))
                return torch.cat([input_ids, new_tokens], dim=1)
        
        class MockTokenizer:
            def __init__(self):
                self.eos_token_id = 2
                self.pad_token_id = 0
                
            def __call__(self, text, **kwargs):
                # Mock tokenization
                tokens = torch.randint(100, 1000, (1, 20))
                return {
                    'input_ids': tokens,
                    'attention_mask': torch.ones_like(tokens)
                }
                
            def decode(self, tokens, **kwargs):
                # Mock decoding
                return f"Generated response for tokens {len(tokens)}"
        
        # Create test configuration
        test_config = {
            'evaluate_tiny_benchmarks': True,
            'tiny_benchmarks': {
                'datasets': ['tinyMMLU'],
                'use_irt_estimates': True,
                'max_samples_per_dataset': 5
            },
            'episodic_memory_analysis': {
                'enabled': True,
                'track_activations': True
            }
        }
        
        # Initialize evaluator
        model = MockModel()
        tokenizer = MockTokenizer()
        
        evaluator = BenchmarkEvaluator(
            model=model,
            tokenizer=tokenizer,
            config=test_config,
            device='cpu'
        )
        
        logger.info(f"✅ BenchmarkEvaluator initialized: {evaluator.enabled}")
        logger.info(f"📦 tinyBenchmarks available: {evaluator.tinybenchmarks_available}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to test BenchmarkEvaluator: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mock_evaluation():
    """Test a mock evaluation run"""
    try:
        logger.info("🚀 Running mock tinyBenchmarks evaluation...")
        
        # Import tinyBenchmarks
        import tinyBenchmarks as tb
        
        # Create simple mock model for testing
        def mock_model_fn(prompt):
            """Mock model function that returns random answers"""
            import random
            choices = ['A', 'B', 'C', 'D']
            return random.choice(choices)
        
        # Test evaluation on a small subset
        logger.info("📊 Testing tinyMMLU evaluation...")
        result = tb.evaluate(mock_model_fn, ['tinyMMLU'], verbose=True)
        
        logger.info(f"✅ Mock evaluation completed!")
        logger.info(f"📈 Results: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Mock evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    logger.info("🔬 Starting tinyBenchmarks integration tests...")
    
    tests = [
        ("tinyBenchmarks Package Availability", test_tinybenchmarks_availability),
        ("HuggingFace Datasets Access", test_huggingface_datasets),
        ("BenchmarkEvaluator Integration", test_benchmark_evaluator),
        ("Mock Evaluation Run", test_mock_evaluation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"🧪 Running: {test_name}")
        logger.info(f"{'='*50}")
        
        start_time = time.time()
        success = test_func()
        duration = time.time() - start_time
        
        results.append((test_name, success, duration))
        
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{status} - {test_name} ({duration:.2f}s)")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("📋 TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, duration in results:
        status = "✅" if success else "❌"
        logger.info(f"{status} {test_name} ({duration:.2f}s)")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! tinyBenchmarks integration is ready.")
        return 0
    else:
        logger.error("💥 Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
