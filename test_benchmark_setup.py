"""
Test script for benchmark evaluation setup
Validates that all benchmark components are working correctly
"""

import sys
import torch
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_benchmark_imports():
    """Test that all benchmark-related imports work"""
    logger.info("🧪 Testing benchmark imports...")
    
    try:
        from src.benchmark_evaluator import BenchmarkEvaluator
        logger.info("✅ BenchmarkEvaluator imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import BenchmarkEvaluator: {e}")
        return False
    
    try:
        import datasets
        logger.info("✅ datasets library available")
    except ImportError:
        logger.warning("⚠️  datasets library not available - some benchmarks may not work")
    
    try:
        import requests
        logger.info("✅ requests library available")
    except ImportError:
        logger.warning("⚠️  requests library not available - dataset downloads may not work")
    
    return True

def test_benchmark_initialization():
    """Test benchmark evaluator initialization"""
    logger.info("🔧 Testing benchmark evaluator initialization...")
    
    try:
        from src.benchmark_evaluator import BenchmarkEvaluator
        
        # Create a dummy model and tokenizer for testing
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(768, 1000)
                
            def forward(self, **kwargs):
                # Simulate model output
                batch_size = kwargs['input_ids'].size(0)
                seq_len = kwargs['input_ids'].size(1)
                
                logits = torch.randn(batch_size, seq_len, 1000)
                return type('Outputs', (), {
                    'logits': logits,
                    'memory_outputs': torch.randn(batch_size, 768) if hasattr(self, 'memory') else None
                })()
                
            def generate(self, **kwargs):
                # Simulate text generation
                batch_size = kwargs['input_ids'].size(0)
                seq_len = kwargs['input_ids'].size(1)
                max_new_tokens = kwargs.get('max_new_tokens', 50)
                
                # Return input_ids + generated tokens
                generated = torch.randint(0, 1000, (batch_size, seq_len + max_new_tokens))
                generated[:, :seq_len] = kwargs['input_ids']
                return generated
        
        class DummyTokenizer:
            def __init__(self):
                self.vocab_size = 1000
                self.eos_token_id = 2
                
            def __call__(self, text, **kwargs):
                # Simulate tokenization
                if isinstance(text, list):
                    batch_size = len(text)
                else:
                    batch_size = 1
                    text = [text]
                
                max_length = kwargs.get('max_length', 256)
                input_ids = torch.randint(0, self.vocab_size, (batch_size, max_length))
                attention_mask = torch.ones(batch_size, max_length)
                
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }
            
            def decode(self, token_ids, **kwargs):
                # Simulate decoding
                return f"Generated text for tokens {len(token_ids)}"
        
        # Test configuration
        test_config = {
            'evaluation': {
                'benchmark_evaluations': {
                    'enabled': True,
                    'max_samples_per_task': 10,  # Small for testing
                    'few_shot_examples': 2,
                    'use_episodic_analysis': True,
                    'evaluate_tiny_mmlu': False,  # Disable to avoid dataset downloads
                    'evaluate_tiny_helm': False,
                    'evaluate_wildchat': True,    # Use synthetic data
                    'evaluate_episodic_benchmarks': True
                }
            }
        }
        
        model = DummyModel()
        tokenizer = DummyTokenizer()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize benchmark evaluator
        evaluator = BenchmarkEvaluator(
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=test_config,
            save_dir="./test_benchmark_results"
        )
        
        logger.info("✅ Benchmark evaluator initialized successfully")
        
        # Test a simple evaluation
        logger.info("🧪 Testing WildChat evaluation...")
        wildchat_results = evaluator.evaluate_wildchat_50m()
        
        if 'error' not in wildchat_results:
            logger.info(f"✅ WildChat evaluation successful: {wildchat_results.get('response_quality', {}).get('average_score', 'N/A')}")
        else:
            logger.warning(f"⚠️  WildChat evaluation had issues: {wildchat_results['error']}")
        
        # Test episodic memory benchmarks
        logger.info("🧪 Testing episodic memory benchmarks...")
        episodic_results = evaluator.evaluate_episodic_memory_benchmarks()
        
        if 'error' not in episodic_results:
            logger.info("✅ Episodic memory benchmarks successful")
        else:
            logger.warning(f"⚠️  Episodic memory benchmarks had issues: {episodic_results['error']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Benchmark evaluator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_integration():
    """Test configuration file integration"""
    logger.info("📋 Testing configuration integration...")
    
    try:
        import yaml
        
        config_path = Path("configs/bitmar_100M_tokens_optimized_memory.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check if benchmark evaluation is properly configured
            benchmark_config = config.get('evaluation', {}).get('benchmark_evaluations', {})
            
            if benchmark_config.get('enabled', False):
                logger.info("✅ Benchmark evaluation enabled in configuration")
                logger.info(f"   • tinyMMLU: {benchmark_config.get('evaluate_tiny_mmlu', False)}")
                logger.info(f"   • tinyHELM: {benchmark_config.get('evaluate_tiny_helm', False)}")
                logger.info(f"   • WildChat: {benchmark_config.get('evaluate_wildchat', False)}")
                logger.info(f"   • Episodic benchmarks: {benchmark_config.get('evaluate_episodic_benchmarks', False)}")
            else:
                logger.warning("⚠️  Benchmark evaluation disabled in configuration")
            
            return True
        else:
            logger.warning(f"⚠️  Configuration file not found: {config_path}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Configuration integration test failed: {e}")
        return False

def main():
    """Run all benchmark tests"""
    logger.info("🚀 Starting benchmark evaluation tests...")
    
    tests = [
        ("Import Test", test_benchmark_imports),
        ("Initialization Test", test_benchmark_initialization),
        ("Configuration Test", test_configuration_integration)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("📊 TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("🎉 All benchmark tests passed! You're ready to run comprehensive evaluations.")
        logger.info("\nTo run benchmark evaluations during training:")
        logger.info("1. Ensure 'benchmark_evaluations.enabled: true' in your config")
        logger.info("2. Install additional dependencies: pip install -r requirements_benchmarks.txt")
        logger.info("3. Run training with: python train_100M_tokens.py --config configs/bitmar_100M_tokens_optimized_memory.yaml")
    else:
        logger.warning("⚠️  Some tests failed. Please check the errors above and install missing dependencies.")

if __name__ == "__main__":
    main()
