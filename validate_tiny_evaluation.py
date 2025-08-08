"""
Validation script for tiny model evaluation integration
Tests the tiny model evaluator without full training
"""

import sys
from pathlib import Path
import torch
import yaml
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_tiny_model_evaluation():
    """Validate that tiny model evaluation is properly integrated"""
    
    try:
        # Load configuration
        config_path = "configs/bitmar_100M_tokens_optimized_memory.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("✅ Configuration loaded successfully")
        
        # Check tiny model evaluation config
        tiny_eval_config = config.get('evaluation', {}).get('tiny_model_evaluations', {})
        if tiny_eval_config.get('enabled', False):
            logger.info("✅ Tiny model evaluation is enabled in config")
            logger.info(f"  • Step frequency: {tiny_eval_config.get('eval_frequency_steps', 2000)}")
            logger.info(f"  • Epoch frequency: {tiny_eval_config.get('eval_frequency_epochs', 1)}")
            
            # Check BabyLM config
            babylm_config = tiny_eval_config.get('babylm_tiny_evaluations', {})
            if babylm_config.get('enabled', False):
                logger.info("✅ BabyLM tiny evaluation is enabled")
                logger.info(f"  • Priority tasks: {babylm_config.get('priority_tasks', [])}")
                logger.info(f"  • Quick eval tasks: {babylm_config.get('quick_eval_tasks', [])}")
        else:
            logger.warning("⚠️  Tiny model evaluation is disabled in config")
        
        # Test import
        try:
            from src.tiny_model_evaluator import TinyModelEvaluator, TinyModelBabyLMEvaluator
            logger.info("✅ Tiny model evaluator imports successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import tiny model evaluator: {e}")
            return False
        
        # Test basic functionality
        logger.info("🔬 Testing tiny model evaluator functionality...")
        
        # Create a dummy model for testing
        class DummyModel:
            def __init__(self):
                self.linear = torch.nn.Linear(10, 5)
                
            def parameters(self):
                return self.linear.parameters()
                
            def eval(self):
                pass
                
            def train(self):
                pass
        
        class DummyTokenizer:
            def __init__(self):
                self.vocab_size = 1000
                
            def __call__(self, text, **kwargs):
                return {
                    'input_ids': torch.randint(0, 1000, (1, 10)),
                    'attention_mask': torch.ones(1, 10)
                }
        
        # Test evaluator creation
        dummy_model = DummyModel()
        dummy_tokenizer = DummyTokenizer()
        
        evaluator = TinyModelEvaluator(
            model=dummy_model,
            tokenizer=dummy_tokenizer,
            device="cpu",
            save_dir="./test_tiny_eval",
            config=config
        )
        
        logger.info("✅ Tiny model evaluator created successfully")
        
        # Test parameter efficiency
        param_eff = evaluator.evaluate_parameter_efficiency(0.75)
        logger.info(f"✅ Parameter efficiency test: {param_eff:.3f}")
        
        # Test memory efficiency  
        mem_eff = evaluator.evaluate_memory_efficiency(0.75)
        logger.info(f"✅ Memory efficiency test: {mem_eff:.3f}")
        
        # Test SD card readiness
        sd_readiness = evaluator.evaluate_sd_card_readiness()
        logger.info(f"✅ SD card readiness test: {sd_readiness:.3f}")
        
        logger.info("🎉 All tiny model evaluation tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    success = validate_tiny_model_evaluation()
    if success:
        print("\n🎯 VALIDATION SUMMARY:")
        print("✅ Tiny model evaluation is properly integrated")
        print("✅ Configuration is correctly set up") 
        print("✅ Evaluator functionality works")
        print("✅ Ready for training with tiny model evaluations")
    else:
        print("\n❌ VALIDATION FAILED")
        print("Please check the error messages above")
