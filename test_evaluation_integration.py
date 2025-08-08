"""
Test script for BabyLM Evaluation Integration
Tests the evaluation pipeline integration without running full training
"""

import sys
import logging
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_evaluation_integration():
    """Test the evaluation integration setup"""
    try:
        from src.training_evaluation_integration import TrainingEvaluationIntegration, BitMarEvaluationWrapper
        logger.info("✅ Successfully imported evaluation integration modules")
        
        # Test creating evaluation integration
        eval_integration = TrainingEvaluationIntegration(
            pipeline_2024_path="d:/BabyLM/evaluation-pipeline-2024",
            pipeline_2025_path="d:/BabyLM/evaluation-pipeline-2025",
            results_dir="test_evaluation_results",
            eval_frequency=1,
            fast_eval_epochs=[1, 2],
            full_eval_epochs=[3]
        )
        
        logger.info("✅ Successfully created TrainingEvaluationIntegration")
        logger.info(f"Fast eval epochs: {eval_integration.fast_eval_epochs}")
        logger.info(f"Full eval epochs: {eval_integration.full_eval_epochs}")
        
        # Test evaluation scheduling
        test_epochs = [1, 2, 3, 4, 5]
        for epoch in test_epochs:
            should_run, is_fast = eval_integration.should_evaluate(epoch)
            status = "No eval"
            if should_run:
                status = "Fast eval" if is_fast else "Full eval"
            logger.info(f"Epoch {epoch}: {status}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing evaluation integration: {e}")
        return False

def test_evaluation_pipelines():
    """Test that evaluation pipeline directories exist"""
    try:
        pipeline_2024 = Path("d:/BabyLM/evaluation-pipeline-2024")
        pipeline_2025 = Path("d:/BabyLM/evaluation-pipeline-2025")
        
        # Check 2024 pipeline
        if not pipeline_2024.exists():
            logger.error(f"❌ Pipeline 2024 directory not found: {pipeline_2024}")
            return False
        
        required_2024_files = ["eval_multimodal.sh"]
        for file_name in required_2024_files:
            file_path = pipeline_2024 / file_name
            if not file_path.exists():
                logger.error(f"❌ Required file not found: {file_path}")
                return False
        
        logger.info("✅ Pipeline 2024 validation passed")
        
        # Check 2025 pipeline
        if not pipeline_2025.exists():
            logger.error(f"❌ Pipeline 2025 directory not found: {pipeline_2025}")
            return False
        
        required_2025_files = ["eval_zero_shot.sh", "eval_finetuning.sh", "evaluation_pipeline"]
        for file_name in required_2025_files:
            file_path = pipeline_2025 / file_name
            if not file_path.exists():
                logger.error(f"❌ Required file not found: {file_path}")
                return False
        
        logger.info("✅ Pipeline 2025 validation passed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating pipelines: {e}")
        return False

def test_dummy_model_creation():
    """Test creating a dummy HuggingFace-compatible model"""
    try:
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
        
        # Create a minimal config
        config = GPT2Config(
            vocab_size=1000,
            n_positions=512,
            n_ctx=512,
            n_embd=384,
            n_layer=6,
            n_head=6,
        )
        
        # Create model
        model = GPT2LMHeadModel(config)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        # Test saving
        test_path = Path("test_hf_model")
        test_path.mkdir(exist_ok=True)
        
        model.save_pretrained(test_path)
        tokenizer.save_pretrained(test_path)
        
        logger.info("✅ Successfully created and saved dummy HuggingFace model")
        
        # Test loading
        loaded_model = GPT2LMHeadModel.from_pretrained(test_path)
        loaded_tokenizer = GPT2Tokenizer.from_pretrained(test_path)
        
        logger.info("✅ Successfully loaded dummy HuggingFace model")
        
        # Clean up
        import shutil
        shutil.rmtree(test_path)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing dummy model creation: {e}")
        return False

def test_config_loading():
    """Test loading the configuration with evaluation settings"""
    try:
        import yaml
        
        config_path = "configs/bitmar_100M_tokens.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check evaluation settings
        if 'evaluation' not in config:
            logger.error("❌ No evaluation section in config")
            return False
        
        eval_config = config['evaluation']
        required_keys = ['enabled', 'pipeline_2024_path', 'pipeline_2025_path', 'fast_eval_epochs', 'full_eval_epochs']
        
        for key in required_keys:
            if key not in eval_config:
                logger.error(f"❌ Missing evaluation config key: {key}")
                return False
        
        logger.info("✅ Configuration validation passed")
        logger.info(f"Evaluation enabled: {eval_config['enabled']}")
        logger.info(f"Fast eval epochs: {eval_config['fast_eval_epochs']}")
        logger.info(f"Full eval epochs: {eval_config['full_eval_epochs']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing config loading: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🧪 Testing BabyLM Evaluation Integration")
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Evaluation Pipelines", test_evaluation_pipelines),
        ("Evaluation Integration", test_evaluation_integration),
        ("Dummy Model Creation", test_dummy_model_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            if test_func():
                logger.info(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
    
    logger.info(f"\n🧪 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("✅ All tests passed! Evaluation integration is ready.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
