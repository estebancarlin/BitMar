#!/usr/bin/env python3
"""
Test script to verify wandb logging fixes
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_wandb_logging():
    """Test wandb logging with error handling"""
    try:
        from src.wandb_logger import BitMarWandbLogger
        
        # Create a test config
        test_config = {
            'model': {
                'text_encoder_dim': 128,
                'vision_latent_size': 64,
                'fusion_hidden_size': 1024,
                'episode_dim': 1024
            },
            'training': {
                'learning_rate': 0.0001,
                'batch_size': 8
            }
        }
        
        logger.info("🧪 Creating wandb logger...")
        wandb_logger = BitMarWandbLogger(
            project_name="bitmar-test",
            config=test_config,
            run_name="test_error_handling"
        )
        
        # Test basic logging
        logger.info("🧪 Testing basic learning rate logging...")
        wandb_logger.log_learning_rate(lr=0.001, step=1)
        
        logger.info("✅ Wandb logging test completed successfully!")
        
        # Clean up
        import wandb
        wandb.finish()
        
    except Exception as e:
        logger.error(f"❌ Wandb logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    logger.info("🚀 Starting wandb logging test...")
    success = test_wandb_logging()
    
    if success:
        logger.info("✅ All tests passed!")
        sys.exit(0)
    else:
        logger.error("❌ Tests failed!")
        sys.exit(1)
