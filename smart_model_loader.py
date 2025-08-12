#!/usr/bin/env python3
"""
Smart BitMar Model Loader
Automatically infers correct config from saved model weights
"""

import sys
import os
import torch
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import BitMarModel, create_bitmar_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_bitnet_state_dict(saved_state_dict: Dict) -> Dict:
    """Remove BitNet quantization parameters"""
    converted_dict = {}
    
    for key, tensor in saved_state_dict.items():
        # Skip quantization-specific parameters
        if key.endswith(('.weight_scale', '.input_scale')):
            continue
        converted_dict[key] = tensor
    
    logger.info(f"Converted {len(saved_state_dict)} -> {len(converted_dict)} parameters (removed quantization)")
    return converted_dict


def infer_config_from_weights(state_dict: Dict, base_config: Dict) -> Dict:
    """
    Automatically infer model configuration from saved weights
    """
    config = base_config.copy()
    
    logger.info("🔍 Inferring model configuration from saved weights...")
    
    for key, tensor in state_dict.items():
        
        # Vision encoder dimensions
        if key == 'vision_encoder.layers.0.weight':
            vision_hidden_dim = tensor.shape[0]
            config['vision_hidden_size'] = vision_hidden_dim
            logger.info(f"📐 Inferred vision_hidden_size: {vision_hidden_dim}")
        
        elif key == 'vision_encoder.output_proj.weight':
            vision_latent_size = tensor.shape[0]  # output dim
            vision_hidden_dim = tensor.shape[1]   # input dim
            config['vision_latent_size'] = vision_latent_size
            config['vision_hidden_size'] = vision_hidden_dim
            logger.info(f"📐 Inferred vision_latent_size: {vision_latent_size}")
            logger.info(f"📐 Confirmed vision_hidden_size: {vision_hidden_dim}")
        
        # Text encoder dimensions
        elif key == 'text_encoder.layers.0.attn.q_proj.weight':
            text_encoder_dim = tensor.shape[0]
            config['text_encoder_dim'] = text_encoder_dim
            logger.info(f"📐 Inferred text_encoder_dim: {text_encoder_dim}")
        
        # Text decoder dimensions
        elif key == 'text_decoder.layers.0.attn.q_proj.weight':
            text_decoder_dim = tensor.shape[0]
            config['text_decoder_dim'] = text_decoder_dim
            logger.info(f"📐 Inferred text_decoder_dim: {text_decoder_dim}")
        
        # Fusion dimensions
        elif key == 'cross_modal_fusion.layers.0.attn.q_proj.weight':
            fusion_hidden_size = tensor.shape[0]
            config['fusion_hidden_size'] = fusion_hidden_size
            logger.info(f"📐 Inferred fusion_hidden_size: {fusion_hidden_size}")
        
        # Memory dimensions
        elif key == 'episodic_memory.memory_bank':
            memory_size, episode_dim = tensor.shape
            config['memory_size'] = memory_size
            config['episode_dim'] = episode_dim
            logger.info(f"📐 Inferred memory_size: {memory_size}")
            logger.info(f"📐 Inferred episode_dim: {episode_dim}")
    
    # Count layers by looking for highest layer index
    def count_layers(prefix: str) -> int:
        layer_indices = []
        for key in state_dict.keys():
            if key.startswith(f'{prefix}.layers.'):
                parts = key.split('.')
                if len(parts) > 2 and parts[2].isdigit():
                    layer_indices.append(int(parts[2]))
        return max(layer_indices) + 1 if layer_indices else 0
    
    # Infer layer counts
    text_encoder_layers = count_layers('text_encoder')
    text_decoder_layers = count_layers('text_decoder')
    fusion_layers = count_layers('cross_modal_fusion')
    
    if text_encoder_layers > 0:
        config['text_encoder_layers'] = text_encoder_layers
        logger.info(f"📐 Inferred text_encoder_layers: {text_encoder_layers}")
    
    if text_decoder_layers > 0:
        config['text_decoder_layers'] = text_decoder_layers
        logger.info(f"📐 Inferred text_decoder_layers: {text_decoder_layers}")
    
    if fusion_layers > 0:
        config['fusion_num_layers'] = fusion_layers
        logger.info(f"📐 Inferred fusion_num_layers: {fusion_layers}")
    
    # Set remaining defaults if not inferred
    defaults = {
        'text_decoder_dim': config.get('text_encoder_dim', 128),
        'text_decoder_layers': config.get('text_encoder_layers', 4),
        'text_decoder_heads': config.get('text_encoder_heads', 4),
        'fusion_num_heads': config.get('text_encoder_heads', 4),
        'fusion_num_layers': config.get('fusion_num_layers', 2),
        'memory_alpha': 0.1,
        'direct_writing': True
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    logger.info("✅ Configuration inference complete!")
    return config


def load_smart_bitmar_model(model_dir: str, device: str = "cuda") -> Tuple[BitMarModel, Dict]:
    """
    Smart loader that infers config from saved weights
    """
    
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    weights_path = model_dir / "pytorch_model.bin"
    
    # Load base config
    logger.info(f"Loading base config from: {config_path}")
    with open(config_path, 'r') as f:
        base_config = json.load(f)
    
    # Load saved weights
    logger.info(f"Loading weights from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            saved_state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            saved_state_dict = checkpoint['state_dict']
        else:
            saved_state_dict = checkpoint
    else:
        saved_state_dict = checkpoint
    
    logger.info(f"Loaded {len(saved_state_dict)} weight tensors from checkpoint")
    
    # Convert BitNet state_dict
    logger.info("Converting BitNet quantized weights...")
    converted_state_dict = convert_bitnet_state_dict(saved_state_dict)
    
    # Infer config from weights
    inferred_config = infer_config_from_weights(converted_state_dict, base_config)
    
    logger.info("📋 Final inferred config:")
    for key, value in inferred_config.items():
        if key not in base_config or base_config[key] != value:
            logger.info(f"  {key}: {value}")
    
    # Create model with inferred config
    logger.info("Creating BitMar model with inferred config...")
    model = create_bitmar_model(inferred_config)
    
    # Load weights
    logger.info("Loading weights into model...")
    missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys ({len(missing_keys)}): {missing_keys[:3]}...")
    if unexpected_keys:
        logger.warning(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:3]}...")
    
    if len(missing_keys) == 0 and len(unexpected_keys) == 0:
        logger.info("🎯 Perfect match! All weights loaded successfully.")
    elif len(missing_keys) < 10 and len(unexpected_keys) < 10:
        logger.info("✅ Good match! Minor mismatches are acceptable.")
    else:
        logger.warning("⚠️  Many mismatches detected. Model may not work properly.")
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    return model, inferred_config


def test_model_inference(model: BitMarModel, config: Dict, device: str) -> bool:
    """Test the model with inference"""
    try:
        logger.info("🧪 Testing model inference...")
        
        batch_size = 1
        seq_len = 10
        
        # Create test inputs
        dummy_input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len)).to(device)
        dummy_attention_mask = torch.ones(batch_size, seq_len).to(device)
        dummy_vision = torch.randn(batch_size, config['vision_encoder_dim']).to(device)
        
        with torch.no_grad():
            output = model(
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention_mask,
                vision_features=dummy_vision,
                mode="inference"
            )
        
        if 'logits' in output:
            logits = output['logits']
            logger.info(f"✅ Model test passed!")
            logger.info(f"   Logits shape: {logits.shape}")
            logger.info(f"   Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
            logger.info(f"   Logits mean: {logits.mean():.3f}")
            return True
        else:
            logger.error(f"❌ No 'logits' in output. Keys: {output.keys()}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart BitMar model loader")
    parser.add_argument("model_dir", help="Path to model directory")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--no-test", action="store_true", help="Skip testing")
    
    args = parser.parse_args()
    
    try:
        # Load model
        model, config = load_smart_bitmar_model(args.model_dir, args.device)
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"🎉 Model loaded successfully!")
        logger.info(f"📊 Total parameters: {total_params:,}")
        logger.info(f"💾 Estimated size: ~{total_params * 4 / 1024**2:.1f} MB")
        
        # Test model
        if not args.no_test:
            test_success = test_model_inference(model, config, args.device)
            if test_success:
                logger.info("🚀 Model is ready for evaluation!")
            else:
                logger.warning("⚠️  Model loaded but inference test failed")
        
        logger.info("🎯 Use this model for evaluation pipelines!")
        
    except Exception as e:
        logger.error(f"❌ Smart loading failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
