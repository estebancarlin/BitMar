"""
Token-Aware Training Script for BitMar 100M Token Model (No Memory)
Multi-GPU compatible training with exactly 100M tokens
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm
import time
import traceback

# Hugging Face Hub integration
try:
    from huggingface_hub import HfApi, create_repo, upload_folder
    from transformers import AutoTokenizer, AutoConfig
    HF_HUB_AVAILABLE = True
    print("✅ Hugging Face Hub integration available")
except ImportError:
    HF_HUB_AVAILABLE = False
    print("⚠️  Hugging Face Hub not available - install with: pip install huggingface_hub")

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_100M_tokens.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import components
from src.dataset import create_data_module
from src.model_no_memory import create_bitmar_model, count_parameters
from src.wandb_logger_no_memory import BitMarWandbLogger
from src.attention_visualizer_no_memory import AttentionHeadAnalyzer

# Try to import FLOPS tracker
try:
    from src.flops_tracker import FLOPsTracker, FLOPsEstimator
    FLOPS_TRACKER_AVAILABLE = True
    logger.info("✅ FLOPS tracker available")
except ImportError:
    FLOPS_TRACKER_AVAILABLE = False
    logger.warning("⚠️  FLOPS tracker not available")

# Try to import token-constrained dataset
try:
    from src.token_constrained_dataset import create_token_constrained_data_module
    TOKEN_CONSTRAINED_AVAILABLE = True
    logger.info("✅ Token-constrained dataset available")
except ImportError:
    try:
        # Try importing without src prefix (in case running from different directory)
        from token_constrained_dataset import create_token_constrained_data_module
        TOKEN_CONSTRAINED_AVAILABLE = True
        logger.info("✅ Token-constrained dataset available (direct import)")
    except ImportError:
        TOKEN_CONSTRAINED_AVAILABLE = False
        logger.warning("⚠️  Token-constrained dataset not available")

# Try to import optional components
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

try:
    from src.adaptive_training_controller import AdaptiveTrainingController, compute_cross_modal_similarity
    ADAPTIVE_TRAINING_AVAILABLE = True
except ImportError:
    ADAPTIVE_TRAINING_AVAILABLE = False

# Try to import attention sinks integration
try:
    from src.attention_sinks_integration import (
        AttentionSinksConfig,
        apply_attention_sinks_to_bitmar_model,
        update_model_kwargs_for_generation_with_sinks
    )
    ATTENTION_SINKS_AVAILABLE = True
    logger.info("✅ Attention Sinks integration available")
except ImportError:
    ATTENTION_SINKS_AVAILABLE = False
    logger.warning("⚠️  Attention Sinks integration not available")


class TokenAwareTrainer:
    """Token-aware trainer for exactly 100M tokens"""

    def __init__(self, config_path: str, device: Optional[str] = None):
        """Initialize trainer with token awareness"""
        # Load configuration with validation
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Validate required config sections
            required_sections = ['token_constraints', 'model', 'data', 'training', 'output']
            for section in required_sections:
                if section not in self.config:
                    raise ValueError(f"Missing required config section: {section}")
                    
            logger.info(f"Configuration loaded successfully from {config_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load config: {e}")

        # Set device with enhanced GPU detection and error handling
        logger.info(f"🔍 GPU Detection:")
        logger.info(f"  • CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  • CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  • GPU {i}: {torch.cuda.get_device_name(i)}")
                logger.info(f"  • GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        
        if device:
            try:
                self.device = torch.device(device)
                # Test if device is available
                if device.startswith('cuda'):
                    if not torch.cuda.is_available():
                        logger.error(f"❌ CUDA not available but {device} requested!")
                        logger.error(f"   Training on CPU will take 15+ hours. Please install CUDA or use --device cpu explicitly")
                        raise RuntimeError(f"CUDA not available for {device}")
                    elif device != "cuda:0" and not torch.cuda.device_count() > int(device.split(':')[1]):
                        logger.warning(f"Device {device} not available, using cuda:0")
                        self.device = torch.device("cuda:0")
                    else:
                        # Test GPU by creating a small tensor
                        test_tensor = torch.tensor([1.0], device=self.device)
                        logger.info(f"✅ Successfully initialized {device}")
                        logger.info(f"   GPU memory allocated: {torch.cuda.memory_allocated(self.device) / 1024**2:.1f} MB")
                logger.info(f"Using device: {self.device}")
            except Exception as e:
                logger.error(f"Failed to set device {device}: {e}")
                logger.error(f"Available devices: {['cpu'] + [f'cuda:{i}' for i in range(torch.cuda.device_count())]}")
                raise RuntimeError(f"Device setup failed: {e}")
        else:
            # Auto-select best available device
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                # Test GPU
                test_tensor = torch.tensor([1.0], device=self.device)
                logger.info(f"✅ Auto-selected GPU: {self.device}")
                logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                self.device = torch.device("cpu")
                logger.warning(f"⚠️  No GPU available, using CPU (training will be very slow!)")
                logger.warning(f"   Expected training time: 15+ hours on CPU with batch_size=96")

        # Token tracking
        self.tokens_processed = 0
        self.target_tokens = self.config['token_constraints']['total_tokens']
        self.token_log_frequency = self.config.get('token_tracking', {}).get('log_frequency', 1000)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_similarity = 0.0
        self.token_exhausted = False

        # Setup directories
        self.setup_directories()
        
        # Setup logging
        self.setup_wandb()
        
        # Initialize carbon tracking if available
        self.setup_carbon_tracking()

        # Setup Hugging Face Hub integration
        self.setup_huggingface_hub()

        logger.info(f"🎯 Token-aware trainer initialized for {self.target_tokens:,} tokens")
        logger.info(f"Device: {self.device}")

    def setup_directories(self):
        """Create output directories"""
        for dir_name in ['checkpoint_dir', 'log_dir', 'attention_dir', 'results_dir', 'token_logs_dir']:
            dir_path = Path(self.config['output'][dir_name])
            dir_path.mkdir(parents=True, exist_ok=True)
            setattr(self, dir_name, dir_path)

    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        wandb_config = self.config.get('wandb', {})
        
        if wandb_config.get('project'):
            try:
                # Enhanced run name with token info
                run_name = f"bitmar-100M-tokens-{wandb.util.generate_id()[:8]}"
                
                self.wandb_logger = BitMarWandbLogger(
                    project_name=wandb_config['project'],
                    config=self.config,
                    entity=wandb_config.get('entity'),
                    run_name=run_name
                )
                self.use_wandb = True
                logger.info("✅ Weights & Biases initialized for 100M token training")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                self.use_wandb = False
                self.wandb_logger = None
        else:
            self.use_wandb = False
            self.wandb_logger = None

    def setup_carbon_tracking(self):
        """Setup carbon emissions tracking"""
        if not CODECARBON_AVAILABLE:
            self.carbon_tracker = None
            return

        try:
            carbon_logs_dir = Path("./carbon_logs_100M")
            carbon_logs_dir.mkdir(exist_ok=True)

            self.carbon_tracker = EmissionsTracker(
                project_name="BitMar-100M-Token-Training",
                experiment_id=f"bitmar-100M-{self.device.type}",
                output_dir=str(carbon_logs_dir),
                output_file="emissions_100M.csv",
                log_level="INFO",
                save_to_file=True,
                tracking_mode="machine"
            )
            logger.info("🌱 Carbon emissions tracking enabled for 100M token training")
        except Exception as e:
            logger.warning(f"Failed to setup carbon tracking: {e}")
            self.carbon_tracker = None

    def setup_huggingface_hub(self):
        """Setup Hugging Face Hub integration"""
        if not HF_HUB_AVAILABLE:
            self.hf_hub_enabled = False
            logger.warning("⚠️  Hugging Face Hub not available - model uploads disabled")
            return

        hf_config = self.config.get('huggingface_hub', {})

        if not hf_config.get('enabled', False):
            self.hf_hub_enabled = False
            logger.info("📤 Hugging Face Hub uploads disabled in config")
            return

        # Get repository ID
        self.hf_repo_id = hf_config.get('repo_id')
        if not self.hf_repo_id:
            self.hf_hub_enabled = False
            logger.warning("⚠️  No Hugging Face repo_id specified - model uploads disabled")
            return

        # Get or set up authentication token with multiple fallback options
        self.hf_token = None

        # 1. Check config file first
        if hf_config.get('token'):
            self.hf_token = hf_config.get('token')
            logger.info("🔑 Using Hugging Face token from config file")

        # 2. Check environment variable
        elif os.getenv('HF_TOKEN'):
            self.hf_token = os.getenv('HF_TOKEN')
            logger.info("🔑 Using Hugging Face token from HF_TOKEN environment variable")

        # 3. Try to get token from huggingface_hub default location
        else:
            try:
                from huggingface_hub import HfFolder
                stored_token = HfFolder.get_token()
                if stored_token:
                    self.hf_token = stored_token
                    logger.info("🔑 Using Hugging Face token from huggingface-cli login")
            except Exception as e:
                logger.debug(f"Failed to get token from HfFolder: {e}")

        # 4. Final fallback - try using HfApi without explicit token (uses cached credentials)
        if not self.hf_token:
            try:
                # Test if we can authenticate without explicit token
                test_api = HfApi()
                user_info = test_api.whoami()
                if user_info:
                    # Authentication worked, we can use the API without explicit token
                    self.hf_token = "cached_credentials"
                    logger.info("🔑 Using cached Hugging Face credentials")
                else:
                    raise Exception("No user info returned")
            except Exception as e:
                logger.debug(f"Failed to use cached credentials: {e}")

        if not self.hf_token:
            self.hf_hub_enabled = False
            logger.warning("⚠️  No Hugging Face token found - model uploads disabled")
            logger.warning("   Options to fix this:")
            logger.warning("   1. Run: huggingface-cli login")
            logger.warning("   2. Set HF_TOKEN environment variable")
            logger.warning("   3. Add token to config file")
            return

        try:
            # Initialize Hugging Face API
            if self.hf_token == "cached_credentials":
                self.hf_api = HfApi()  # Use cached credentials
            else:
                self.hf_api = HfApi(token=self.hf_token)

            # Test authentication
            user_info = self.hf_api.whoami()
            logger.info(f"✅ Authenticated with Hugging Face as: {user_info['name']}")

            # Check if repository exists, create if not
            try:
                repo_info = self.hf_api.repo_info(self.hf_repo_id, token=self.hf_token)
                logger.info(f"✅ Repository found: {self.hf_repo_id}")
            except Exception:
                logger.info(f"📤 Creating new repository: {self.hf_repo_id}")
                try:
                    create_repo(
                        repo_id=self.hf_repo_id,
                        token=self.hf_token,
                        private=hf_config.get('private', True),
                        exist_ok=True
                    )
                    logger.info(f"✅ Repository created: {self.hf_repo_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to create repository: {e}")
                    self.hf_hub_enabled = False
                    return

            # Store configuration
            self.hf_hub_enabled = True
            self.hf_upload_after_epoch = hf_config.get('upload_after_epoch', True)
            self.hf_upload_final_model = hf_config.get('upload_final_model', True)
            self.hf_commit_message_template = hf_config.get('commit_message_template',
                "BitMar 100M tokens - Epoch {epoch} - {tokens_processed:,} tokens processed")
            self.hf_create_model_card = hf_config.get('create_model_card', True)
            self.hf_model_card_template = hf_config.get('model_card_template', "")

            logger.info("🤗 Hugging Face Hub integration initialized:")
            logger.info(f"  • Repository: {self.hf_repo_id}")
            logger.info(f"  • Upload after epoch: {self.hf_upload_after_epoch}")
            logger.info(f"  • Upload final model: {self.hf_upload_final_model}")
            logger.info(f"  • Create model card: {self.hf_create_model_card}")

        except Exception as e:
            logger.error(f"❌ Failed to setup Hugging Face Hub: {e}")
            self.hf_hub_enabled = False

    def create_model_card(self, epoch: int, final: bool = False) -> str:
        """Create model card content"""
        if not self.hf_model_card_template:
            return ""

        try:
            # Format template with current training state
            card_content = self.hf_model_card_template.format(
                epoch=epoch + 1,
                tokens_processed=self.tokens_processed,
                best_similarity=self.best_similarity,
                repo_id=self.hf_repo_id,
                text_encoder_layers=self.config['model']['text_encoder_layers'],
                text_encoder_dim=self.config['model']['text_encoder_dim'],
                vision_latent_size=self.config['model']['vision_latent_size']
            )

            # Add training status
            if final:
                card_content += f"\n\n## Training Status\n- **Status**: Completed\n"
            else:
                card_content += f"\n\n## Training Status\n- **Status**: In Progress (Epoch {epoch + 1})\n"

            card_content += f"- **Tokens Processed**: {self.tokens_processed:,}\n"
            card_content += f"- **Best Cross-modal Similarity**: {self.best_similarity:.4f}\n"

            return card_content
        except Exception as e:
            logger.warning(f"Failed to create model card: {e}")
            return ""

    def prepare_model_for_upload(self, checkpoint_path: Path) -> Path:
        """Prepare model files for Hugging Face upload"""
        try:
            # Create temporary directory for HF model files
            hf_model_dir = self.checkpoint_dir / "hf_model_temp"
            hf_model_dir.mkdir(exist_ok=True)

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Save model state dict in HF format
            model_path = hf_model_dir / "pytorch_model.bin"
            torch.save(checkpoint['model_state_dict'], model_path)

            # Create config.json for the model
            model_config = {
                "architectures": ["BitMarModel"],
                "model_type": "bitmar",
                "vocab_size": self.config['model']['vocab_size'],
                "text_encoder_dim": self.config['model']['text_encoder_dim'],
                "text_encoder_layers": self.config['model']['text_encoder_layers'],
                "text_encoder_heads": self.config['model']['text_encoder_heads'],
                "vision_encoder_dim": self.config['model']['vision_encoder_dim'],
                "vision_latent_size": self.config['model']['vision_latent_size'],
                "fusion_hidden_size": self.config['model']['fusion_hidden_size'],
                "max_seq_len": self.config['model']['max_seq_len'],
                "dropout": self.config['model']['dropout'],
                "torch_dtype": "float32",
                "transformers_version": "4.0.0"
            }

            config_path = hf_model_dir / "config.json"
            with open(config_path, 'w') as f:
                import json
                json.dump(model_config, f, indent=2)

            # Save training metadata
            training_metadata = {
                "epoch": checkpoint['epoch'],
                "global_step": checkpoint['global_step'],
                "tokens_processed": checkpoint['tokens_processed'],
                "target_tokens": checkpoint['target_tokens'],
                "best_similarity": checkpoint['best_similarity'],
                "training_config": self.config
            }

            metadata_path = hf_model_dir / "training_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(training_metadata, f, indent=2)

            logger.info(f"✅ Model prepared for upload in: {hf_model_dir}")
            return hf_model_dir

        except Exception as e:
            logger.error(f"❌ Failed to prepare model for upload: {e}")
            raise

    def upload_checkpoint_to_hf(self, epoch: int, final: bool = False):
        """Upload model checkpoint to Hugging Face Hub"""
        if not self.hf_hub_enabled:
            return

        try:
            logger.info(f"📤 Uploading {'final ' if final else ''}model to Hugging Face Hub...")

            # Get checkpoint path
            if final:
                checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pt'
            else:
                checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}_tokens_{self.tokens_processed}.pt'

            if not checkpoint_path.exists():
                logger.warning(f"⚠️  Checkpoint not found: {checkpoint_path}")
                return

            # Prepare model files
            hf_model_dir = self.prepare_model_for_upload(checkpoint_path)

            # Create model card if enabled
            if self.hf_create_model_card:
                model_card_content = self.create_model_card(epoch, final)
                if model_card_content:
                    readme_path = hf_model_dir / "README.md"
                    with open(readme_path, 'w') as f:
                        f.write(model_card_content)

            # Create commit message
            commit_message = self.hf_commit_message_template.format(
                epoch=epoch + 1,
                tokens_processed=self.tokens_processed
            )

            if final:
                commit_message = f"Final model - {commit_message}"

            # Upload to Hugging Face Hub
            logger.info(f"📤 Uploading files to {self.hf_repo_id}...")
            self.hf_api.upload_folder(
                folder_path=str(hf_model_dir),
                repo_id=self.hf_repo_id,
                token=self.hf_token,
                commit_message=commit_message
            )

            logger.info(f"✅ Model uploaded successfully to: https://huggingface.co/{self.hf_repo_id}")

            # Log to wandb if available
            if self.use_wandb:
                try:
                    wandb.log({
                        f'huggingface/upload_success': True,
                        f'huggingface/epoch': epoch + 1,
                        f'huggingface/repo_url': f"https://huggingface.co/{self.hf_repo_id}"
                    }, step=self.global_step)
                except Exception as e:
                    logger.warning(f"Failed to log HF upload to wandb: {e}")

            # Cleanup temporary directory
            import shutil
            shutil.rmtree(hf_model_dir, ignore_errors=True)

        except Exception as e:
            logger.error(f"❌ Failed to upload model to Hugging Face Hub: {e}")

            # Log failure to wandb if available
            if self.use_wandb:
                try:
                    wandb.log({
                        f'huggingface/upload_success': False,
                        f'huggingface/error': str(e)
                    }, step=self.global_step)
                except Exception:
                    pass

    def custom_collate_fn(self, batch):
        """Custom collate function that handles missing keys gracefully and ensures proper padding"""
        if not batch:
            return {}
            
        # Get all keys from first sample as baseline
        first_sample = batch[0]
        all_keys = set(first_sample.keys())
        
        # Add any missing keys from other samples
        for sample in batch[1:]:
            all_keys.update(sample.keys())
        
        result = {}
        batch_size = len(batch)
        
        # Special handling for sequence-based tensors that need padding
        sequence_keys = ['input_ids', 'attention_mask', 'labels']
        
        for key in all_keys:
            values = []
            for i, sample in enumerate(batch):
                if key in sample:
                    value = sample[key]
                    # Ensure tensor conversion for specific keys
                    if key in ['vision_index', 'has_vision', 'index']:
                        if not torch.is_tensor(value):
                            if key == 'vision_index' or key == 'index':
                                value = torch.tensor(value, dtype=torch.long)
                            elif key == 'has_vision':
                                value = torch.tensor(value, dtype=torch.bool)
                    values.append(value)
                else:
                    # Provide sensible defaults for missing keys
                    if key == 'vision_index':
                        values.append(torch.tensor(i, dtype=torch.long))  # Use batch index as default
                    elif key == 'has_vision':
                        values.append(torch.tensor(True, dtype=torch.bool))  # Default to having vision
                    elif key == 'index':
                        values.append(torch.tensor(i, dtype=torch.long))  # Use batch index
                    else:
                        # For other keys, use zero tensor with same shape as first valid sample
                        for sample in batch:
                            if key in sample and sample[key] is not None:
                                if torch.is_tensor(sample[key]):
                                    values.append(torch.zeros_like(sample[key]))
                                else:
                                    values.append(sample[key])  # Copy first valid value
                                break
                        else:
                            values.append(None)  # No valid sample found
            
            # Handle sequence keys that need padding
            if key in sequence_keys and all(v is not None for v in values):
                try:
                    if all(torch.is_tensor(v) for v in values):
                        # Find maximum sequence length
                        if values[0].dim() > 0:
                            max_len = max(v.size(0) if v.dim() > 0 else 1 for v in values)
                            
                            # Pad all sequences to max length
                            padded_values = []
                            for v in values:
                                if v.dim() == 0:
                                    # Scalar tensor, convert to sequence
                                    padded = torch.full((max_len,), v.item(), dtype=v.dtype)
                                elif v.size(0) < max_len:
                                    # Pad sequence
                                    pad_size = max_len - v.size(0)
                                    if key == 'input_ids' or key == 'labels':
                                        # Pad with pad_token_id or -100 for labels
                                        pad_value = -100 if key == 'labels' else 0
                                        padded = torch.cat([v, torch.full((pad_size,), pad_value, dtype=v.dtype)])
                                    elif key == 'attention_mask':
                                        # Pad attention mask with 0s
                                        padded = torch.cat([v, torch.zeros(pad_size, dtype=v.dtype)])
                                    else:
                                        # Default padding with zeros
                                        padded = torch.cat([v, torch.zeros(pad_size, dtype=v.dtype)])
                                else:
                                    padded = v
                                padded_values.append(padded)
                            
                            result[key] = torch.stack(padded_values)
                        else:
                            # All scalars, just stack
                            result[key] = torch.stack(values)
                    else:
                        result[key] = values
                except Exception as e:
                    logger.warning(f"Failed to pad and stack key '{key}': {e}")
                    result[key] = values
            else:
                # Non-sequence keys or regular handling
                if all(v is not None for v in values):
                    try:
                        # Check if all values are tensors and can be stacked
                        if all(torch.is_tensor(v) for v in values):
                            # Ensure all tensors have the same shape for stackable keys
                            if key in ['vision_index', 'has_vision', 'index'] or all(v.shape == values[0].shape for v in values):
                                result[key] = torch.stack(values)
                            else:
                                # Different shapes, keep as list
                                result[key] = values
                        else:
                            # Mixed types or non-tensors, keep as list
                            result[key] = values
                    except Exception as e:
                        logger.warning(f"Failed to stack key '{key}': {e}")
                        result[key] = values
                else:
                    # Some values are None, filter them out or handle specially
                    filtered_values = [v for v in values if v is not None]
                    if filtered_values:
                        result[key] = filtered_values
        
        return result

    def setup_model_and_data(self):
        """Setup model and token-constrained data"""
        logger.info("Setting up model and token-constrained data...")

        # Clear any existing model artifacts to prevent dimension mismatches
        checkpoint_dir = Path(self.config['output']['checkpoint_dir'])
        if checkpoint_dir.exists():
            logger.info("Checkpoint directory exists - using fresh model initialization to avoid dimension conflicts")

        # Check if we should use token-constrained dataset
        if self.config.get('token_constraints') and TOKEN_CONSTRAINED_AVAILABLE:
            logger.info("🎯 Using token-constrained dataset for 100M tokens")
            # Pass the full data config including token constraints
            data_config = self.config['data'].copy()
            data_config['token_constraints'] = self.config['token_constraints']
            self.data_module = create_token_constrained_data_module(data_config)
        else:
            if self.config.get('token_constraints') and not TOKEN_CONSTRAINED_AVAILABLE:
                logger.warning("⚠️  Token constraints specified but token-constrained dataset not available")
            logger.info("📊 Using standard BabyLM dataset")
            self.data_module = create_data_module(self.config['data'])
        
        self.data_module.setup(rebuild_cache=getattr(self, 'rebuild_cache', False))

        # Override train_dataloader to use custom collate function
        original_train_dataloader = self.data_module.train_dataloader
        def custom_train_dataloader():
            from torch.utils.data import DataLoader
            
            # Get the dataset - handle different data module types
            if hasattr(self.data_module, 'train_dataset'):
                dataset = self.data_module.train_dataset
            elif hasattr(self.data_module, 'dataset'):
                dataset = self.data_module.dataset
            else:
                logger.error("No dataset found in data module")
                raise AttributeError("Data module has no dataset attribute")
            
            # Get data module attributes safely
            batch_size = getattr(self.data_module, 'batch_size', self.config['data']['batch_size'])
            num_workers = getattr(self.data_module, 'num_workers', self.config['data']['num_workers'])
            pin_memory = getattr(self.data_module, 'pin_memory', self.config['data'].get('pin_memory', True))
            persistent_workers = getattr(self.data_module, 'persistent_workers', self.config['data'].get('persistent_workers', True))
            
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers if num_workers > 0 else False,
                drop_last=True,
                collate_fn=self.custom_collate_fn
            )
        self.data_module.train_dataloader = custom_train_dataloader

        # Get and log token statistics if available
        if hasattr(self.data_module, 'get_token_statistics'):
            token_stats = self.data_module.get_token_statistics()
            logger.info("📊 Token Statistics:")
            for key, value in token_stats.items():
                if isinstance(value, int):
                    logger.info(f"  • {key}: {value:,}")
                else:
                    logger.info(f"  • {key}: {value}")

            # Verify token constraints
            if token_stats['total_tokens'] != self.target_tokens:
                logger.warning(f"Token mismatch: Expected {self.target_tokens:,}, got {token_stats['total_tokens']:,}")
        else:
            logger.warning("⚠️  Token statistics not available - using standard dataset")
            logger.info(f"Target tokens: {self.target_tokens:,}")

        # Create model
        logger.info("Creating BitMar model with updated configuration...")
        logger.info(f"Model config dimensions:")
        logger.info(f"  • text_encoder_dim: {self.config['model']['text_encoder_dim']}")
        logger.info(f"  • vision_latent_size: {self.config['model']['vision_latent_size']}")
        logger.info(f"  • fusion_hidden_size: {self.config['model']['fusion_hidden_size']}")
        
        self.model = create_bitmar_model(self.config['model'])

        # Apply attention sinks if enabled and available
        if ATTENTION_SINKS_AVAILABLE and self.config.get('attention_sinks', {}).get('enabled', False):
            try:
                logger.info("🔄 Applying attention sinks to BitMar model...")

                # Create attention sinks configuration
                attention_sinks_config = AttentionSinksConfig(
                    enable_attention_sinks=True,
                    attention_sink_size=self.config['attention_sinks'].get('attention_sink_size', 4),
                    attention_sink_window_size=self.config['attention_sinks'].get('attention_sink_window_size', 1020),
                    inject_to_text_encoder=self.config['attention_sinks'].get('inject_to_text_encoder', True),
                    inject_to_text_decoder=self.config['attention_sinks'].get('inject_to_text_decoder', True),
                    position_shift_enabled=self.config['attention_sinks'].get('position_shift_enabled', True)
                )

                # Apply attention sinks to the model
                self.model = apply_attention_sinks_to_bitmar_model(self.model, attention_sinks_config)

                # Get attention sinks statistics
                if hasattr(self.model, 'get_attention_sinks_stats'):
                    stats = self.model.get_attention_sinks_stats()
                    logger.info("✅ Attention Sinks successfully applied:")
                    logger.info(f"  • Attention sink size: {stats.get('attention_sink_size', 'N/A')}")
                    logger.info(f"  • Window size: {stats.get('attention_sink_window_size', 'N/A')}")
                    logger.info(f"  • Cache size: {stats.get('cache_size', 'N/A')}")
                    logger.info(f"  • Layers with sinks: {stats.get('layers_with_attention_sinks', 'N/A')}")

                    # Log to wandb if available
                    if self.use_wandb:
                        try:
                            wandb.log({
                                'attention_sinks/enabled': True,
                                'attention_sinks/sink_size': stats.get('attention_sink_size', 0),
                                'attention_sinks/window_size': stats.get('attention_sink_window_size', 0),
                                'attention_sinks/cache_size': stats.get('cache_size', 0),
                                'attention_sinks/layers_count': stats.get('layers_with_attention_sinks', 0)
                            })
                        except Exception as e:
                            logger.warning(f"Failed to log attention sinks stats to wandb: {e}")
                else:
                    logger.info("✅ Attention Sinks applied successfully")

            except Exception as e:
                logger.error(f"❌ Failed to apply attention sinks: {e}")
                logger.warning("⚠️  Continuing training without attention sinks")
        elif self.config.get('attention_sinks', {}).get('enabled', False):
            logger.warning("⚠️  Attention sinks enabled in config but integration not available")
        else:
            logger.info("📝 Attention sinks disabled in configuration")

        self.model.to(self.device)

        # Verify model is on correct device
        logger.info(f"🎮 Device Verification:")
        logger.info(f"  • Model device: {next(self.model.parameters()).device}")
        logger.info(f"  • Expected device: {self.device}")
        
        if self.device.type == 'cuda':
            logger.info(f"  • GPU memory before training: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
            logger.info(f"  • GPU memory reserved: {torch.cuda.memory_reserved(self.device) / 1024**3:.2f} GB")
            
            # Test a forward pass to ensure everything works on GPU
            try:
                test_input = torch.randn(1, 10, device=self.device)
                logger.info("  • GPU functionality test: ✅ Passed")
                del test_input
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"  • GPU functionality test: ❌ Failed - {e}")
                raise RuntimeError(f"GPU test failed: {e}")

        # Log model info
        param_count = count_parameters(self.model)
        logger.info(f"Model created with {param_count['total_parameters']:,} total parameters")
        logger.info(f"Trainable parameters: {param_count['trainable_parameters']:,}")
        logger.info(f"Non-trainable parameters: {param_count['non_trainable_parameters']:,}")

        # Setup optimizer with token-aware configuration
        self.setup_optimizer()

        # Initialize attention analyzer
        self.attention_analyzer = AttentionHeadAnalyzer(
            model=self.model,
            tokenizer=self.model.tokenizer,
            save_dir=str(self.attention_dir),
            wandb_logger=self.wandb_logger,
            track_top_k=self.config.get('attention_analysis', {}).get('track_top_k', 5)
        )

        # Setup adaptive training if enabled
        self.setup_adaptive_training()

        # Setup FLOPS tracking
        self.setup_flops_tracking()

    def setup_optimizer(self):
        """Setup optimizer and scheduler for 100M token training"""
        # Calculate total training steps based on exact token count
        train_loader = self.data_module.train_dataloader()
        
        # Estimate steps per epoch
        steps_per_epoch = len(train_loader)
        
        # Calculate how many steps we need to process exactly 100M tokens
        avg_tokens_per_batch = self.target_tokens // (steps_per_epoch * self.config['training']['max_epochs'])
        estimated_total_steps = self.target_tokens // avg_tokens_per_batch

        logger.info(f"Training planning:")
        logger.info(f"  • Steps per epoch: {steps_per_epoch}")
        logger.info(f"  • Estimated avg tokens per batch: {avg_tokens_per_batch}")
        logger.info(f"  • Estimated total steps: {estimated_total_steps}")

        # Create optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Create scheduler with restarts
        scheduler_config = self.config['training'].get('scheduler_config', {})
        
        # Validate scheduler parameters
        T_0 = int(scheduler_config.get('T_0', 2000))
        T_mult = scheduler_config.get('T_mult', 2)
        
        # Ensure T_mult is an integer >= 1
        if isinstance(T_mult, float):
            T_mult = max(1, int(T_mult))
            logger.warning(f"Converting T_mult from float to int: {T_mult}")
        elif not isinstance(T_mult, int) or T_mult < 1:
            T_mult = 2
            logger.warning(f"Invalid T_mult, using default: {T_mult}")
        
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=self.config['training']['learning_rate'] * scheduler_config.get('eta_min_ratio', 0.1)
        )
        
        logger.info(f"Scheduler configured: T_0={T_0}, T_mult={T_mult}")

        logger.info(f"✅ Optimizer and scheduler configured for 100M token training")

    def setup_adaptive_training(self):
        """Setup adaptive training controller"""
        if not ADAPTIVE_TRAINING_AVAILABLE or not self.config['model'].get('enable_adaptive_training', False):
            self.adaptive_controller = None
            return

        adaptive_config = self.config.get('adaptive_training', {})
        adaptive_logs_dir = Path("./logs/adaptive_training_100M")
        adaptive_logs_dir.mkdir(parents=True, exist_ok=True)

        self.adaptive_controller = AdaptiveTrainingController(
            similarity_window_size=adaptive_config.get('similarity_window_size', 200),
            drop_threshold=adaptive_config.get('drop_threshold', 0.12),
            min_steps_between_interventions=adaptive_config.get('min_steps_between_interventions', 800),
            freeze_duration_steps=adaptive_config.get('freeze_duration_steps', 1500),
            loss_rebalance_factor=adaptive_config.get('loss_rebalance_factor', 2.0),
            similarity_smoothing_alpha=adaptive_config.get('similarity_smoothing_alpha', 0.15),
            save_dir=str(adaptive_logs_dir)
        )

        logger.info("🤖 Adaptive training controller enabled for 100M token training")

    def setup_flops_tracking(self):
        """Setup FLOPS tracking system"""
        if not FLOPS_TRACKER_AVAILABLE:
            self.flops_tracker = None
            logger.warning("⚠️  FLOPS tracking not available")
            return

        try:
            # Get FLOPS tracking configuration
            flops_config = self.config.get('flops_tracking', {})
            log_frequency = flops_config.get('log_frequency', 100)

            # Create FLOPS logs directory
            flops_logs_dir = Path("./flops_logs_100M")
            flops_logs_dir.mkdir(parents=True, exist_ok=True)

            # Initialize FLOPS tracker
            self.flops_tracker = FLOPsTracker(
                model=self.model,
                log_frequency=log_frequency,
                save_dir=str(flops_logs_dir)
            )

            # Log model computational complexity
            self.flops_tracker.log_model_complexity()

            # Estimate theoretical FLOPS for the model
            batch_size = self.config['data']['batch_size']
            seq_length = self.config['model']['max_seq_len']

            # Estimate transformer FLOPS
            transformer_flops = FLOPsEstimator.estimate_transformer_flops(
                batch_size=batch_size,
                seq_length=seq_length,
                d_model=self.config['model']['text_encoder_dim'],
                num_layers=self.config['model']['text_encoder_layers'],
                num_heads=self.config['model']['text_encoder_heads'],
                vocab_size=self.config['model']['vocab_size']
            )

            # Estimate vision encoder FLOPS
            vision_flops = FLOPsEstimator.estimate_vision_encoder_flops(
                batch_size=batch_size,
                vision_dim=self.config['model']['vision_encoder_dim'],
                latent_dim=self.config['model']['vision_latent_size']
            )

            logger.info("🔢 FLOPS Tracker initialized:")
            logger.info(f"  • Log frequency: {log_frequency} steps")
            logger.info(f"  • Save directory: {flops_logs_dir}")
            logger.info("🔢 Theoretical FLOPS estimates per forward pass:")
            logger.info(f"  • Transformer: {self.flops_tracker._format_flops(transformer_flops['total_flops'])}")
            logger.info(f"  • Vision encoder: {self.flops_tracker._format_flops(vision_flops['total_flops'])}")
            logger.info(f"  • Total estimated: {self.flops_tracker._format_flops(transformer_flops['total_flops'] + vision_flops['total_flops'])}")

        except Exception as e:
            logger.warning(f"Failed to setup FLOPS tracking: {e}")
            self.flops_tracker = None

    def count_tokens_in_batch(self, batch: Dict) -> int:
        """Count actual tokens in a batch"""
        attention_mask = batch['attention_mask']
        return attention_mask.sum().item()

    def log_token_progress(self):
        """Log token consumption progress"""        
        logger.info(f"🎯 Tokens processed so far: {self.tokens_processed:,}")
        logger.info(f"   Dataset size: {self.target_tokens:,} tokens")
        
        # Log to wandb with error handling
        if self.use_wandb:
            try:
                wandb.log({
                    'token_progress/processed': self.tokens_processed,
                    'token_progress/target': self.target_tokens
                }, step=self.global_step)
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")
                # Disable wandb if it keeps failing
                self.use_wandb = False

    def save_token_checkpoint(self):
        """Save checkpoint with token information and automatic cleanup"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'tokens_processed': self.tokens_processed,
            'target_tokens': self.target_tokens,
            'best_similarity': self.best_similarity,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}_tokens_{self.tokens_processed}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)

        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

        # Cleanup old checkpoints - keep only top 5 most recent epoch checkpoints
        self.cleanup_old_checkpoints()

    def cleanup_old_checkpoints(self):
        """Keep only the 5 most recent epoch checkpoints and delete older ones"""
        try:
            # Get all epoch checkpoint files
            checkpoint_pattern = "checkpoint_epoch_*.pt"
            checkpoint_files = list(self.checkpoint_dir.glob(checkpoint_pattern))

            if len(checkpoint_files) <= 5:
                return  # No cleanup needed

            # Sort by modification time (newest first)
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Keep only the 5 most recent, delete the rest
            files_to_delete = checkpoint_files[5:]

            for old_checkpoint in files_to_delete:
                try:
                    old_checkpoint.unlink()  # Delete the file
                    logger.info(f"🗑️  Deleted old checkpoint: {old_checkpoint.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete checkpoint {old_checkpoint.name}: {e}")

            if files_to_delete:
                logger.info(f"✅ Cleanup completed: kept {min(5, len(checkpoint_files))} checkpoints, deleted {len(files_to_delete)}")

        except Exception as e:
            logger.warning(f"Failed to cleanup old checkpoints: {e}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with token awareness"""
        self.model.train()
        train_loader = self.data_module.train_dataloader()
        
        epoch_losses = []
        epoch_metrics = {
            'train_loss': 0.0,
            'cross_modal_similarity': 0.0,
            'tokens_in_epoch': 0
        }

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} | Tokens: {self.tokens_processed:,}")

        for batch_idx, batch in enumerate(progress_bar):
            # Count tokens in this batch for logging purposes
            batch_tokens = self.count_tokens_in_batch(batch)

            # Start FLOPS tracking for this step
            if self.flops_tracker:
                self.flops_tracker.start_step()

            try:
                # Move batch to device and ensure all required keys exist
                processed_batch = {}
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        processed_batch[k] = v.to(self.device)
                    elif isinstance(v, list):
                        # Handle list of tensors or mixed types
                        if all(torch.is_tensor(item) for item in v):
                            # Try to stack if all are tensors
                            try:
                                processed_batch[k] = torch.stack(v).to(self.device)
                            except:
                                # If stacking fails, use first item or create default
                                if k in ['vision_index', 'has_vision']:
                                    if k == 'vision_index':
                                        processed_batch[k] = torch.arange(len(v), device=self.device)
                                    else:  # has_vision
                                        processed_batch[k] = torch.ones(len(v), dtype=torch.bool, device=self.device)
                                else:
                                    processed_batch[k] = v[0].to(self.device) if v else None
                        else:
                            # Mixed types or non-tensors
                            processed_batch[k] = v
                    else:
                        processed_batch[k] = v
                
                batch = processed_batch
                
                # Ensure required keys exist and are properly formatted
                if 'vision_index' not in batch or not torch.is_tensor(batch['vision_index']):
                    batch['vision_index'] = torch.arange(batch['input_ids'].size(0), device=self.device)
                
                if 'has_vision' not in batch or not torch.is_tensor(batch['has_vision']):
                    batch['has_vision'] = torch.ones(batch['input_ids'].size(0), dtype=torch.bool, device=self.device)

                # Validate and potentially reshape vision features
                if 'vision_features' in batch:
                    vf_shape = batch['vision_features'].shape
                    logger.debug(f"Vision features shape: {vf_shape}")
                    
                    # Handle potential extra dimensions in vision features
                    if len(vf_shape) == 3 and vf_shape[1] == 1:  # [batch, 1, 768]
                        logger.debug("Removing singleton dimension from vision features")
                        batch['vision_features'] = batch['vision_features'].squeeze(1)  # [batch, 768]
                        logger.debug(f"Reshaped vision features: {batch['vision_features'].shape}")
                    elif len(vf_shape) == 3 and vf_shape[1] != 1:  # [batch, N, 768] where N > 1
                        logger.debug("Flattening multi-dimensional vision features")
                        batch['vision_features'] = batch['vision_features'].view(vf_shape[0], -1)  # [batch, N*768]
                        # Take only first 768 features if we have more
                        if batch['vision_features'].size(1) > 768:
                            batch['vision_features'] = batch['vision_features'][:, :768]
                        logger.debug(f"Reshaped vision features: {batch['vision_features'].shape}")
                    elif len(vf_shape) == 2:  # [batch, 768] - already correct
                        logger.debug("Vision features shape is correct")
                    else:
                        logger.warning(f"Unexpected vision features shape: {vf_shape}")
                        # Try to flatten to [batch, 768]
                        batch['vision_features'] = batch['vision_features'].view(vf_shape[0], -1)
                        if batch['vision_features'].size(1) != 768:
                            if batch['vision_features'].size(1) > 768:
                                batch['vision_features'] = batch['vision_features'][:, :768]
                            else:
                                # Pad with zeros if too small
                                pad_size = 768 - batch['vision_features'].size(1)
                                batch['vision_features'] = torch.cat([
                                    batch['vision_features'], 
                                    torch.zeros(vf_shape[0], pad_size, device=batch['vision_features'].device)
                                ], dim=1)
                        logger.debug(f"Normalized vision features: {batch['vision_features'].shape}")
                
                # Forward pass with detailed error tracking
                try:
                    logger.debug(f"Starting forward pass for step {self.global_step}")
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        vision_features=batch['vision_features'],
                        labels=batch['labels'],
                        step=self.global_step,
                        has_vision=batch.get('has_vision', torch.ones(batch['input_ids'].size(0), dtype=torch.bool)),
                        adaptive_controller=self.adaptive_controller
                    )
                    logger.debug(f"Forward pass completed successfully for step {self.global_step}")
                except Exception as forward_error:
                    logger.error(f"Forward pass failed at step {self.global_step}: {forward_error}")
                    logger.error(f"Error type: {type(forward_error).__name__}")
                    logger.error(f"Error details: {str(forward_error)}")
                    
                    # Log model architecture info for debugging
                    logger.error(f"Model architecture details:")
                    if hasattr(self.model, 'text_encoder'):
                        logger.error(f"  • Text encoder dim: {self.model.text_encoder.dim}")
                    if hasattr(self.model, 'vision_encoder'):
                        logger.error(f"  • Vision encoder output: {getattr(self.model.vision_encoder, 'output_proj', None)}")
                    if hasattr(self.model, 'fusion'):
                        logger.error(f"  • Fusion hidden dim: {self.model.fusion.hidden_dim}")
                    
                    raise forward_error

                loss = outputs['loss']

                # Check for valid loss
                if not torch.isfinite(loss):
                    logger.warning(f"Invalid loss at step {self.global_step}: {loss.item()}")
                    continue

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.config['training']['gradient_clip_val'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip_val']
                    )

                self.optimizer.step()
                self.scheduler.step()

                # End FLOPS tracking and get metrics
                flops_metrics = None
                if self.flops_tracker:
                    try:
                        flops_metrics = self.flops_tracker.end_step(
                            batch_size=batch['input_ids'].size(0),
                            sequence_length=batch['input_ids'].size(1)
                        )

                        # Log FLOPS periodically
                        if self.flops_tracker.should_log():
                            self.flops_tracker.log_flops(
                                metrics=flops_metrics,
                                logger_func=logger.info,
                                wandb_logger=wandb if self.use_wandb else None,
                                step=self.global_step
                            )
                    except Exception as e:
                        logger.warning(f"FLOPS tracking failed: {e}")

                # Update token count
                self.tokens_processed += batch_tokens
                epoch_metrics['tokens_in_epoch'] += batch_tokens

                # Update metrics
                epoch_losses.append(loss.item())

                # Compute cross-modal similarity if available
                if outputs.get('text_features') is not None and outputs.get('vision_latent') is not None:
                    try:
                        similarity = self._compute_cross_modal_similarity(
                            outputs['text_features'], outputs['vision_latent']
                        )
                        epoch_metrics['cross_modal_similarity'] += similarity
                        
                        # Update best similarity
                        if similarity > self.best_similarity:
                            self.best_similarity = similarity
                    except Exception as e:
                        logger.warning(f"Cross-modal similarity computation failed: {e}")

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'tokens': f"{self.tokens_processed:,}",
                    'epoch': f"{self.current_epoch + 1}/{self.config['training']['max_epochs']}"
                })

                # Log token progress periodically
                if self.global_step % self.token_log_frequency == 0:
                    self.log_token_progress()

                # Enhanced wandb logging
                if self.use_wandb and self.global_step % 100 == 0:
                    # Use comprehensive WandB logging instead of basic logging
                    if self.wandb_logger:
                        try:
                            # Log comprehensive metrics including quantization
                            self.wandb_logger.log_consolidated_metrics(
                                outputs=outputs,
                                epoch=epoch,
                                step=self.global_step,
                                lr=self.optimizer.param_groups[0]['lr'],
                                model=self.model
                            )

                            # Also log token-specific metrics
                            wandb.log({
                                'tokens/processed': self.tokens_processed,
                                'tokens/batch_size': batch_tokens,
                                'step': self.global_step
                            }, step=self.global_step)

                        except Exception as e:
                            logger.warning(f"Failed to log comprehensive metrics to wandb: {e}")
                            # Fallback to basic logging
                            log_dict = {
                                'train/loss': loss.item(),
                                'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                                'tokens/processed': self.tokens_processed,
                                'tokens/batch_size': batch_tokens,
                                'step': self.global_step
                            }

                            # Only add similarity if it was computed
                            if outputs.get('text_features') is not None and outputs.get('vision_latent') is not None:
                                try:
                                    current_similarity = self._compute_cross_modal_similarity(
                                        outputs['text_features'], outputs['vision_latent']
                                    )
                                    log_dict['train/cross_modal_similarity'] = current_similarity
                                except Exception as e:
                                    logger.warning(f"Failed to compute similarity for wandb: {e}")

                            try:
                                wandb.log(log_dict, step=self.global_step)
                            except Exception as e:
                                logger.warning(f"Failed to log to wandb during training: {e}")
                                self.use_wandb = False
                    else:
                        # Fallback when wandb_logger is not available
                        log_dict = {
                            'train/loss': loss.item(),
                            'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                            'tokens/processed': self.tokens_processed,
                            'tokens/batch_size': batch_tokens,
                            'step': self.global_step
                        }

                        # Only add similarity if it was computed
                        if outputs.get('text_features') is not None and outputs.get('vision_latent') is not None:
                            try:
                                current_similarity = self._compute_cross_modal_similarity(
                                    outputs['text_features'], outputs['vision_latent']
                                )
                                log_dict['train/cross_modal_similarity'] = current_similarity
                            except Exception as e:
                                logger.warning(f"Failed to compute similarity for wandb: {e}")

                        try:
                            wandb.log(log_dict, step=self.global_step)
                        except Exception as e:
                            logger.warning(f"Failed to log to wandb during training: {e}")
                            self.use_wandb = False

                self.global_step += 1

                # Save checkpoint based on step frequency if specified
                if hasattr(self, 'save_every_n_steps') and self.save_every_n_steps is not None:
                    if self.global_step % self.save_every_n_steps == 0:
                        logger.info(f"💾 Saving step-based checkpoint at step {self.global_step}")
                        self.save_step_checkpoint()

                # Save checkpoint periodically (default behavior)
                if self.global_step % 5000 == 0:
                    self.save_token_checkpoint()

            except Exception as e:
                logger.error(f"Training step failed at step {self.global_step}: {e}")
                
                # Enhanced error logging for tensor size mismatches
                if "size of tensor" in str(e) and "must match" in str(e):
                    logger.error(f"Tensor size mismatch details:")
                    logger.error(f"  Batch shapes:")
                    for k, v in batch.items():
                        if torch.is_tensor(v):
                            logger.error(f"    {k}: {v.shape}")
                    logger.error(f"  Model config:")
                    logger.error(f"    Max seq length: {self.config['model']['max_seq_len']}")
                
                # Clear any gradients and free memory
                if hasattr(self, 'optimizer'):
                    self.optimizer.zero_grad()
                
                # Clear CUDA cache if using GPU
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                continue

        # Calculate epoch metrics
        if epoch_losses:
            epoch_metrics['train_loss'] = np.mean(epoch_losses)
            epoch_metrics['cross_modal_similarity'] = epoch_metrics['cross_modal_similarity'] / len(epoch_losses)
        
        logger.info(f"Epoch {epoch} completed:")
        logger.info(f"  • Loss: {epoch_metrics['train_loss']:.4f}")
        logger.info(f"  • Cross-modal similarity: {epoch_metrics['cross_modal_similarity']:.4f}")
        logger.info(f"  • Tokens in epoch: {epoch_metrics['tokens_in_epoch']:,}")
        logger.info(f"  • Total tokens processed: {self.tokens_processed:,}")

        return epoch_metrics

    def _compute_cross_modal_similarity(self, text_features: torch.Tensor, vision_features: torch.Tensor) -> float:
        """Compute cross-modal similarity"""
        try:
            # Pool text features if needed
            if text_features.dim() == 3:  # [batch, seq, dim]
                text_pooled = text_features.mean(dim=1)  # [batch, dim]
            else:
                text_pooled = text_features

            # Ensure same dimensions
            if text_pooled.size(-1) != vision_features.size(-1):
                min_dim = min(text_pooled.size(-1), vision_features.size(-1))
                text_pooled = text_pooled[:, :min_dim]
                vision_features = vision_features[:, :min_dim]

            # Compute cosine similarity
            cos_sim = torch.cosine_similarity(text_pooled, vision_features, dim=1)
            return cos_sim.mean().item()
        except Exception as e:
            logger.warning(f"Cross-modal similarity computation failed: {e}")
            return 0.0

    def train(self):
        """Main training loop with token awareness"""
        logger.info("🚀 Starting 100M token training...")
        
        # Start carbon tracking
        if self.carbon_tracker:
            self.carbon_tracker.start()

        # Setup model and data
        self.setup_model_and_data()

        try:
            for epoch in range(self.config['training']['max_epochs']):
                logger.info(f"Starting epoch {epoch + 1}/{self.config['training']['max_epochs']}")
                
                self.current_epoch = epoch
                epoch_metrics = self.train_epoch(epoch)

                # Save checkpoint after each epoch
                self.save_token_checkpoint()

                # Upload checkpoint to Hugging Face Hub after each epoch
                if self.hf_hub_enabled and self.hf_upload_after_epoch:
                    self.upload_checkpoint_to_hf(epoch)

                # Run fast evaluation after each epoch if enabled
                if hasattr(self, 'enable_fast_eval') and self.enable_fast_eval:
                    self.run_fast_evaluation_after_epoch(epoch)

                # Log epoch summary to wandb with error handling
                if self.use_wandb:
                    try:
                        wandb.log({
                            'epoch/train_loss': epoch_metrics['train_loss'],
                            'epoch/cross_modal_similarity': epoch_metrics['cross_modal_similarity'],
                            'epoch/tokens_processed': self.tokens_processed,
                            'epoch/tokens_in_epoch': epoch_metrics['tokens_in_epoch'],
                            'epoch/number': epoch
                        }, step=self.global_step)
                    except Exception as e:
                        logger.warning(f"Failed to log epoch summary to wandb: {e}")
                        self.use_wandb = False

                # Continue training for all epochs (no token limit stopping)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            # Stop carbon tracking
            if self.carbon_tracker:
                emissions = self.carbon_tracker.stop()
                logger.info(f"🌱 Carbon emissions: {emissions:.6f} kg CO2")

            # Final checkpoint
            self.save_token_checkpoint()

            # Upload final model to Hugging Face Hub
            if self.hf_hub_enabled and self.hf_upload_final_model:
                self.upload_checkpoint_to_hf(self.current_epoch, final=True)

            # Run full evaluation at the end if enabled
            if hasattr(self, 'enable_full_eval') and self.enable_full_eval:
                self.run_full_evaluation_final()

            # Final FLOPS summary and cleanup
            if self.flops_tracker:
                try:
                    # Generate final FLOPS statistics
                    final_stats = self.flops_tracker.get_summary_stats()
                    logger.info("🔢 Final FLOPS Summary:")
                    logger.info(f"  • Total FLOPS: {final_stats.get('flops_formatted', 'N/A')}")
                    logger.info(f"  • Total training time: {final_stats.get('total_time', 0):.1f}s")
                    logger.info(f"  • Average FLOPS/step: {self.flops_tracker._format_flops(final_stats.get('avg_flops_per_step', 0))}")
                    logger.info(f"  • Average throughput: {final_stats.get('avg_throughput_formatted', 'N/A')}")
                    logger.info(f"  • Peak throughput: {self.flops_tracker._format_flops(final_stats.get('peak_throughput', 0))}/s")

                    # Save FLOPS statistics
                    self.flops_tracker.save_statistics("final_flops_statistics.json")

                    # Log to wandb if available
                    if self.use_wandb:
                        try:
                            wandb.log({
                                'final_flops/total_flops': final_stats.get('total_flops', 0),
                                'final_flops/avg_flops_per_step': final_stats.get('avg_flops_per_step', 0),
                                'final_flops/avg_throughput': final_stats.get('avg_throughput', 0),
                                'final_flops/peak_throughput': final_stats.get('peak_throughput', 0),
                                'final_flops/total_time': final_stats.get('total_time', 0)
                            })
                        except Exception as e:
                            logger.warning(f"Failed to log final FLOPS to wandb: {e}")

                    # Cleanup FLOPS tracker
                    self.flops_tracker.cleanup()
                    logger.info("✅ FLOPS tracking completed and cleaned up")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to complete FLOPS tracking: {e}")

            # Final token summary
            logger.info("🎯 Final Token Summary:")
            logger.info(f"  • Target tokens: {self.target_tokens:,}")
            logger.info(f"  • Processed tokens: {self.tokens_processed:,}")
            logger.info(f"  • Completion: {(self.tokens_processed/self.target_tokens)*100:.2f}%")
            logger.info(f"  • Best cross-modal similarity: {self.best_similarity:.4f}")

            if self.use_wandb:
                try:
                    wandb.finish()
                except Exception as e:
                    logger.warning(f"Failed to finish wandb run: {e}")

    def run_fast_evaluation_after_epoch(self, epoch: int):
        """Run fast evaluation after completing an epoch"""
        try:
            logger.info(f"🧪 Running fast evaluation after epoch {epoch}")

            # Get the checkpoint path for this epoch
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}_tokens_{self.tokens_processed}.pt'

            # Import and run evaluation scripts
            import subprocess
            import sys

            eval_results_dir = Path("evaluation_results") / f"epoch_{epoch}"
            eval_results_dir.mkdir(parents=True, exist_ok=True)

            eval_success = {'2025': False, '2024': False}

            # Run 2025 pipeline (text + multimodal fast evaluation)
            eval_2025_path = Path("../evaluation-pipeline-2025")
            if eval_2025_path.exists():
                try:
                    cmd = [
                        sys.executable, "evaluate_bitmar_2025.py",
                        "--model_path", str(checkpoint_path),
                        "--eval_type", "fast",
                        "--evaluation_pipeline_path", str(eval_2025_path),
                        "--output_dir", str(eval_results_dir / "2025_results")
                    ]

                    logger.info("Running 2025 pipeline fast evaluation...")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

                    if result.returncode == 0:
                        logger.info("✅ 2025 pipeline evaluation completed successfully")
                        eval_success['2025'] = True
                    else:
                        logger.warning(f"⚠️ 2025 pipeline evaluation failed: {result.stderr}")

                except Exception as e:
                    logger.warning(f"⚠️ 2025 pipeline evaluation error: {e}")
            else:
                logger.info("⚠️ 2025 evaluation pipeline not found, skipping")

            # Run 2024 pipeline (multimodal only fast evaluation)
            eval_2024_path = Path("../evaluation-pipeline-2024")
            if eval_2024_path.exists():
                try:
                    cmd = [
                        sys.executable, "evaluate_bitmar_2024.py",
                        "--model_path", str(checkpoint_path),
                        "--eval_type", "fast",
                        "--evaluation_pipeline_path", str(eval_2024_path),
                        "--output_dir", str(eval_results_dir / "2024_results")
                    ]

                    logger.info("Running 2024 pipeline fast evaluation...")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

                    if result.returncode == 0:
                        logger.info("✅ 2024 pipeline evaluation completed successfully")
                        eval_success['2024'] = True
                    else:
                        logger.warning(f"⚠️ 2024 pipeline evaluation failed: {result.stderr}")

                except Exception as e:
                    logger.warning(f"⚠️ 2024 pipeline evaluation error: {e}")
            else:
                logger.info("⚠️ 2024 evaluation pipeline not found, skipping")

            # Log evaluation success to wandb
            if self.use_wandb:
                try:
                    wandb.log({
                        f'epoch_{epoch}/eval_2025_success': eval_success['2025'],
                        f'epoch_{epoch}/eval_2024_success': eval_success['2024'],
                    }, step=self.global_step)
                except Exception as e:
                    logger.warning(f"Failed to log evaluation results to wandb: {e}")

            logger.info(f"📊 Fast evaluation completed for epoch {epoch}")
            logger.info(f"  • 2025 pipeline: {'✅' if eval_success['2025'] else '❌'}")
            logger.info(f"  • 2024 pipeline: {'✅' if eval_success['2024'] else '❌'}")

        except Exception as e:
            logger.error(f"❌ Fast evaluation failed for epoch {epoch}: {e}")

    def run_full_evaluation_final(self):
        """Run full evaluation on the final model"""
        try:
            logger.info("🧪 Running full evaluation on final model")

            # Get the final checkpoint path
            final_checkpoint = self.checkpoint_dir / 'latest_checkpoint.pt'

            import subprocess
            import sys

            eval_results_dir = Path("evaluation_results") / "final"
            eval_results_dir.mkdir(parents=True, exist_ok=True)

            eval_success = {'2025': False, '2024': False}

            # Run 2025 pipeline (text + multimodal full evaluation)
            eval_2025_path = Path("../evaluation-pipeline-2025")
            if eval_2025_path.exists():
                try:
                    cmd = [
                        sys.executable, "evaluate_bitmar_2025.py",
                        "--model_path", str(final_checkpoint),
                        "--eval_type", "full",
                        "--evaluation_pipeline_path", str(eval_2025_path),
                        "--output_dir", str(eval_results_dir / "2025_results")
                    ]

                    logger.info("Running 2025 pipeline full evaluation...")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hours

                    if result.returncode == 0:
                        logger.info("✅ 2025 pipeline full evaluation completed successfully")
                        eval_success['2025'] = True
                    else:
                        logger.warning(f"⚠️ 2025 pipeline full evaluation failed: {result.stderr}")

                except Exception as e:
                    logger.warning(f"⚠️ 2025 pipeline full evaluation error: {e}")
            else:
                logger.info("⚠️ 2025 evaluation pipeline not found, skipping")

            # Run 2024 pipeline (multimodal only full evaluation)
            eval_2024_path = Path("../evaluation-pipeline-2024")
            if eval_2024_path.exists():
                try:
                    cmd = [
                        sys.executable, "evaluate_bitmar_2024.py",
                        "--model_path", str(final_checkpoint),
                        "--eval_type", "full",
                        "--evaluation_pipeline_path", str(eval_2024_path),
                        "--output_dir", str(eval_results_dir / "2024_results")
                    ]

                    logger.info("Running 2024 pipeline full evaluation...")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hours

                    if result.returncode == 0:
                        logger.info("✅ 2024 pipeline full evaluation completed successfully")
                        eval_success['2024'] = True
                    else:
                        logger.warning(f"⚠️ 2024 pipeline full evaluation failed: {result.stderr}")

                except Exception as e:
                    logger.warning(f"⚠️ 2024 pipeline full evaluation error: {e}")
            else:
                logger.info("⚠️ 2024 evaluation pipeline not found, skipping")

            # Log final evaluation success to wandb
            if self.use_wandb:
                try:
                    wandb.log({
                        'final/eval_2025_success': eval_success['2025'],
                        'final/eval_2024_success': eval_success['2024'],
                    })
                except Exception as e:
                    logger.warning(f"Failed to log final evaluation results to wandb: {e}")

            logger.info("📊 Full evaluation completed on final model")
            logger.info(f"  • 2025 pipeline: {'✅' if eval_success['2025'] else '❌'}")
            logger.info(f"  • 2024 pipeline: {'✅' if eval_success['2024'] else '❌'}")

        except Exception as e:
            logger.error(f"❌ Full evaluation failed: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train BitMar with exactly 100M tokens")
    
    parser.add_argument("--config", type=str, default="configs/bitmar_100M_tokens.yaml",
                       help="Path to configuration file")
    parser.add_argument("--device", type=str, help="Device to use (cuda:0, cpu)")
    parser.add_argument("--rebuild_cache", action="store_true",
                       help="Rebuild token-constrained dataset cache")
    parser.add_argument("--save_every_n_steps", type=int, default=None,
                       help="Save checkpoint every N training steps (optional)")
    parser.add_argument("--enable_fast_eval", action="store_true",
                       help="Enable fast evaluation after each epoch")
    parser.add_argument("--enable_full_eval", action="store_true",
                       help="Enable full evaluation at the end")
    parser.add_argument("--disable_fast_eval", action="store_true",
                       help="Disable fast evaluation after each epoch")
    parser.add_argument("--disable_full_eval", action="store_true",
                       help="Disable full evaluation at the end")

    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = TokenAwareTrainer(args.config, device=args.device)
        trainer.rebuild_cache = args.rebuild_cache  # Pass rebuild_cache to trainer
        trainer.save_every_n_steps = args.save_every_n_steps  # Pass step-based saving option

        # Set evaluation flags (default to True unless explicitly disabled)
        # Check environment variables first (for bash script compatibility)
        env_fast_eval = os.getenv('BITMAR_ENABLE_FAST_EVAL', 'true').lower() == 'true'
        env_full_eval = os.getenv('BITMAR_ENABLE_FULL_EVAL', 'true').lower() == 'true'

        # Command line arguments override environment variables
        if args.disable_fast_eval:
            trainer.enable_fast_eval = False
        elif args.enable_fast_eval:
            trainer.enable_fast_eval = True
        else:
            trainer.enable_fast_eval = env_fast_eval

        if args.disable_full_eval:
            trainer.enable_full_eval = False
        elif args.enable_full_eval:
            trainer.enable_full_eval = True
        else:
            trainer.enable_full_eval = env_full_eval

        logger.info(f"🧪 Evaluation settings:")
        logger.info(f"  • Fast evaluation (after epochs): {trainer.enable_fast_eval}")
        logger.info(f"  • Full evaluation (at end): {trainer.enable_full_eval}")

        # Start training
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
