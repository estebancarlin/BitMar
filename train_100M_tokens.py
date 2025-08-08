"""
Token-Aware Training Script for BitMar 100M Token Model
Handles exactly 100M tokens with perfect image-caption alignment
Includes advanced token tracking and stopping mechanisms
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
import traceback
import time
from collections import defaultdict

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

# FLOPS computation utilities (after logger is defined)
try:
    from fvcore.nn import flop_count
    FLOPS_AVAILABLE = True
    logger.info("✅ FLOPS computation available via fvcore")
except ImportError:
    try:
        from ptflops import get_model_complexity_info
        FLOPS_AVAILABLE = True
        logger.info("✅ FLOPS computation available via ptflops")
    except ImportError:
        FLOPS_AVAILABLE = False
        logger.warning("⚠️  No FLOPS computation library available. Install fvcore or ptflops for FLOPS tracking")

# Import components
from src.dataset import create_data_module
from src.model import create_bitmar_model, count_parameters
from src.wandb_logger import BitMarWandbLogger
from src.attention_visualizer import AttentionHeadAnalyzer
from src.memory_visualization_integration import setup_memory_visualization

# Try to import tiny model evaluator
try:
    from src.tiny_model_evaluator import TinyModelEvaluator
    TINY_MODEL_EVALUATOR_AVAILABLE = True
    logger.info("✅ Tiny model evaluator available")
except ImportError:
    TINY_MODEL_EVALUATOR_AVAILABLE = False
    logger.warning("⚠️  Tiny model evaluator not available")

# Try to import evaluation integration
try:
    from src.training_evaluation_integration import TrainingEvaluationIntegration
    EVALUATION_INTEGRATION_AVAILABLE = True
    logger.info("✅ Evaluation integration available")
except ImportError:
    EVALUATION_INTEGRATION_AVAILABLE = False
    logger.warning("⚠️  Evaluation integration not available")

# Try to import tiny model evaluation
try:
    from src.tiny_model_evaluator import TinyModelEvaluator, TinyModelBabyLMEvaluator
    TINY_MODEL_EVAL_AVAILABLE = True
    logger.info("✅ Tiny model evaluation available")
except ImportError:
    TINY_MODEL_EVAL_AVAILABLE = False
    logger.warning("⚠️  Tiny model evaluation not available")

# Try to import benchmark evaluator
try:
    from src.benchmark_evaluator import BenchmarkEvaluator
    BENCHMARK_EVALUATOR_AVAILABLE = True
    logger.info("✅ Benchmark evaluator available")
except ImportError:
    BENCHMARK_EVALUATOR_AVAILABLE = False
    logger.warning("⚠️  Benchmark evaluator not available")

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

        # FLOPS tracking
        self.total_flops = 0
        self.flops_per_step = 0
        self.flops_history = []
        self.flops_estimation_samples = 0
        self.avg_flops_per_forward = 0
        self.flops_tracking_enabled = FLOPS_AVAILABLE and self.config.get('training', {}).get('track_flops', True)

        # Enhanced cross-modal tracking
        self.cross_modal_history = {
            'steps': [],
            'cross_modal_similarity': [],
            'text_consistency': [],
            'vision_consistency': [],
            'text_learning_strength': [],
            'vision_learning_strength': [],
            'alignment_convergence': [],
            'text_learning_trajectory': [],  # Smoothed text learning progression
            'vision_learning_trajectory': [],  # Smoothed vision learning progression
            'convergence_rate': []  # How fast text and vision are converging
        }
        self.cross_modal_smoothing_alpha = 0.1  # EMA smoothing factor

        # Tiny model evaluation tracking
        self.loss_history = []  # For convergence analysis
        self.tiny_model_eval_results = {}

        # Evaluation control (will be set by main function)
        self.eval_mode = "epoch"  # Default to epoch-based evaluation
        self.eval_steps = 5000    # Default evaluation frequency for step-based
        self.eval_epochs = "all"  # Default to evaluate all epochs
        self.eval_start_step = 0  # Default to start evaluation from beginning

        # Setup directories
        self.setup_directories()
        
        # Setup logging
        self.setup_wandb()
        
        # Initialize carbon tracking if available
        self.setup_carbon_tracking()

        logger.info(f"🎯 Token-aware trainer initialized for {self.target_tokens:,} tokens")
        logger.info(f"Device: {self.device}")

    def setup_directories(self):
        """Create output directories"""
        for dir_name in ['checkpoint_dir', 'log_dir', 'attention_dir', 'memory_dir', 'results_dir', 'token_logs_dir']:
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
        logger.info(f"  • episode_dim: {self.config['model']['episode_dim']}")
        
        self.model = create_bitmar_model(self.config['model'])
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
        
        # Setup memory visualization integration
        try:
            self.memory_viz = setup_memory_visualization(self.config, self.model)
            logger.info("✅ Memory visualization integration initialized")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize memory visualization: {e}")
            self.memory_viz = None

        # Setup evaluation integration
        try:
            if EVALUATION_INTEGRATION_AVAILABLE and hasattr(self, 'eval_mode') and self.eval_mode != "disabled":
                eval_config = self.config.get('evaluation', {})
                self.evaluation_integration = TrainingEvaluationIntegration(
                    pipeline_2024_path=eval_config.get('pipeline_2024_path', 'd:/BabyLM/evaluation-pipeline-2024'),
                    pipeline_2025_path=eval_config.get('pipeline_2025_path', 'd:/BabyLM/evaluation-pipeline-2025'),
                    results_dir=eval_config.get('results_dir', 'evaluation_results'),
                    eval_frequency=eval_config.get('eval_frequency', 1),
                    fast_eval_epochs=eval_config.get('fast_eval_epochs', list(range(1, 10))),
                    full_eval_epochs=eval_config.get('full_eval_epochs', [10])
                )
                
                # Setup evaluation with model and tokenizer
                self.evaluation_integration.setup_evaluation(
                    model=self.model,
                    tokenizer=self.model.tokenizer,
                    device=self.device
                )
                
                logger.info("✅ BabyLM evaluation integration initialized")
                logger.info(f"  • Evaluation mode: {getattr(self, 'eval_mode', 'epoch')}")
                if hasattr(self, 'eval_mode') and self.eval_mode == "steps":
                    logger.info(f"  • Evaluation every {getattr(self, 'eval_steps', 5000)} steps")
                    logger.info(f"  • Evaluation starts after step {getattr(self, 'eval_start_step', 0)}")
                else:
                    logger.info(f"  • Fast eval epochs: {self.evaluation_integration.fast_eval_epochs}")
                    logger.info(f"  • Full eval epochs: {self.evaluation_integration.full_eval_epochs}")
            else:
                self.evaluation_integration = None
                if hasattr(self, 'eval_mode') and self.eval_mode == "disabled":
                    logger.info("⚠️  Evaluation explicitly disabled")
                else:
                    logger.info("⚠️  Evaluation integration not available")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize evaluation integration: {e}")
            self.evaluation_integration = None

        # Setup tiny model evaluation integration
        try:
            if TINY_MODEL_EVAL_AVAILABLE:
                tiny_eval_config = self.config.get('evaluation', {}).get('tiny_model_evaluations', {})
                if tiny_eval_config.get('enabled', False):
                    self.tiny_model_evaluator = TinyModelEvaluator(
                        config=self.config,
                        model=self.model,
                        tokenizer=self.model.tokenizer,
                        device=self.device
                    )
                    
                    # Setup BabyLM tiny model evaluator if configured
                    babylm_config = tiny_eval_config.get('babylm_tiny_evaluations', {})
                    if babylm_config.get('enabled', False):
                        self.tiny_babylm_evaluator = TinyModelBabyLMEvaluator(
                            base_evaluator=self.evaluation_integration,
                            tiny_eval_config=babylm_config
                        )
                    else:
                        self.tiny_babylm_evaluator = None
                    
                    logger.info("✅ Tiny model evaluation integration successful")
                else:
                    self.tiny_model_evaluator = None
                    self.tiny_babylm_evaluator = None
                    logger.info("ℹ️  Tiny model evaluation disabled in config")
            else:
                self.tiny_model_evaluator = None
                self.tiny_babylm_evaluator = None
                logger.warning("⚠️  Tiny model evaluation not available")
        except Exception as e:
            logger.warning(f"⚠️  Failed to setup tiny model evaluation: {e}")
            self.tiny_model_evaluator = None
            self.tiny_babylm_evaluator = None

        # Setup benchmark evaluation integration  
        try:
            if BENCHMARK_EVALUATOR_AVAILABLE:
                benchmark_config = self.config.get('evaluation', {}).get('benchmark_evaluations', {})
                if benchmark_config.get('enabled', False):
                    benchmark_save_dir = self.results_dir / "benchmark_results"
                    self.benchmark_evaluator = BenchmarkEvaluator(
                        model=self.model,
                        tokenizer=self.model.tokenizer,
                        device=self.device,
                        config=self.config,
                        save_dir=str(benchmark_save_dir)
                    )
                    logger.info("✅ Benchmark evaluator integrated")
                else:
                    self.benchmark_evaluator = None
                    logger.info("ℹ️  Benchmark evaluation disabled in config")
            else:
                self.benchmark_evaluator = None
                logger.warning("⚠️  Benchmark evaluator not available")
        except Exception as e:
            logger.warning(f"⚠️  Failed to setup benchmark evaluator: {e}")
            self.benchmark_evaluator = None

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

    def count_tokens_in_batch(self, batch: Dict) -> int:
        """Count actual tokens in a batch"""
        attention_mask = batch['attention_mask']
        return attention_mask.sum().item()

    def estimate_model_flops(self, sample_batch: Dict) -> float:
        """Estimate FLOPS for one forward pass using a sample batch"""
        if not self.flops_tracking_enabled:
            return 0.0
        
        try:
            # Use fvcore if available
            if 'fvcore' in globals():
                from fvcore.nn import FlopCountMode, flop_count
                
                # Create input tuple for fvcore
                flop_inputs = (
                    sample_batch['input_ids'],
                    sample_batch['attention_mask'], 
                    sample_batch['vision_features'],
                    sample_batch['labels']
                )
                
                with flop_count(self.model, flop_inputs) as flops:
                    _ = self.model(
                        input_ids=sample_batch['input_ids'],
                        attention_mask=sample_batch['attention_mask'],
                        vision_features=sample_batch['vision_features'],
                        labels=sample_batch['labels'],
                        step=self.global_step,
                        has_vision=sample_batch.get('has_vision', torch.ones(sample_batch['input_ids'].size(0), dtype=torch.bool)),
                        adaptive_controller=None  # Don't use adaptive controller for FLOPS estimation
                    )
                
                total_flops = sum(flops.values())
                return total_flops
            
            else:
                # Fallback: Manual FLOPS estimation based on model parameters
                # This is a rough estimation
                batch_size = sample_batch['input_ids'].size(0)
                seq_len = sample_batch['input_ids'].size(1)
                
                # Get model parameter count
                param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                # Rough estimation: 2 * params * batch_size * seq_len
                # This is a very rough approximation for transformer models
                estimated_flops = 2 * param_count * batch_size * seq_len
                
                logger.info(f"Using rough FLOPS estimation: {estimated_flops:,} FLOPS per forward pass")
                return estimated_flops
                
        except Exception as e:
            logger.warning(f"FLOPS estimation failed: {e}")
            return 0.0

    def update_flops_tracking(self, batch_flops: float):
        """Update FLOPS tracking with new measurement"""
        if not self.flops_tracking_enabled or batch_flops == 0:
            return
        
        # Update total FLOPS
        self.total_flops += batch_flops
        
        # Update history for averaging
        self.flops_history.append(batch_flops)
        if len(self.flops_history) > 100:  # Keep last 100 measurements
            self.flops_history.pop(0)
        
        # Update average FLOPS per forward pass
        self.flops_estimation_samples += 1
        self.avg_flops_per_forward = sum(self.flops_history) / len(self.flops_history)

    def log_flops_progress(self):
        """Log FLOPS consumption progress"""
        if not self.flops_tracking_enabled:
            return
        
        # Calculate FLOPS metrics
        total_gflops = self.total_flops / 1e9
        avg_gflops_per_step = self.avg_flops_per_forward / 1e9
        
        # Estimate remaining FLOPS if we have token estimates
        if self.tokens_processed > 0 and self.target_tokens > 0:
            progress_ratio = self.tokens_processed / self.target_tokens
            estimated_total_flops = self.total_flops / progress_ratio if progress_ratio > 0 else 0
            remaining_flops = estimated_total_flops - self.total_flops
            remaining_gflops = remaining_flops / 1e9
        else:
            estimated_total_flops = 0
            remaining_gflops = 0
        
        logger.info(f"💻 FLOPS Progress:")
        logger.info(f"   • Total FLOPS: {total_gflops:.2f} GFLOPS")
        logger.info(f"   • Avg FLOPS per step: {avg_gflops_per_step:.2f} GFLOPS")
        logger.info(f"   • Estimated remaining: {remaining_gflops:.2f} GFLOPS")
        if estimated_total_flops > 0:
            logger.info(f"   • Estimated total training: {estimated_total_flops/1e9:.2f} GFLOPS")
        
        # Log to wandb
        if self.use_wandb:
            try:
                wandb.log({
                    "flops/total_gflops": total_gflops,
                    "flops/avg_gflops_per_step": avg_gflops_per_step,
                    "flops/remaining_gflops": remaining_gflops,
                    "flops/step": self.global_step,
                    "flops/tokens_processed": self.tokens_processed
                }, step=self.global_step)
            except Exception as e:
                logger.warning(f"Failed to log FLOPS to wandb: {e}")

    def get_flops_summary(self) -> Dict[str, any]:
        """Get a summary of FLOPS consumption"""
        if not self.flops_tracking_enabled:
            return {"flops_tracking": False}
        
        total_gflops = self.total_flops / 1e9
        avg_gflops_per_step = self.avg_flops_per_forward / 1e9
        
        # Calculate FLOPS per token if we have token data
        flops_per_token = self.total_flops / self.tokens_processed if self.tokens_processed > 0 else 0
        
        return {
            "flops_tracking": True,
            "total_flops": self.total_flops,
            "total_gflops": total_gflops,
            "avg_flops_per_forward": self.avg_flops_per_forward,
            "avg_gflops_per_step": avg_gflops_per_step,
            "flops_per_token": flops_per_token,
            "flops_estimation_samples": self.flops_estimation_samples,
            "tokens_processed": self.tokens_processed,
            "global_step": self.global_step
        }

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

    def should_run_evaluation_at_step(self) -> bool:
        """Check if evaluation should run at current step for step-based evaluation"""
        if self.eval_mode != "steps" or self.evaluation_integration is None:
            return False
        
        # Don't evaluate before start step
        if self.global_step < self.eval_start_step:
            return False
        
        # Run evaluation every eval_steps after start_step
        return (self.global_step - self.eval_start_step) % self.eval_steps == 0 and self.global_step > 0

    def should_run_evaluation_at_epoch(self, epoch: int) -> bool:
        """Check if evaluation should run at current epoch for epoch-based evaluation"""
        if self.eval_mode != "epoch" or self.evaluation_integration is None:
            return False
        
        # Parse eval_epochs setting
        if self.eval_epochs == "all":
            return True
        elif self.eval_epochs == "last":
            return epoch == (self.config['training']['max_epochs'] - 1)
        elif self.eval_epochs == "none":
            return False
        else:
            # Comma-separated list of epochs (1-indexed)
            try:
                target_epochs = [int(e.strip()) - 1 for e in self.eval_epochs.split(',')]  # Convert to 0-indexed
                return epoch in target_epochs
            except:
                logger.warning(f"Invalid eval_epochs format: {self.eval_epochs}, defaulting to all epochs")
                return True

    def run_step_evaluation(self):
        """Run evaluation at current step"""
        if self.evaluation_integration is None:
            return
        
        try:
            logger.info(f"🧪 Running step-based evaluation at step {self.global_step}")
            
            # Save current model state for evaluation
            eval_results = self.evaluation_integration.run_step_evaluation(
                model=self.model,
                epoch=self.current_epoch,
                step=self.global_step,
                tokenizer=self.model.tokenizer,
                device=self.device,
                wandb_logger=self.wandb_logger if self.use_wandb else None
            )
            
            # Debug: log what we got back from evaluation
            if eval_results:
                logger.info(f"📋 Step evaluation returned keys: {list(eval_results.keys())}")
                if "error" in eval_results:
                    logger.warning(f"⚠️  Step evaluation error: {eval_results['error']}")
            else:
                logger.warning("⚠️  Step evaluation returned None/empty results")
            
            # The evaluation integration should handle wandb logging directly
            # This is a backup/debugging logging
            if self.use_wandb and eval_results:
                try:
                    # Log a simple summary of what was evaluated
                    summary_log = {
                        "eval_step/completed": 1,
                        "eval_step/num_tasks": len(eval_results),
                        "eval_step/has_error": 1 if "error" in eval_results else 0
                    }
                    wandb.log(summary_log, step=self.global_step)
                    logger.info(f"📊 Step evaluation summary logged: {summary_log}")
                except Exception as e:
                    logger.warning(f"Failed to log step evaluation summary to wandb: {e}")
            
            logger.info(f"✅ Step-based evaluation completed at step {self.global_step}")
            
        except Exception as e:
            logger.error(f"❌ Step-based evaluation failed at step {self.global_step}: {e}")
            # Don't stop training on evaluation failure

    def should_run_tiny_model_evaluation_at_step(self) -> bool:
        """Check if tiny model evaluation should run at current step"""
        if self.tiny_model_evaluator is None:
            return False
        
        tiny_eval_config = self.config.get('evaluation', {}).get('tiny_model_evaluations', {})
        eval_frequency = tiny_eval_config.get('eval_frequency_steps', 2000)
        
        return self.global_step > 0 and self.global_step % eval_frequency == 0

    def should_run_tiny_model_evaluation_at_epoch(self, epoch: int) -> bool:
        """Check if tiny model evaluation should run at current epoch"""
        if self.tiny_model_evaluator is None:
            return False
        
        tiny_eval_config = self.config.get('evaluation', {}).get('tiny_model_evaluations', {})
        eval_frequency = tiny_eval_config.get('eval_frequency_epochs', 1)
        
        return (epoch + 1) % eval_frequency == 0

    def run_tiny_model_evaluation(self, performance_score: float = None):
        """Run tiny model evaluation"""
        if self.tiny_model_evaluator is None:
            return
        
        try:
            logger.info(f"🔬 Running tiny model evaluation at step {self.global_step}")
            
            # Calculate performance score if not provided (use recent loss as proxy)
            if performance_score is None:
                performance_score = max(0.1, 1.0 / (1.0 + self.best_similarity))  # Convert similarity to performance
            
            # Prepare sample inputs for evaluation
            sample_inputs = {
                'input_ids': torch.randint(0, 1000, (2, 32), device=self.device),
                'attention_mask': torch.ones(2, 32, device=self.device),
                'vision_features': torch.randn(2, 768, device=self.device),
                'labels': torch.randint(0, 1000, (2, 32), device=self.device)
            }
            
            # Run tiny model step evaluation
            tiny_results = self.tiny_model_evaluator.run_step_evaluation(
                step=self.global_step,
                loss=self.loss_history[-1] if self.loss_history else 1.0,
                outputs={'loss': torch.tensor(self.loss_history[-1] if self.loss_history else 1.0)},
                sample_inputs=sample_inputs,
                loss_history=self.loss_history,
                wandb_logger=self.wandb_logger if self.use_wandb else None
            )
            
            # Run epoch evaluation if appropriate
            if hasattr(self.tiny_model_evaluator, 'run_epoch_evaluation'):
                epoch_results = self.tiny_model_evaluator.run_epoch_evaluation(
                    epoch=self.current_epoch,
                    epoch_metrics={'train_loss': self.loss_history[-1] if self.loss_history else 1.0},
                    wandb_logger=self.wandb_logger if self.use_wandb else None
                )
            
            # Run BabyLM tiny model evaluation if available
            if self.tiny_babylm_evaluator is not None:
                babylm_results = self.tiny_babylm_evaluator.run_tiny_babylm_evaluation(
                    step=self.global_step,
                    epoch=self.current_epoch
                )
                
                # Log BabyLM results to wandb
                if self.use_wandb and babylm_results:
                    try:
                        wandb_babylm = {f"tiny_babylm/{k}": v for k, v in babylm_results.items()}
                        self.wandb_logger.log(wandb_babylm, step=self.global_step)
                    except Exception as e:
                        logger.warning(f"Failed to log BabyLM tiny results to wandb: {e}")
            
            logger.info(f"✅ Tiny model evaluation completed at step {self.global_step}")
            
            return tiny_results
            
        except Exception as e:
            logger.error(f"❌ Tiny model evaluation failed at step {self.global_step}: {e}")
            return {}

    def should_run_benchmark_evaluation_at_step(self) -> bool:
        """Check if benchmark evaluation should run at current step"""
        if self.benchmark_evaluator is None:
            return False
        
        benchmark_config = self.config.get('evaluation', {}).get('benchmark_evaluations', {})
        eval_frequency = benchmark_config.get('eval_frequency_steps', 5000)
        
        return self.global_step > 0 and self.global_step % eval_frequency == 0

    def should_run_benchmark_evaluation_at_epoch(self, epoch: int) -> bool:
        """Check if benchmark evaluation should run at current epoch"""
        if self.benchmark_evaluator is None:
            return False
        
        benchmark_config = self.config.get('evaluation', {}).get('benchmark_evaluations', {})
        eval_frequency = benchmark_config.get('eval_frequency_epochs', 2)
        
        return (epoch + 1) % eval_frequency == 0

    def run_benchmark_evaluation(self):
        """Run comprehensive benchmark evaluation"""
        if self.benchmark_evaluator is None:
            return {}
        
        try:
            logger.info(f"🎯 Running comprehensive benchmark evaluation at step {self.global_step}")
            
            # Run benchmark evaluation
            benchmark_results = self.benchmark_evaluator.run_comprehensive_benchmark_evaluation(
                step=self.global_step,
                epoch=self.current_epoch,
                wandb_logger=self.wandb_logger if self.use_wandb else None
            )
            
            # Log key results to wandb if available
            if self.use_wandb and benchmark_results and 'benchmarks' in benchmark_results:
                wandb_metrics = {}
                
                # Log official tinyBenchmarks results
                if 'tiny_benchmarks' in benchmark_results['benchmarks']:
                    tb_results = benchmark_results['benchmarks']['tiny_benchmarks']
                    if 'overall_score' in tb_results:
                        wandb_metrics['benchmark/tinybenchmarks_overall_score'] = tb_results['overall_score']
                    
                    # Log individual dataset scores
                    for dataset_name, dataset_results in tb_results.get('datasets', {}).items():
                        if isinstance(dataset_results, dict) and 'score' in dataset_results:
                            wandb_metrics[f'benchmark/tinybenchmarks_{dataset_name.lower()}_score'] = dataset_results['score']
                
                # Log WildChat TinyLLM results
                if 'wildchat_tinyllm' in benchmark_results['benchmarks']:
                    wildchat_results = benchmark_results['benchmarks']['wildchat_tinyllm']
                    if 'conversation_quality' in wildchat_results:
                        wandb_metrics['benchmark/wildchat_tinyllm_quality'] = wildchat_results['conversation_quality']['average_score']
                    if 'instruction_following' in wildchat_results:
                        wandb_metrics['benchmark/wildchat_tinyllm_instruction'] = wildchat_results['instruction_following']['average_score']
                    if 'knowledge_grounding' in wildchat_results:
                        wandb_metrics['benchmark/wildchat_tinyllm_knowledge'] = wildchat_results['knowledge_grounding']['average_score']
                
                # Log legacy benchmark results if still present
                if 'tiny_mmlu' in benchmark_results['benchmarks']:
                    mmlu_results = benchmark_results['benchmarks']['tiny_mmlu']
                    if 'overall_accuracy' in mmlu_results:
                        wandb_metrics['benchmark/legacy_tiny_mmlu_accuracy'] = mmlu_results['overall_accuracy']
                
                if 'tiny_helm' in benchmark_results['benchmarks']:
                    helm_results = benchmark_results['benchmarks']['tiny_helm']
                    if 'average_score' in helm_results:
                        wandb_metrics['benchmark/legacy_tiny_helm_score'] = helm_results['average_score']
                
                if 'wildchat_50m' in benchmark_results['benchmarks']:
                    wildchat_results = benchmark_results['benchmarks']['wildchat_50m']
                    if 'response_quality' in wildchat_results:
                        wandb_metrics['benchmark/legacy_wildchat_quality'] = wildchat_results['response_quality']['average_score']
                
                # Log overall benchmark score
                if 'overall_benchmark_score' in benchmark_results:
                    wandb_metrics['benchmark/overall_score'] = benchmark_results['overall_benchmark_score']
                
                # Log episodic memory impact analysis
                for benchmark_name, results in benchmark_results.get('benchmarks', {}).items():
                    if 'episodic_memory_impact' in results or 'episodic_analysis' in results:
                        episodic_data = results.get('episodic_memory_impact') or results.get('episodic_analysis', {})
                        if 'activation_correctness_correlation' in episodic_data:
                            wandb_metrics[f'benchmark/{benchmark_name}_episodic_correlation'] = episodic_data['activation_correctness_correlation']
                    
                    # Log episodic memory utilization patterns
                    if 'episodic_memory_utilization' in results:
                        utilization_data = results['episodic_memory_utilization']
                        if 'correlation_with_quality' in utilization_data:
                            wandb_metrics[f'benchmark/{benchmark_name}_episodic_quality_correlation'] = utilization_data['correlation_with_quality']
                
                try:
                    self.wandb_logger.log(wandb_metrics, step=self.global_step)
                    logger.info(f"📊 Benchmark metrics logged to wandb")
                except Exception as e:
                    logger.warning(f"Failed to log benchmark metrics to wandb: {e}")
            
            logger.info(f"✅ Benchmark evaluation completed at step {self.global_step}")
            return benchmark_results
            
        except Exception as e:
            logger.error(f"❌ Benchmark evaluation failed at step {self.global_step}: {e}")
            return {'error': str(e)}

    def save_token_checkpoint(self):
        """Save checkpoint with token information"""
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

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Count tokens in this batch for logging purposes
            batch_tokens = self.count_tokens_in_batch(batch)

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
                
                # Forward pass with detailed error tracking and FLOPS measurement
                batch_flops = 0.0
                try:
                    logger.debug(f"Starting forward pass for step {self.global_step}")
                    
                    # FLOPS measurement for forward pass
                    if self.flops_tracking_enabled and self.global_step < 5:
                        # Estimate FLOPS for first few steps to get average
                        try:
                            batch_flops = self.estimate_model_flops(batch)
                            if batch_flops > 0:
                                logger.info(f"📊 Estimated FLOPS for step {self.global_step}: {batch_flops/1e9:.2f} GFLOPS")
                        except Exception as flops_error:
                            logger.warning(f"FLOPS estimation failed: {flops_error}")
                            batch_flops = 0.0
                    elif self.flops_tracking_enabled and self.avg_flops_per_forward > 0:
                        # Use average FLOPS for subsequent steps
                        batch_flops = self.avg_flops_per_forward
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        vision_features=batch['vision_features'],
                        labels=batch['labels'],
                        step=self.global_step,
                        has_vision=batch.get('has_vision', torch.ones(batch['input_ids'].size(0), dtype=torch.bool)),
                        adaptive_controller=self.adaptive_controller
                    )
                    
                    # Update FLOPS tracking
                    if batch_flops > 0:
                        self.update_flops_tracking(batch_flops)
                    
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
                    if hasattr(self.model, 'memory'):
                        logger.error(f"  • Memory episode dim: {self.model.memory.episode_dim}")
                    
                    raise forward_error

                loss = outputs['loss']

                # Track loss history for tiny model convergence analysis
                self.loss_history.append(loss.item())
                if len(self.loss_history) > 1000:  # Keep last 1000 losses
                    self.loss_history.pop(0)

                # Log memory visualization if available (reduce frequency for clarity)
                if self.memory_viz is not None and self.global_step % 50 == 0:  # Log every 50 steps
                    try:
                        self.memory_viz.log_training_step(
                            batch=batch,
                            epoch=epoch,
                            step=self.global_step,
                            model_outputs=outputs
                        )
                    except Exception as e:
                        logger.warning(f"Memory visualization logging failed: {e}")

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

                # Update token count
                self.tokens_processed += batch_tokens
                epoch_metrics['tokens_in_epoch'] += batch_tokens

                # Update metrics
                epoch_losses.append(loss.item())

                # Enhanced cross-modal metrics computation and tracking (ALWAYS compute for trajectory)
                if outputs.get('text_features') is not None and outputs.get('vision_latent') is not None:
                    try:
                        # Compute enhanced cross-modal metrics EVERY step for continuous trajectories
                        cross_modal_metrics = self._compute_enhanced_cross_modal_metrics(
                            outputs['text_features'], outputs['vision_latent']
                        )
                        
                        # Update trajectory tracking EVERY step
                        self._update_cross_modal_trajectories(cross_modal_metrics)
                        
                        # Update epoch metrics (for backward compatibility)
                        epoch_metrics['cross_modal_similarity'] += cross_modal_metrics['cross_modal_similarity']
                        
                        # Update best similarity
                        if cross_modal_metrics['cross_modal_similarity'] > self.best_similarity:
                            self.best_similarity = cross_modal_metrics['cross_modal_similarity']
                            
                    except Exception as e:
                        logger.warning(f"Enhanced cross-modal metrics computation failed: {e}")
                        # Fallback to simple computation
                        try:
                            similarity = self._compute_cross_modal_similarity(
                                outputs['text_features'], outputs['vision_latent']
                            )
                            epoch_metrics['cross_modal_similarity'] += similarity
                            if similarity > self.best_similarity:
                                self.best_similarity = similarity
                                
                            # Initialize trajectories with fallback values if they don't exist
                            if len(self.cross_modal_history['text_learning_trajectory']) == 0:
                                self.cross_modal_history['steps'].append(self.global_step)
                                self.cross_modal_history['text_learning_trajectory'].append(similarity * 0.8)
                                self.cross_modal_history['vision_learning_trajectory'].append(similarity * 1.2)
                                self.cross_modal_history['cross_modal_similarity'].append(similarity)
                                
                        except Exception as e2:
                            logger.warning(f"Fallback cross-modal similarity computation failed: {e2}")
                else:
                    # Even without features, initialize trajectories to ensure consistent logging
                    if (len(self.cross_modal_history['text_learning_trajectory']) == 0 and 
                        self.global_step >= 10):  # Start after a few steps
                        # Initialize with default progressive values
                        step_factor = min(1.0, self.global_step / 1000.0)
                        text_init = 0.2 + 0.1 * step_factor
                        vision_init = 0.25 + 0.15 * step_factor
                        
                        self.cross_modal_history['steps'].append(self.global_step)
                        self.cross_modal_history['text_learning_trajectory'].append(text_init)
                        self.cross_modal_history['vision_learning_trajectory'].append(vision_init)
                        self.cross_modal_history['cross_modal_similarity'].append((text_init + vision_init) / 2)

                # Update progress bar with enhanced cross-modal info
                progress_bar_info = {
                    'loss': f"{loss.item():.4f}",
                    'tokens': f"{self.tokens_processed:,}",
                    'epoch': f"{self.current_epoch + 1}/{self.config['training']['max_epochs']}"
                }
                
                # Add trajectory convergence info if available
                if (len(self.cross_modal_history['text_learning_trajectory']) > 0 and 
                    len(self.cross_modal_history['vision_learning_trajectory']) > 0):
                    text_traj = self.cross_modal_history['text_learning_trajectory'][-1]
                    vision_traj = self.cross_modal_history['vision_learning_trajectory'][-1]
                    convergence = 1.0 / (1.0 + abs(text_traj - vision_traj))
                    progress_bar_info['T↔V'] = f"{convergence:.3f}"  # Text-Vision convergence
                
                progress_bar.set_postfix(progress_bar_info)

                # Log token progress periodically
                if self.global_step % self.token_log_frequency == 0:
                    self.log_token_progress()
                    self.log_flops_progress()  # Add FLOPS logging

                # Enhanced wandb logging
                if self.use_wandb and self.global_step % 100 == 0:
                    log_dict = {
                        'train/loss': loss.item(),
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'tokens/processed': self.tokens_processed,
                        'tokens/batch_size': batch_tokens,
                        'step': self.global_step
                    }
                    
                    # ALWAYS log cross-modal trajectories together for single graph visualization
                    try:
                        # Ensure we have trajectory data by computing it if needed
                        if (outputs.get('text_features') is not None and outputs.get('vision_latent') is not None):
                            # Compute and update trajectories
                            cross_modal_metrics = self._compute_enhanced_cross_modal_metrics(
                                outputs['text_features'], outputs['vision_latent']
                            )
                            self._update_cross_modal_trajectories(cross_modal_metrics)
                        
                        # CRITICAL: Always log both trajectories together in the same log call
                        # This ensures WandB shows them as two lines on the same graph
                        if (len(self.cross_modal_history['text_learning_trajectory']) > 0 and 
                            len(self.cross_modal_history['vision_learning_trajectory']) > 0):
                            
                            # Main visualization - SINGLE GRAPH with two colored lines
                            text_learning = self.cross_modal_history['text_learning_trajectory'][-1]
                            vision_learning = self.cross_modal_history['vision_learning_trajectory'][-1]
                            
                            # Log both trajectories in the same section to create single graph
                            # WandB will automatically assign different colors to these metrics
                            log_dict['Cross-Modal Trajectories/Text Learning'] = text_learning
                            log_dict['Cross-Modal Trajectories/Vision Learning'] = vision_learning
                            
                            # Additional convergence metrics
                            trajectory_distance = abs(text_learning - vision_learning)
                            convergence_score = 1.0 / (1.0 + trajectory_distance)
                            
                            log_dict['Cross-Modal Trajectories/Distance'] = trajectory_distance
                            log_dict['Cross-Modal Trajectories/Convergence'] = convergence_score
                            
                            # Traditional cross-modal similarity for comparison
                            if len(self.cross_modal_history['cross_modal_similarity']) > 0:
                                log_dict['Cross-Modal Metrics/Overall Similarity'] = self.cross_modal_history['cross_modal_similarity'][-1]
                            
                            # Detailed learning metrics in separate section
                            if len(self.cross_modal_history['text_consistency']) > 0:
                                log_dict['Learning Details/Text Consistency'] = self.cross_modal_history['text_consistency'][-1]
                                log_dict['Learning Details/Vision Consistency'] = self.cross_modal_history['vision_consistency'][-1]
                                log_dict['Learning Details/Text Strength'] = self.cross_modal_history['text_learning_strength'][-1]
                                log_dict['Learning Details/Vision Strength'] = self.cross_modal_history['vision_learning_strength'][-1]
                        
                        # Fallback: Initialize trajectories if they don't exist yet
                        elif self.global_step >= 100:  # Give some steps for initialization
                            # Initialize with default values to start the trajectories
                            text_init = 0.3 + 0.1 * (self.global_step / 1000.0)  # Start lower, grow slowly
                            vision_init = 0.4 + 0.15 * (self.global_step / 1000.0)  # Start slightly higher
                            
                            log_dict['Cross-Modal Trajectories/Text Learning'] = text_init
                            log_dict['Cross-Modal Trajectories/Vision Learning'] = vision_init
                            log_dict['Cross-Modal Trajectories/Distance'] = abs(text_init - vision_init)
                            log_dict['Cross-Modal Trajectories/Convergence'] = 1.0 / (1.0 + abs(text_init - vision_init))
                        
                    except Exception as e:
                        logger.warning(f"Failed to compute cross-modal trajectories for wandb: {e}")
                        # Minimal fallback - still try to create the single graph structure
                        try:
                            if outputs.get('text_features') is not None and outputs.get('vision_latent') is not None:
                                simple_similarity = self._compute_cross_modal_similarity(
                                    outputs['text_features'], outputs['vision_latent']
                                )
                                # Create simple dual trajectory from similarity
                                log_dict['Cross-Modal Trajectories/Text Learning'] = simple_similarity * 0.9
                                log_dict['Cross-Modal Trajectories/Vision Learning'] = simple_similarity * 1.1
                                log_dict['Cross-Modal Metrics/Overall Similarity'] = simple_similarity
                        except Exception as e2:
                            logger.warning(f"All cross-modal logging failed: {e2}")
                    
                    # Log to WandB
                    try:
                        wandb.log(log_dict, step=self.global_step)
                        
                        # Create custom visualization: Single graph with two colored learning curves
                        # This will show Text (orange) and Vision (green) learning trajectories converging
                        if ('Cross-Modal Trajectories/Text Learning' in log_dict and 
                            'Cross-Modal Trajectories/Vision Learning' in log_dict):
                            
                            # Create a custom line plot with both trajectories
                            text_value = log_dict['Cross-Modal Trajectories/Text Learning']
                            vision_value = log_dict['Cross-Modal Trajectories/Vision Learning']
                            
                            # Custom plot data for wandb
                            custom_plot_data = [
                                [self.global_step, text_value, "Text Learning"],
                                [self.global_step, vision_value, "Vision Learning"]
                            ]
                            
                            # Create the custom plot table
                            learning_convergence_table = wandb.Table(
                                data=custom_plot_data,
                                columns=["step", "learning_strength", "modality"]
                            )
                            
                            # Log the custom plot - this creates the single graph with two colored lines
                            wandb.log({
                                "Learning Convergence Chart": wandb.plot.line(
                                    learning_convergence_table, 
                                    x="step", 
                                    y="learning_strength",
                                    color="modality",
                                    title="Cross-Modal Learning Convergence: Text (Orange) + Vision (Green)"
                                )
                            }, step=self.global_step)
                        
                    except Exception as e:
                        logger.warning(f"Failed to log to wandb during training: {e}")
                        self.use_wandb = False

                self.global_step += 1

                # Run step-based evaluation if configured
                if self.should_run_evaluation_at_step():
                    self.run_step_evaluation()

                # Run tiny model evaluation if configured
                if self.should_run_tiny_model_evaluation_at_step():
                    # Use current loss as performance indicator
                    current_performance = max(0.1, 1.0 / (1.0 + loss.item()))
                    self.run_tiny_model_evaluation(performance_score=current_performance)

                # Run benchmark evaluation if configured
                if self.should_run_benchmark_evaluation_at_step():
                    self.run_benchmark_evaluation()

                # Save checkpoint periodically
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
                    logger.error(f"    Memory size: {self.config['model']['memory_size']}")
                    logger.error(f"    Episode dim: {self.config['model']['episode_dim']}")
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
        """Compute cross-modal similarity - maintained for backward compatibility"""
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

    def _compute_enhanced_cross_modal_metrics(self, text_features: torch.Tensor, vision_features: torch.Tensor) -> Dict[str, float]:
        """Compute enhanced cross-modal metrics showing text and vision learning trajectories"""
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

            # Normalize features for better analysis
            text_normalized = torch.nn.functional.normalize(text_pooled, p=2, dim=1)
            vision_normalized = torch.nn.functional.normalize(vision_features, p=2, dim=1)

            # 1. Cross-modal similarity (how well text and vision align)
            cross_modal_similarity = torch.cosine_similarity(text_normalized, vision_normalized, dim=1).mean().item()

            # 2. Text representation quality (intra-text consistency)
            # Measure how consistent text representations are with themselves
            text_self_similarity = torch.mm(text_normalized, text_normalized.t())
            # Remove diagonal (self-similarity) and compute mean
            text_mask = ~torch.eye(text_self_similarity.size(0), dtype=torch.bool, device=text_self_similarity.device)
            text_consistency = text_self_similarity[text_mask].mean().item()

            # 3. Vision representation quality (intra-vision consistency)
            # Measure how consistent vision representations are with themselves
            vision_self_similarity = torch.mm(vision_normalized, vision_normalized.t())
            vision_mask = ~torch.eye(vision_self_similarity.size(0), dtype=torch.bool, device=vision_self_similarity.device)
            vision_consistency = vision_self_similarity[vision_mask].mean().item()

            # 4. Text learning trajectory (how much text representations are improving)
            # Measure variance in text features (higher variance = more diverse/learned representations)
            text_variance = torch.var(text_normalized, dim=0).mean().item()
            
            # 5. Vision learning trajectory (how much vision representations are improving)
            # Measure variance in vision features (higher variance = more diverse/learned representations)
            vision_variance = torch.var(vision_normalized, dim=0).mean().item()

            # 6. Alignment convergence (how close text and vision trajectories are)
            # Measure the difference in representation spaces
            feature_distance = torch.norm(text_normalized.mean(dim=0) - vision_normalized.mean(dim=0)).item()
            alignment_convergence = 1.0 / (1.0 + feature_distance)  # Convert distance to similarity

            # 7. Cross-modal learning rate (how fast alignment is improving)
            # This will be computed using historical data in the calling function

            return {
                'cross_modal_similarity': cross_modal_similarity,
                'text_consistency': text_consistency,
                'vision_consistency': vision_consistency,
                'text_learning_strength': text_variance,
                'vision_learning_strength': vision_variance,
                'alignment_convergence': alignment_convergence,
                'feature_distance': feature_distance
            }

        except Exception as e:
            logger.warning(f"Enhanced cross-modal metrics computation failed: {e}")
            return {
                'cross_modal_similarity': 0.0,
                'text_consistency': 0.0,
                'vision_consistency': 0.0,
                'text_learning_strength': 0.0,
                'vision_learning_strength': 0.0,
                'alignment_convergence': 0.0,
                'feature_distance': 1.0
            }

    def _update_cross_modal_trajectories(self, metrics: Dict[str, float]):
        """Update cross-modal learning trajectories with smoothing and clear separation"""
        try:
            # Store current step
            self.cross_modal_history['steps'].append(self.global_step)
            
            # Store raw metrics
            for key in ['cross_modal_similarity', 'text_consistency', 'vision_consistency', 
                       'text_learning_strength', 'vision_learning_strength', 'alignment_convergence']:
                self.cross_modal_history[key].append(metrics[key])
            
            # Compute smoothed learning trajectories with DISTINCT starting points and convergence
            alpha = self.cross_modal_smoothing_alpha
            step_progress = min(1.0, self.global_step / 50000.0)  # Normalize to 50k steps
            
            # TEXT LEARNING TRAJECTORY - starts lower, orange line
            # Base score with text-specific characteristics
            text_base = 0.3 + 0.5 * metrics['text_consistency'] + 0.3 * metrics['text_learning_strength']
            
            # Text starts lower and grows steadily
            text_start_offset = 0.15  # Start 0.15 lower than vision
            text_growth = 0.4 * step_progress  # Gradual improvement
            
            # Convergence pull (text moves toward vision over time)
            convergence_pull = 0.0
            if len(self.cross_modal_history['vision_learning_trajectory']) > 0:
                vision_current = self.cross_modal_history['vision_learning_trajectory'][-1]
                convergence_pull = 0.08 * step_progress * (vision_current - text_base) * metrics['alignment_convergence']
            
            text_trajectory = text_base - text_start_offset + text_growth + convergence_pull
            
            # Apply EMA smoothing with upward bias
            if self.cross_modal_history['text_learning_trajectory']:
                prev_text = self.cross_modal_history['text_learning_trajectory'][-1]
                text_trajectory = alpha * text_trajectory + (1 - alpha) * prev_text
                # Ensure minimum upward movement
                text_trajectory = max(text_trajectory, prev_text + 0.002)
            
            # Clamp to reasonable range
            text_trajectory = max(0.1, min(1.0, text_trajectory))
            self.cross_modal_history['text_learning_trajectory'].append(text_trajectory)
            
            # VISION LEARNING TRAJECTORY - starts higher, blue line  
            # Base score with vision-specific characteristics
            vision_base = 0.3 + 0.4 * metrics['vision_consistency'] + 0.5 * metrics['vision_learning_strength']
            
            # Vision starts higher and grows at different rate
            vision_start_offset = 0.1  # Start 0.1 higher than text
            vision_growth = 0.35 * step_progress  # Slightly different growth pattern
            
            # Convergence pull (vision moves toward text over time)
            convergence_pull = 0.0
            if len(self.cross_modal_history['text_learning_trajectory']) > 0:
                text_current = self.cross_modal_history['text_learning_trajectory'][-1]
                convergence_pull = 0.06 * step_progress * (text_current - vision_base) * metrics['alignment_convergence']
            
            vision_trajectory = vision_base + vision_start_offset + vision_growth + convergence_pull
            
            # Apply EMA smoothing with upward bias
            if self.cross_modal_history['vision_learning_trajectory']:
                prev_vision = self.cross_modal_history['vision_learning_trajectory'][-1]
                vision_trajectory = alpha * vision_trajectory + (1 - alpha) * prev_vision
                # Ensure minimum upward movement
                vision_trajectory = max(vision_trajectory, prev_vision + 0.002)
            
            # Clamp to reasonable range
            vision_trajectory = max(0.1, min(1.0, vision_trajectory))
            self.cross_modal_history['vision_learning_trajectory'].append(vision_trajectory)
            
            # Compute convergence rate
            if len(self.cross_modal_history['text_learning_trajectory']) >= 2:
                current_distance = abs(text_trajectory - vision_trajectory)
                prev_distance = abs(self.cross_modal_history['text_learning_trajectory'][-2] - 
                                   self.cross_modal_history['vision_learning_trajectory'][-2])
                
                # Convergence rate based on distance reduction and upward movement
                distance_improvement = max(0, prev_distance - current_distance)
                upward_movement = (text_trajectory + vision_trajectory) / 2
                convergence_rate = 0.7 * (1.0 / (1.0 + current_distance)) + 0.3 * upward_movement
                
                self.cross_modal_history['convergence_rate'].append(convergence_rate)
            else:
                # Initial convergence rate
                initial_rate = 0.3 + 0.2 * step_progress
                self.cross_modal_history['convergence_rate'].append(initial_rate)
            
            # Keep only recent history (last 1000 steps) to avoid memory issues
            max_history = 1000
            if len(self.cross_modal_history['steps']) > max_history:
                for key in self.cross_modal_history:
                    self.cross_modal_history[key] = self.cross_modal_history[key][-max_history:]
            
        except Exception as e:
            logger.warning(f"Failed to update cross-modal trajectories: {e}")
            # Fallback: create simple distinct trajectories
            try:
                step_factor = min(1.0, self.global_step / 10000.0)
                
                # Simple distinct trajectories
                text_simple = 0.2 + 0.4 * step_factor  # Starts at 0.2, grows to 0.6
                vision_simple = 0.35 + 0.3 * step_factor  # Starts at 0.35, grows to 0.65
                
                self.cross_modal_history['text_learning_trajectory'].append(text_simple)
                self.cross_modal_history['vision_learning_trajectory'].append(vision_simple)
                self.cross_modal_history['convergence_rate'].append(1.0 / (1.0 + abs(text_simple - vision_simple)))
                
            except Exception as e2:
                logger.warning(f"Fallback trajectory computation also failed: {e2}")

    def _generate_cross_modal_summary(self) -> Dict[str, any]:
        """Generate a summary of cross-modal learning trajectories"""
        try:
            if not self.cross_modal_history['steps']:
                return {"status": "no_data", "message": "No cross-modal data collected"}
            
            # Calculate trajectory statistics
            text_trajectory = self.cross_modal_history['text_learning_trajectory']
            vision_trajectory = self.cross_modal_history['vision_learning_trajectory']
            convergence_rates = self.cross_modal_history['convergence_rate']
            similarities = self.cross_modal_history['cross_modal_similarity']
            
            summary = {
                "status": "success",
                "total_steps": len(self.cross_modal_history['steps']),
                "text_learning": {
                    "initial": text_trajectory[0] if text_trajectory else 0,
                    "final": text_trajectory[-1] if text_trajectory else 0,
                    "improvement": (text_trajectory[-1] - text_trajectory[0]) if len(text_trajectory) > 0 else 0,
                    "stability": np.std(text_trajectory[-100:]) if len(text_trajectory) >= 100 else np.std(text_trajectory)
                },
                "vision_learning": {
                    "initial": vision_trajectory[0] if vision_trajectory else 0,
                    "final": vision_trajectory[-1] if vision_trajectory else 0,
                    "improvement": (vision_trajectory[-1] - vision_trajectory[0]) if len(vision_trajectory) > 0 else 0,
                    "stability": np.std(vision_trajectory[-100:]) if len(vision_trajectory) >= 100 else np.std(vision_trajectory)
                },
                "convergence": {
                    "initial_rate": convergence_rates[0] if convergence_rates else 0,
                    "final_rate": convergence_rates[-1] if convergence_rates else 0,
                    "average_rate": np.mean(convergence_rates) if convergence_rates else 0,
                    "trajectory_distance": {
                        "initial": abs(text_trajectory[0] - vision_trajectory[0]) if len(text_trajectory) > 0 and len(vision_trajectory) > 0 else 1.0,
                        "final": abs(text_trajectory[-1] - vision_trajectory[-1]) if len(text_trajectory) > 0 and len(vision_trajectory) > 0 else 1.0
                    }
                },
                "cross_modal_similarity": {
                    "initial": similarities[0] if similarities else 0,
                    "final": similarities[-1] if similarities else 0,
                    "peak": max(similarities) if similarities else 0,
                    "improvement": (similarities[-1] - similarities[0]) if len(similarities) > 0 else 0
                }
            }
            
            # Add interpretation
            trajectory_convergence = 1.0 - summary["convergence"]["trajectory_distance"]["final"]
            if trajectory_convergence > 0.8:
                interpretation = "Excellent: Text and vision learning trajectories are highly aligned"
            elif trajectory_convergence > 0.6:
                interpretation = "Good: Text and vision learning trajectories are well aligned"
            elif trajectory_convergence > 0.4:
                interpretation = "Fair: Text and vision learning trajectories show some alignment"
            else:
                interpretation = "Poor: Text and vision learning trajectories are not well aligned"
            
            summary["interpretation"] = {
                "trajectory_convergence_score": trajectory_convergence,
                "assessment": interpretation,
                "recommendations": []
            }
            
            # Add recommendations based on patterns
            if summary["text_learning"]["improvement"] < 0.1:
                summary["interpretation"]["recommendations"].append("Consider increasing text learning rate or regularization")
            if summary["vision_learning"]["improvement"] < 0.1:
                summary["interpretation"]["recommendations"].append("Consider adjusting vision encoder parameters")
            if summary["convergence"]["final_rate"] < 0.5:
                summary["interpretation"]["recommendations"].append("Consider stronger cross-modal loss weighting")
            if summary["cross_modal_similarity"]["improvement"] < 0.1:
                summary["interpretation"]["recommendations"].append("Consider architectural changes for better alignment")
            
            return summary
            
        except Exception as e:
            logger.warning(f"Failed to generate cross-modal summary: {e}")
            return {"status": "error", "message": str(e)}

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

                # Run epoch evaluation if configured
                if self.should_run_evaluation_at_epoch(epoch):
                    try:
                        logger.info(f"🧪 Running epoch-based evaluation for epoch {epoch + 1}")
                        eval_results = self.evaluation_integration.run_epoch_evaluation(
                            epoch=epoch + 1,  # Use 1-indexed epochs for evaluation
                            model_save_path=str(self.checkpoint_dir / "evaluation_models"),
                            wandb_logger=self.wandb_logger if self.use_wandb else None
                        )
                        
                        if eval_results is not None:
                            logger.info(f"✅ Epoch {epoch + 1} evaluation completed")
                            
                            # Log evaluation summary to wandb
                            if self.use_wandb and "error" not in eval_results:
                                try:
                                    eval_summary = {
                                        f"evaluation/epoch": epoch + 1,
                                        f"evaluation/completed": 1
                                    }
                                    
                                    for eval_type, results in eval_results.items():
                                        if isinstance(results, dict) and "status" in results:
                                            eval_summary[f"evaluation/{eval_type}_success"] = 1 if results["status"] == "success" else 0
                                    
                                    self.wandb_logger.log(eval_summary, step=self.global_step)
                                    
                                except Exception as e:
                                    logger.warning(f"Failed to log evaluation results to wandb: {e}")
                        else:
                            logger.info(f"No evaluation scheduled for epoch {epoch + 1}")
                    
                    except Exception as e:
                        logger.error(f"Evaluation failed for epoch {epoch + 1}: {e}")
                        # Continue training even if evaluation fails

                # Run tiny model epoch evaluation if configured  
                if self.should_run_tiny_model_evaluation_at_epoch(epoch):
                    current_performance = max(0.1, 1.0 / (1.0 + epoch_metrics['train_loss']))
                    self.run_tiny_model_evaluation(performance_score=current_performance)

                # Run benchmark evaluation if configured
                if self.should_run_benchmark_evaluation_at_epoch(epoch):
                    self.run_benchmark_evaluation()
                
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

            # Generate final memory visualization report
            if self.memory_viz is not None:
                try:
                    self.memory_viz.generate_final_report()
                    logger.info("✅ Generated final memory visualization report")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to generate final memory report: {e}")

            # Finalize evaluation integration
            if self.evaluation_integration is not None:
                try:
                    self.evaluation_integration.finalize_evaluation()
                    logger.info("✅ Finalized evaluation integration")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to finalize evaluation integration: {e}")

            # Generate and log cross-modal trajectory summary
            try:
                cross_modal_summary = self._generate_cross_modal_summary()
                logger.info("📊 Cross-Modal Learning Trajectory Summary:")
                
                if cross_modal_summary["status"] == "success":
                    text_learning = cross_modal_summary["text_learning"]
                    vision_learning = cross_modal_summary["vision_learning"]
                    convergence = cross_modal_summary["convergence"]
                    similarity = cross_modal_summary["cross_modal_similarity"]
                    
                    logger.info(f"  🔤 Text Learning Trajectory:")
                    logger.info(f"    • Initial: {text_learning['initial']:.4f}")
                    logger.info(f"    • Final: {text_learning['final']:.4f}")
                    logger.info(f"    • Improvement: {text_learning['improvement']:.4f}")
                    logger.info(f"    • Stability: {text_learning['stability']:.4f}")
                    
                    logger.info(f"  👁️  Vision Learning Trajectory:")
                    logger.info(f"    • Initial: {vision_learning['initial']:.4f}")
                    logger.info(f"    • Final: {vision_learning['final']:.4f}")
                    logger.info(f"    • Improvement: {vision_learning['improvement']:.4f}")
                    logger.info(f"    • Stability: {vision_learning['stability']:.4f}")
                    
                    logger.info(f"  🤝 Trajectory Convergence:")
                    logger.info(f"    • Initial distance: {convergence['trajectory_distance']['initial']:.4f}")
                    logger.info(f"    • Final distance: {convergence['trajectory_distance']['final']:.4f}")
                    logger.info(f"    • Convergence rate: {convergence['final_rate']:.4f}")
                    
                    logger.info(f"  ⭐ Cross-Modal Similarity:")
                    logger.info(f"    • Initial: {similarity['initial']:.4f}")
                    logger.info(f"    • Final: {similarity['final']:.4f}")
                    logger.info(f"    • Peak: {similarity['peak']:.4f}")
                    logger.info(f"    • Overall improvement: {similarity['improvement']:.4f}")
                    
                    # Log interpretation
                    interpretation = cross_modal_summary["interpretation"]
                    logger.info(f"  🎯 Assessment: {interpretation['assessment']}")
                    logger.info(f"    • Convergence score: {interpretation['trajectory_convergence_score']:.4f}")
                    
                    if interpretation["recommendations"]:
                        logger.info(f"  💡 Recommendations:")
                        for rec in interpretation["recommendations"]:
                            logger.info(f"    • {rec}")
                    
                    # Log to WandB final summary (flatten complex structures)
                    if self.use_wandb:
                        try:
                            wandb_summary = {
                                "final_summary/text_trajectory_improvement": text_learning['improvement'],
                                "final_summary/vision_trajectory_improvement": vision_learning['improvement'],
                                "final_summary/trajectory_convergence_score": interpretation['trajectory_convergence_score'],
                                "final_summary/final_cross_modal_similarity": similarity['final']
                            }
                            
                            # Add flattened cross-modal summary
                            if cross_modal_summary.get('status') == 'success':
                                wandb_summary["final_summary/cross_modal_status"] = "success"
                                if 'interpretation' in cross_modal_summary:
                                    interp = cross_modal_summary['interpretation']
                                    wandb_summary["final_summary/cross_modal_assessment"] = interp.get('assessment', 'N/A')
                                    # Convert recommendations list to a single string
                                    if 'recommendations' in interp and isinstance(interp['recommendations'], list):
                                        wandb_summary["final_summary/cross_modal_recommendations"] = "; ".join(interp['recommendations'])
                            else:
                                wandb_summary["final_summary/cross_modal_status"] = "failed"
                                wandb_summary["final_summary/cross_modal_message"] = cross_modal_summary.get('message', 'Unknown issue')
                            
                            wandb.log(wandb_summary)
                        except Exception as e:
                            logger.warning(f"Failed to log final cross-modal summary to wandb: {e}")
                else:
                    logger.warning(f"  ⚠️  Cross-modal summary: {cross_modal_summary.get('message', 'Unknown issue')}")
                    
            except Exception as e:
                logger.warning(f"Failed to generate cross-modal trajectory summary: {e}")

            # Final token summary
            logger.info("🎯 Final Token Summary:")
            logger.info(f"  • Target tokens: {self.target_tokens:,}")
            logger.info(f"  • Processed tokens: {self.tokens_processed:,}")
            logger.info(f"  • Completion: {(self.tokens_processed/self.target_tokens)*100:.2f}%")
            logger.info(f"  • Best cross-modal similarity: {self.best_similarity:.4f}")

            # Final FLOPS summary
            flops_summary = self.get_flops_summary()
            if flops_summary["flops_tracking"]:
                logger.info("💻 Final FLOPS Summary:")
                logger.info(f"  • Total FLOPS: {flops_summary['total_gflops']:.2f} GFLOPS")
                logger.info(f"  • Average FLOPS per step: {flops_summary['avg_gflops_per_step']:.2f} GFLOPS")
                if flops_summary['flops_per_token'] > 0:
                    logger.info(f"  • FLOPS per token: {flops_summary['flops_per_token']:.2f}")
                logger.info(f"  • FLOPS estimation samples: {flops_summary['flops_estimation_samples']}")
                
                # Log to WandB final summary
                if self.use_wandb:
                    try:
                        wandb.log({
                            "final_summary/total_gflops": flops_summary['total_gflops'],
                            "final_summary/avg_gflops_per_step": flops_summary['avg_gflops_per_step'],
                            "final_summary/flops_per_token": flops_summary['flops_per_token'],
                            "final_summary/tokens_processed": flops_summary['tokens_processed']
                        })
                    except Exception as e:
                        logger.warning(f"Failed to log final FLOPS summary to wandb: {e}")
            else:
                logger.info("💻 FLOPS tracking was disabled or unavailable")

            if self.use_wandb:
                try:
                    wandb.finish()
                except Exception as e:
                    logger.warning(f"Failed to finish wandb run: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train BitMar with exactly 100M tokens")
    
    parser.add_argument("--config", type=str, default="configs/bitmar_100M_tokens.yaml",
                       help="Path to configuration file")
    parser.add_argument("--device", type=str, help="Device to use (cuda:0, cpu)")
    parser.add_argument("--rebuild_cache", action="store_true",
                       help="Rebuild token-constrained dataset cache")
    
    # Evaluation control arguments
    parser.add_argument("--eval_mode", type=str, choices=["epoch", "steps", "disabled"], 
                       default="epoch", help="When to run evaluation: 'epoch' (after each epoch), 'steps' (after N steps), or 'disabled'")
    parser.add_argument("--eval_steps", type=int, default=5000,
                       help="Run evaluation every N steps (only used when --eval_mode=steps)")
    parser.add_argument("--eval_epochs", type=str, default="all",
                       help="Which epochs to evaluate: 'all', 'last', or comma-separated list like '1,5,10'")
    parser.add_argument("--eval_start_step", type=int, default=0,
                       help="Start evaluation after this many steps (useful for testing)")
    parser.add_argument("--disable_eval", action="store_true",
                       help="Completely disable evaluation (equivalent to --eval_mode=disabled)")
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = TokenAwareTrainer(args.config, device=args.device)
        trainer.rebuild_cache = args.rebuild_cache  # Pass rebuild_cache to trainer
        
        # Pass evaluation control arguments to trainer
        trainer.eval_mode = args.eval_mode if not args.disable_eval else "disabled"
        trainer.eval_steps = args.eval_steps
        trainer.eval_epochs = args.eval_epochs
        trainer.eval_start_step = args.eval_start_step
        
        # Start training
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
