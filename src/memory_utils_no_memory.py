# """
# Episodic Memory Utilities for Edge Deployment
# Provides helper functions for managing external episodic memory storage
# """

# import torch
# import json
# import logging
# from pathlib import Path
# from typing import Dict, Optional, List
# import time

# logger = logging.getLogger(__name__)


# class MemoryManager:
#     """Manager for episodic memory operations"""

#     def __init__(self, model, base_path: str = "memory_storage"):
#         """
#         Initialize memory manager

#         Args:
#             model: BitMar model with episodic memory
#             base_path: Base directory for memory storage
#         """
#         self.model = model
#         self.base_path = Path(base_path)
#         self.base_path.mkdir(exist_ok=True)

#     def export_memory_for_edge(self, export_name: str = None, compress: bool = True) -> str:
#         """
#         Export episodic memory for edge deployment

#         Args:
#             export_name: Name for the exported memory file
#             compress: Whether to compress the memory

#         Returns:
#             Path to exported memory file
#         """
#         if export_name is None:
#             timestamp = int(time.time())
#             export_name = f"edge_memory_{timestamp}"

#         export_path = self.base_path / f"{export_name}.pt"

#         # Save memory with compression for edge deployment
#         saved_path = self.model.memory.save_external_memory(
#             str(export_path),
#             compress=compress
#         )

#         # Create deployment info file
#         info_path = export_path.with_suffix('.json')
#         deployment_info = {
#             'model_name': 'BitMar',
#             'memory_version': self.model.memory._memory_version,
#             'memory_size': self.model.memory.memory_size,
#             'episode_dim': self.model.memory.episode_dim,
#             'compressed': compress,
#             'compatible_model_config': {
#                 'memory_size': self.model.memory.memory_size,
#                 'episode_dim': self.model.memory.episode_dim,
#                 'memory_alpha': self.model.memory.alpha
#             },
#             'export_timestamp': time.time(),
#             'deployment_instructions': {
#                 'step1': 'Copy this file to SD card or external storage',
#                 'step2': 'Initialize model with external_storage=True',
#                 'step3': 'Call model.memory.load_external_memory(path_to_this_file)',
#                 'step4': 'Model is ready for inference with trained memory'
#             }
#         }

#         with open(info_path, 'w') as f:
#             json.dump(deployment_info, f, indent=2)

#         logger.info(f"🚀 Memory exported for edge deployment:")
#         logger.info(f"   Memory file: {saved_path}")
#         logger.info(f"   Info file: {info_path}")
#         logger.info(f"   Compressed: {compress}")

#         return str(export_path)

#     def create_edge_deployment_package(self, package_name: str = None) -> str:
#         """
#         Create complete deployment package for edge device

#         Args:
#             package_name: Name for the deployment package

#         Returns:
#             Path to deployment package directory
#         """
#         if package_name is None:
#             timestamp = int(time.time())
#             package_name = f"bitmar_edge_package_{timestamp}"

#         package_dir = self.base_path / package_name
#         package_dir.mkdir(exist_ok=True)

#         # Export compressed memory
#         memory_path = self.export_memory_for_edge(
#             f"{package_name}_memory",
#             compress=True
#         )

#         # Copy memory files to package
#         import shutil
#         shutil.copy2(memory_path, package_dir / "episodic_memory.pt")
#         shutil.copy2(
#             Path(memory_path).with_suffix('.json'),
#             package_dir / "memory_info.json"
#         )

#         # Create deployment script
#         deployment_script = f'''#!/usr/bin/env python3
# """
# BitMar Edge Deployment Script
# Automatically loads the model with external episodic memory
# """

# import torch
# import sys
# from pathlib import Path

# # Add your model loading code here
# # Example:
# # sys.path.append('path/to/bitmar/src')
# # from model_no_memory import create_bitmar_model

# def load_edge_model(config_path="model_config.json", device="cpu"):
#     """Load BitMar model with external memory for edge deployment"""
    
#     # Load your model configuration
#     # config = load_config(config_path)
    
#     # Create model with external storage enabled
#     # model = create_bitmar_model(config)
    
#     # Enable external storage for episodic memory
#     memory_path = Path(__file__).parent / "episodic_memory.pt"
#     # model.memory.enable_external_storage(
#     #     storage_path=str(memory_path),
#     #     compress=True,
#     #     lazy=True  # Load only when needed
#     # )
    
#     # Load the trained memory
#     # success = model.memory.load_external_memory(str(memory_path), device=device)
#     # if success:
#     #     print("✅ Episodic memory loaded successfully!")
#     # else:
#     #     print("⚠️ Failed to load episodic memory, using random initialization")
    
#     # return model

# if __name__ == "__main__":
#     model = load_edge_model()
#     print("🚀 BitMar model ready for edge inference!")
# '''

#         script_path = package_dir / "deploy_edge_model.py"
#         with open(script_path, 'w') as f:
#             f.write(deployment_script)

#         # Create README
#         readme_content = f'''# BitMar Edge Deployment Package

# This package contains everything needed to deploy BitMar with trained episodic memory on an edge device.

# ## Contents

# - `episodic_memory.pt`: Compressed episodic memory data
# - `memory_info.json`: Memory metadata and deployment info
# - `deploy_edge_model.py`: Example deployment script
# - `README.md`: This file

# ## Quick Start

# 1. Copy this entire folder to your edge device
# 2. Modify `deploy_edge_model.py` to include your model loading code
# 3. Run: `python deploy_edge_model.py`

# ## Memory Statistics

# - Memory slots: {self.model.memory.memory_size}
# - Episode dimension: {self.model.memory.episode_dim}
# - Compression: Enabled
# - Estimated memory size: ~{self.model.memory.memory_size * self.model.memory.episode_dim * 1 / 1024:.1f}KB (compressed)

# ## Compatibility

# This memory is compatible with BitMar models that have:
# - memory_size = {self.model.memory.memory_size}
# - episode_dim = {self.model.memory.episode_dim}

# ## Notes for Edge Deployment

# - The episodic memory can be stored on SD card or external storage
# - Lazy loading is supported - memory loads only when needed
# - Memory is compressed to reduce storage footprint
# - No internet connection required for inference

# Generated on: {time.ctime()}
# '''

#         readme_path = package_dir / "README.md"
#         with open(readme_path, 'w') as f:
#             f.write(readme_content)

#         logger.info(f"📦 Edge deployment package created: {package_dir}")
#         logger.info(f"   Package includes:")
#         logger.info(f"   - Compressed episodic memory")
#         logger.info(f"   - Deployment script")
#         logger.info(f"   - Documentation")

#         return str(package_dir)

#     def verify_memory_compatibility(self, memory_path: str) -> bool:
#         """
#         Verify that external memory is compatible with current model

#         Args:
#             memory_path: Path to external memory file

#         Returns:
#             True if compatible, False otherwise
#         """
#         try:
#             memory_data = torch.load(memory_path, map_location='cpu')

#             expected_size = self.model.memory.memory_size
#             expected_dim = self.model.memory.episode_dim

#             actual_size = memory_data['metadata']['memory_size']
#             actual_dim = memory_data['metadata']['episode_dim']

#             compatible = (actual_size == expected_size and actual_dim == expected_dim)

#             if compatible:
#                 logger.info(f"✅ Memory compatibility verified")
#                 logger.info(f"   Memory size: {actual_size} (matches)")
#                 logger.info(f"   Episode dim: {actual_dim} (matches)")
#             else:
#                 logger.error(f"❌ Memory compatibility check failed")
#                 logger.error(f"   Expected: size={expected_size}, dim={expected_dim}")
#                 logger.error(f"   Actual: size={actual_size}, dim={actual_dim}")

#             return compatible

#         except Exception as e:
#             logger.error(f"❌ Failed to verify memory compatibility: {e}")
#             return False

#     def list_available_memories(self) -> List[Dict]:
#         """List all available memory files in storage"""
#         memory_files = []

#         for file_path in self.base_path.glob("*.pt"):
#             try:
#                 info_path = file_path.with_suffix('.json')
#                 if info_path.exists():
#                     with open(info_path, 'r') as f:
#                         info = json.load(f)

#                     memory_files.append({
#                         'name': file_path.stem,
#                         'path': str(file_path),
#                         'size_mb': file_path.stat().st_size / (1024 * 1024),
#                         'created': info.get('creation_timestamp', 0),
#                         'version': info.get('version', 1),
#                         'compressed': info.get('compressed', False),
#                         'compatible': self.verify_memory_compatibility(str(file_path))
#                     })
#             except Exception as e:
#                 logger.warning(f"Could not read memory file {file_path}: {e}")

#         return sorted(memory_files, key=lambda x: x['created'], reverse=True)


# def prepare_model_for_evaluation(model, memory_mode: str = "integrated"):
#     """
#     Prepare model for evaluation with specified memory mode

#     Args:
#         model: BitMar model
#         memory_mode: "integrated" (default) or "external"
#     """
#     if memory_mode == "external":
#         # For demonstration, save and reload memory to test external storage
#         logger.info("🔄 Testing external memory mode...")

#         # Save current memory state
#         temp_path = "temp_memory_test.pt"
#         model.memory.save_external_memory(temp_path, compress=True)

#         # Enable external storage mode
#         model.memory.enable_external_storage(
#             storage_path=temp_path,
#             compress=True,
#             lazy=False  # Load immediately for evaluation
#         )

#         # Load the memory back
#         success = model.memory.load_external_memory(temp_path)
#         if success:
#             logger.info("✅ External memory mode enabled successfully")
#         else:
#             logger.warning("⚠️ Failed to enable external memory, using integrated mode")
#             model.memory.disable_external_storage()

#         # Cleanup temp file
#         import os
#         if os.path.exists(temp_path):
#             os.remove(temp_path)
#     else:
#         # Ensure integrated mode
#         model.memory.disable_external_storage()
#         logger.info("✅ Using integrated memory mode for evaluation")


# def create_checkpoint_with_separate_memory(model, checkpoint_path: str,
#                                          separate_memory: bool = True):
#     """
#     Save model checkpoint with option to separate episodic memory

#     Args:
#         model: BitMar model to save
#         checkpoint_path: Path to save main checkpoint
#         separate_memory: If True, save memory separately
#     """
#     checkpoint_path = Path(checkpoint_path)

#     # Get model state dict
#     state_dict = model.state_dict()

#     if separate_memory:
#         # Remove memory-related keys from main checkpoint
#         memory_keys = [k for k in state_dict.keys() if k.startswith('memory.')]
#         main_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('memory.')}

#         # Save main model without memory
#         main_checkpoint = {
#             'model_state_dict': main_state_dict,
#             'config': model.config,
#             'separate_memory': True,
#             'memory_info': {
#                 'memory_size': model.memory.memory_size,
#                 'episode_dim': model.memory.episode_dim,
#                 'memory_alpha': model.memory.alpha
#             }
#         }

#         torch.save(main_checkpoint, checkpoint_path)

#         # Save memory separately
#         memory_path = checkpoint_path.with_suffix('.memory.pt')
#         model.memory.save_external_memory(str(memory_path), compress=True)

#         logger.info(f"💾 Checkpoint saved with separate memory:")
#         logger.info(f"   Main model: {checkpoint_path}")
#         logger.info(f"   Memory: {memory_path}")

#         return str(checkpoint_path), str(memory_path)
#     else:
#         # Save everything together (compatibility mode)
#         full_checkpoint = {
#             'model_state_dict': state_dict,
#             'config': model.config,
#             'separate_memory': False
#         }

#         torch.save(full_checkpoint, checkpoint_path)
#         logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")

#         return str(checkpoint_path), None


# def load_checkpoint_with_memory_options(model, checkpoint_path: str,
#                                        memory_path: str = None,
#                                        memory_mode: str = "auto"):
#     """
#     Load model checkpoint with flexible memory loading options

#     Args:
#         model: BitMar model to load into
#         checkpoint_path: Path to main checkpoint
#         memory_path: Optional path to separate memory file
#         memory_mode: "auto", "integrated", or "external"
#     """
#     checkpoint_path = Path(checkpoint_path)

#     # Load main checkpoint
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')

#     # Load main model weights (excluding memory if separate)
#     model.load_state_dict(checkpoint['model_state_dict'], strict=False)

#     # Handle memory loading based on mode
#     if checkpoint.get('separate_memory', False) or memory_path:
#         # Memory was saved separately
#         memory_file = memory_path or checkpoint_path.with_suffix('.memory.pt')

#         if memory_mode == "external":
#             # Load memory in external mode
#             model.memory.enable_external_storage(str(memory_file))
#             success = model.memory.load_external_memory(str(memory_file))
#         else:
#             # Load memory in integrated mode
#             model.memory.disable_external_storage()
#             success = model.memory.load_external_memory(str(memory_file))

#         if success:
#             logger.info(f"✅ Memory loaded from: {memory_file}")
#         else:
#             logger.warning(f"⚠️ Failed to load memory from: {memory_file}")
#     else:
#         # Memory is integrated in the checkpoint
#         logger.info("✅ Using integrated memory from checkpoint")

#     return model
